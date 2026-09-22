import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
import numpy as np
import torch.nn.functional as F

from tqdm import trange, tqdm

import os
import sys

# Ensure parent folder is on sys.path so we can import util from there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project root: ", project_root)
sys.path.insert(0, project_root)

from util import load_model, load_dataset
from attacks import project_patch
from simulation import normalize_yaw as normalize_yaw_t

# source for UNet: https://github.com/jbergq/simple-diffusion-model/


def solve_quadratic(a, b, c):
    """
    Solves ax^2 + bx + c = 0, returning two real roots.
    Returns (None, None) if no real roots exist.
    """
    # Handle degenerate case (a=0 -> linear equation)
    if np.abs(a) < 1e-9:
        if np.abs(b) < 1e-9:
            return (None, None) # No solution
        root = -c / b
        return (root, root)
        
    discriminant = b**2 - 4*a*c
    
    if discriminant < -1e-9: # Allow for small float errors
        # No real roots
        return (None, None)
    
    # Ensure discriminant is non-negative
    sqrt_discriminant = np.sqrt(max(0.0, discriminant))
    
    # Calculate roots
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)
    
    return (root1, root2)

def bb_from_xyz(camera_intrinsic, camera_extrinsic, new_xyz, radius):
    """
    Calculates the 2D bounding box silhouette of a sphere
    centered at new_xyz (in drone frame) with a given RADIUS.
    
    This is the inverse of the xyz_from_bb function.
    """

    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    ox = camera_intrinsic[0, 2]
    oy = camera_intrinsic[1, 2]

    # === Step 1: Transform from World Coords to Camera Coords ===
    # camera_extrinsic is T_c_w (World-to-Camera)
    # print(new_xyz.shape)
    new_xyz_h = np.array([*new_xyz, 1.0])
    xyz_h = camera_extrinsic @ new_xyz_h
    xyz = xyz_h[:3] # 3D point (sphere center) in camera coordinates

    # === Step 2: Get Center Ray (ac) and Distance ===
    distance = np.linalg.norm(xyz)
    
    # Object is behind the camera (z is negative or zero)
    if xyz[2] <= 1e-6:
        print("Error: Object is behind or inside the camera.")
        return None

    # Camera is inside the sphere, silhouette is not defined
    if distance < radius:
        print(f"Error: Camera is inside the object (distance {distance} < radius {radius}).")
        return None

    # 'ac' is the ray to the center, projected onto the z=1 plane
    xc_cam = xyz[0] / xyz[2]
    yc_cam = xyz[1] / xyz[2]
    ac = np.array([xc_cam, yc_cam, 1.0])
    norm_ac_sq = np.dot(ac, ac) # norm(ac)^2

    # === Step 3: Find Tangent Angle ===
    # From geometry: sin(theta/2) = RADIUS / distance
    # We need cos^2(theta/2) = 1 - sin^2(theta/2)
    # This is the squared cosine of the angle between the center ray (ac)
    # and any tangent ray (a_tangent).
    cos_half_theta_sq = 1.0 - (radius / distance)**2
    
    # This is the common 'A' term for our quadratic solvers
    A = norm_ac_sq * cos_half_theta_sq

    # === Step 4: Solve for Horizontal Tangents (x1_cam, x2_cam) ===
    # We solve a quadratic equation for x_cam, given yc_cam
    # (A - xc_cam^2) * x^2 - (2*K1*xc_cam) * x + (A*K1 - K1^2) = 0
    # where K1 = yc_cam^2 + 1.0
    
    K1_x = yc_cam**2 + 1.0
    a_x = A - xc_cam**2
    b_x = -2 * K1_x * xc_cam
    c_x = K1_x * (A - K1_x)
    
    x1_cam, x2_cam = solve_quadratic(a_x, b_x, c_x)
    
    if x1_cam is None:
        print("Error: Could not find horizontal tangent rays.")
        return None

    # === Step 5: Solve for Vertical Tangents (y1_cam, y2_cam) ===
    # We solve a symmetric quadratic equation for y_cam, given xc_cam
    # (A - yc_cam^2) * y^2 - (2*K2*yc_cam) * y + (A*K2 - K2^2) = 0
    # where K2 = xc_cam^2 + 1.0
    
    K1_y = xc_cam**2 + 1.0
    a_y = A - yc_cam**2
    b_y = -2 * K1_y * yc_cam
    c_y = K1_y * (A - K1_y)
    
    y1_cam, y2_cam = solve_quadratic(a_y, b_y, c_y)
    
    if y1_cam is None:
        print("Error: Could not find vertical tangent rays.")
        return None

    # === Step 6: Convert Rays back to Pixel Coordinates ===
    px_1 = x1_cam * fx + ox
    px_2 = x2_cam * fx + ox
    
    py_1 = y1_cam * fy + oy
    py_2 = y2_cam * fy + oy

    # === Step 7: Construct Bounding Box ===
    bb = np.array([
        min(px_1, px_2), # bb[0] = x_min
        min(py_1, py_2), # bb[1] = y_min
        max(px_1, px_2), # bb[2] = x_max
        max(py_1, py_2)  # bb[3] = y_max
    ])
    
    return bb

def construct_T_matrix(sf, tx, ty):
    T_matrix = torch.zeros((3, 3), device=sf.device)
    scale_T = torch.eye(2, device=sf.device) * sf
    T_matrix[:2, :2] = scale_T
    T_matrix[0, 2] = tx
    T_matrix[1, 2] = ty
    T_matrix[2, 2] = 1.0
    return T_matrix

def T_matrix(pose):
    T = torch.eye(4, dtype=torch.float32).to(pose.device)
    normalized_yaw = normalize_yaw_t(pose[3])

    sin_yaw = torch.sin(normalized_yaw).to(torch.float32)
    cos_yaw = torch.cos(normalized_yaw).to(torch.float32)
    zero = torch.zeros_like(sin_yaw, device=pose.device, dtype=torch.float32)

    rotation_matrix_row1 = torch.stack([cos_yaw, -sin_yaw, zero], dim=-1)
    rotation_matrix_row2 = torch.stack([sin_yaw, cos_yaw, zero], dim=-1)
    rotation_matrix_row3 = torch.tensor([0., 0., 1.], device=pose.device, dtype=torch.float32)
    R = torch.stack((rotation_matrix_row1, rotation_matrix_row2, rotation_matrix_row3), dim=0)

    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T


class ConvBlock(nn.Module):
    """Simple convolutional block: Conv2D -> BatchNorm -> Activation."""

    def __init__(self, in_size: int, out_size: int, activation: Callable = nn.ReLU) -> None:
        """Constructs the ConvBlock.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.act = activation(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # print("conv block x shape: ", x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class PositionalEncoding(nn.Module):
    """Transformer sinusoidal positional encoding."""

    def __init__(self, max_time_steps: int, embedding_size: int, device: torch.device, n: int = 10000) -> None:
        """Constructs the PositionalEncoding.

        Args:
            max_time_steps (int): Number of timesteps that can be uniquely represented by encoding.
            embedding_size (int): Size of returned time embedding.
            n (int, optional): User-defined scalar. Defaults to 10000.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.n = n

    def forward(self, t: Tensor) -> Tensor:
        """Returns embedding encoding time step `t`.

        Args:
            t (Tensor): Time step.

        Returns:
            Tensor: Returned position embedding.
        """
        freqs = torch.arange(start=0, end=self.embedding_size//2, dtype=torch.float32, device=t.device)
        freqs = freqs / (self.embedding_size // 2)
        freqs = (1 / self.n) ** freqs
        t = t.ger(freqs.to(t.dtype))
        t = torch.cat([t.cos(), t.sin()], dim=1)
        return t


class TargetEncoding(nn.Module):
    def __init__(self, patch_size: tuple[int, int], target_size: int = 7, embed_channels: int = 16):
        super().__init__()
        self.embed_channels = embed_channels
        self.h, self.w = patch_size
        
        # We map the 7 scalars to a larger vector (e.g. 128)
        self.linear = nn.Linear(target_size, 128) 
        
        # Then project to (embed_channels * h * w)
        # We use a larger kernel or different stride if needed, but 
        # a simple projection is usually fine.
        self.project = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, self.embed_channels * self.h * self.w)
        )

    def forward(self, target: Tensor) -> Tensor:
        b = target.shape[0]
        # Linear projection
        x = self.linear(target)
        # Expand to full image size
        x = self.project(x)
        # Reshape to (Batch, Channels, H, W)
        x = x.view(b, self.embed_channels, self.h, self.w)
        return x

def conv3x3(
    in_size: int,
    out_size: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_size,
        out_size,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_size: int, out_size: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False)


class ResNetBlockUp(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        skip_size: int,
        t_size: Optional[int] = None,
        activation: Callable = nn.SiLU,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.block = ResNetBlock(in_size + skip_size, out_size, activation, t_size=t_size)

    def forward(self, x: Tensor, x_skip: Tensor = None, t_emb: Tensor = None) -> Tensor:
        # Upsample by factor 2, but ensure the upsampled feature map exactly
        # matches the spatial size of the skip connection to avoid off-by-one
        # mismatches caused by odd-sized inputs / integer division.
        x = self.up(x)

        # If a skip connection is present, resize the upsampled tensor to the
        # skip's (height, width) exactly before concatenation. This handles
        # cases where simple scale_factor doubling produces sizes that differ
        # by 1 due to rounding during downsampling.
        if x_skip is not None:
                # debug prints kept intentionally for troubleshooting
                # print("x shape before concat in ResNetBlockUp: ", x.shape)
                # print("x_skip shape before concat in ResNetBlockUp: ", x_skip.shape)
            target_h, target_w = x_skip.shape[-2], x_skip.shape[-1]
            if x.shape[-2] != target_h or x.shape[-1] != target_w:
                # use bilinear interpolation to avoid introducing artifacts in feature maps
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            x = torch.cat([x, x_skip], dim=1)
        out = self.block(x, t_emb)
        return out


class ResNetBlockDown(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, t_size: Optional[int] = None, activation: Callable = nn.SiLU
    ) -> None:
        super().__init__()

        self.block = ResNetBlock(in_size, out_size, activation, stride=2, t_size=t_size)

    def forward(self, x: Tensor, t_emb: Tensor = None) -> Tensor:
        out = self.block(x, t_emb)

        return out


class ResNetBlock(nn.Module):
    """ResNet block with injection of positional encoding."""

    def __init__(
        self, in_size: int, out_size: int, activation: Callable = nn.SiLU, stride: int = 1, t_size: Optional[int] = None
    ) -> None:
        """Constructs the ResNetBlock.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.SiLU.
            stride (int): Stride of first convolutional layer (and skip convolution if in_size != out_size).
            t_size (int): Size of time positional embedding.
        """
        super().__init__()

        self.act = activation(inplace=False)

        self.t_proj = nn.Sequential(self.act, nn.Linear(t_size, out_size)) if t_size is not None else None

        self.conv1 = conv3x3(in_size, out_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = conv3x3(out_size, out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.skip_conv: Optional[nn.Sequential] = None
        if in_size != out_size:
            self.skip_conv = nn.Sequential(conv1x1(in_size, out_size, stride), nn.BatchNorm2d(out_size))

    def forward(self, x: Tensor, t_emb: Tensor = None) -> Tensor:
        x_skip = x

        if self.skip_conv is not None:
            x_skip = self.skip_conv(x_skip)

        # First hidden layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # Inject positional encoding in hidden state.
        if t_emb is not None and self.t_proj is not None:
            t_emb = self.t_proj(t_emb).unsqueeze(2).unsqueeze(2) # shape was (b, 128), now is (b, 128, 1, 1)
            # print(t_emb.shape)
            # print(x.shape)
            x = t_emb + x #rearrange(t_emb, "b c -> b c 1 1") + x

        # Second hidden layer.
        x = self.conv2(x)
        x = self.bn2(x)

        # Residual connection + activation.
        x += x_skip
        out = self.act(x)

        return out

class UNet(nn.Module):
    """UNet with ResNet blocks and injection of positional encoding."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
        num_layers: int = 5,
        features_start: int = 128,
        t_emb_channels: int = 16, # This controls how "loud" the target is
        t_emb_size: int = 512,
        max_time_steps: int = 1000,
        patch_size: tuple[int, int] = (45, 80),
        target_emb : bool = True
    ) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
            PositionalEncoding(max_time_steps, t_emb_size, device), nn.Linear(t_emb_size, t_emb_size)
        )

        if target_emb:
            self.target_embedding = TargetEncoding(patch_size, target_size=7, embed_channels=t_emb_channels)

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        self.num_layers = num_layers

        # --- THE FIX IS HERE ---
        if target_emb:
            # We must account for the extra channels from the target embedding
            # Input channels = image channels (in_size) + target embedding channels (t_emb_channels)
            input_channels = in_size + t_emb_channels 
            self.conv_in = nn.Sequential(
                ConvBlock(input_channels, features_start), 
                ConvBlock(features_start, features_start)
            )
        else:
            self.conv_in = nn.Sequential(
                ConvBlock(in_size, features_start), 
                ConvBlock(features_start, features_start)
            )
        # -----------------------

        # Create encoder and decoder stages.
        layers = []
        feats = features_start
        for _ in range(num_layers - 1):  # Encoder
            layers.append(ResNetBlockDown(feats, feats * 2, t_size=t_emb_size))
            feats *= 2
        for _ in range(num_layers - 1):  # Decoder
            layers.append(ResNetBlockUp(feats, feats // 2, skip_size=feats // 2, t_size=t_emb_size))
            feats //= 2
        self.layers = nn.ModuleList(layers)

        self.conv_out = nn.Conv2d(feats, out_size, kernel_size=1)

    def forward(self, x: Tensor, target: Tensor, t: Tensor) -> Tensor:
        t_emb = self.t_embedding(t.flatten()) 
        target_emb = self.target_embedding(target)
        
        # This concatenation creates (B, 1+16, H, W) = (B, 17, H, W)
        x = torch.concat((x, target_emb), dim=1)

        x = self.conv_in(x)

        x_i = [x]

        # Encoder stage.
        for layer in self.layers[: self.num_layers - 1]:
            x_i.append(layer(x=x_i[-1], t_emb=t_emb))

        # Decoder stage.
        for i, layer in enumerate(self.layers[self.num_layers - 1 :]):
            x_i[-1] = layer(x=x_i[-1], x_skip=x_i[-2 - i], t_emb=t_emb)

        out = self.conv_out(x_i[-1])

        return out

class DiffusionModel():
    def __init__(self, device, in_size=1, out_size=1, lr=3e-3, patch_size=(45, 80), prediction_model_name='frontnet'):
        self.in_size = in_size    # number of channels (1 -> grayscale)
        self.out_size = out_size  # number of channels
        self.lr = lr
        self.sigma_data = 0.5

        self.device = device

        # store the patch size used by the diffusion model (height, width)
        self.patch_size = patch_size
        self.model = UNet(in_size=self.in_size, out_size=self.out_size, num_layers=3, patch_size=self.patch_size, device=self.device).to(device)

        # self.prediction_model_name = prediction_model_name
        # if self.prediction_model_name == 'yolov5':
        #     from yolo_bounding import YOLOBox
        #     self.prediction_model = YOLOBox()
        #     self.batch_size = 8
        
        # elif self.prediction_model_name == 'frontnet':
        #     self.prediction_model = load_model(f"{project_root}/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
        #     self.prediction_model.eval()
        #     self.batch_size = 128

        # for param in self.prediction_model.parameters():
        #     param.requires_grad = False

        # self.train_set = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)
        # self.test_set = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = 1, train=False, train_set_size=0.9, IMRC=True)

    def denoised_prediction(self, x, conditioning, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, conditioning, c_noise)
        D_x = c_skip * x + c_out * F_x
        D_x = torch.tanh(D_x)  # Ensure output is in [-1, 1]
        return D_x
    
    def train(self, patch, target, conditioning, device: torch.device, nepochs: int = 10):
        P_mean = -1.2
        P_std = 1.2
        p_unconditioned = 0.1

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1., total_iters=1000)
        all_reco_losses = []
        all_prediction_losses = []
        all_losses = []

        reco_losses = []
        prediction_losses = []
        losses = []
        adv_losses = []
        tv_losses = []

        # batch_size = 64
        n_samples = 1

        patch = torch.tensor(patch, dtype=torch.float32).to(device).reshape(1, 1, self.patch_size[0], self.patch_size[1])
        target_t = torch.tensor(target, dtype=torch.float32).to(device).reshape(1, -1)
        conditioning = torch.tensor([*conditioning, *target], dtype=torch.float32).to(device).reshape(1, -1)

        print("patch shape before scaling: ", patch.shape)
        print("target shape before scaling: ", target_t.shape)
        print("conditioning shape before scaling: ", conditioning.shape)

        patch = (2 * patch) - 1  # Scale to [-1, 1]

        for epoch in trange(nepochs):

            #for imgs, _ in tqdm(self.train_set):
            # for [patches, conditioning] in data_loader:
            #     patches = (2 * patches.to(device)) - 1
            #     conditioning = conditioning.to(device)
            optimizer.zero_grad()

            # patches = (2 * patches.to(device)) - 1
            # conditioning = conditioning.to(device)
            # optimizer.zero_grad()
            sigmas = P_std * torch.randn(patch.shape[0], 1, 1, 1, device=patch.device) + P_mean
            sigmas = sigmas.exp()

            noise = torch.randn_like(patch) * sigmas # Sample DIFFERENT random noise for each datapoint
            
            model_in = patch + noise # Noise corrupt the data 
            out = self.denoised_prediction(model_in, conditioning, sigmas)
            weight = (sigmas ** 2 + self.sigma_data ** 2) / (sigmas * self.sigma_data) ** 2
            reconstruction_loss = torch.mean(weight * (patch - out)**2) # Compute loss on prediction
            reco_losses.append(reconstruction_loss.detach().cpu().numpy())
            all_reco_losses.append(reconstruction_loss.detach().cpu().numpy())

            

                # sf = np.random.uniform(0.5, 1.1, 1,)
                # if self.prediction_model_name == 'frontnet':
                #     tx = np.random.uniform(-20., 140., 1,)
                #     ty = np.random.uniform(-10., 80., 1,)
                # elif self.prediction_model_name == 'yolov5':
                #     tx = np.random.uniform(-20., 630., 1,)
                #     ty = np.random.uniform(-10., 310., 1,)
                # x = np.random.uniform(0., 1.5, 1,)
                # y = np.random.uniform(-1, 1, 1,)
                # z = np.random.uniform(-0.5, 0.5, 1,)
                # yaw = np.random.uniform(-0.3, 0.3, 1,)

                # # print("conditioning random shape: ", (sf.shape, tx.shape, ty.shape, x.shape, y.shape, z.shape, yaw.shape))

                # conditioning_random = torch.tensor(np.stack((sf, tx, ty, x, y, z, yaw)).T, dtype=torch.float32).to(device)
                # sampled_patches = self.sample(n_samples=1, targets=conditioning_random, device=device, n_steps=2).to(device)
                # # print("sampled_patches shape: ", sampled_patches.shape, " min: ", torch.min(sampled_patches), " max: ", torch.max(sampled_patches))

                # # print("DEBUGGING")
                # # print("positions shape: ", positions.shape)
                # # print("example position: ", positions[0])
                # T = construct_T_matrix(conditioning_random[0, 0], conditioning_random[0, 1], conditioning_random[0, 2]).unsqueeze(0).to(device)
                #     # print("T_matrices shape: ", T_matrices.shape)
                # # T_matrices = torch.stack([construct_T_matrix(s, tx, ty) for s, tx, ty in zip(conditioning_random[:, 0], conditioning_random[:, 1], conditioning_random[:, 2])]).to(device)



                
                # # --- 2. THE FIX: Sample Random Noise Levels (Sigmas) ---
                # # Instead of running the full loop, we pick random points in the diffusion process
                # rnd_normal = torch.randn([n_samples, 1, 1, 1], device=device)
                # sigmas = (rnd_normal * P_std + P_mean).exp()
                
                # # Create noisy inputs (Latent state at time t)
                # # Since we want to generate from scratch, we treat 'x_0' as pure noise initially
                # # effectively asking: "If the image looks like this noise, what should the result be?"
                # noise_latents = torch.randn(n_samples, 1, *self.patch_size, device=device) * sigmas

                # # --- 3. Single-Step Denoised Prediction ---
                # # We ask the model: "What is the final adversarial patch (x_0) hidden in this noise?"
                # # This backprops through only 1 UNet pass, not 25.
                # denoised_patches = self.denoised_prediction(noise_latents, conditioning_random, sigmas)
                
                # # Clamp to valid image range
                # denoised_patches = denoised_patches.clamp(0., 1.)
                # denoised_patches = denoised_patches.repeat_interleave(self.train_set.batch_size, dim=0)
                
                # conditioning_random = conditioning_random.repeat_interleave(self.train_set.batch_size, dim=0)
                
                # #sampled_patches = sampled_patches.repeat_interleave(self.train_set.batch_size, dim=0)
                # T_matrices = T.repeat_interleave(self.train_set.batch_size, dim=0)


                # # print("conditioning_random shape after repeat: ", conditioning_random.shape)
                # # print("sampled_patches shape after repeat: ", sampled_patches.shape)
                # # print("T_matrices shape after repeat: ", T_matrices.shape)

                # target_yaw_v = conditioning_random[:, -1] # of shape [n_samples,]
                # # target_yaw_v = target_yaw_v.repeat_interleave(self.train_set.batch_size, dim=0)
                # #target_yaw_v = target_yaw_noisy
                # # print("target_yaw_v shape: ", target_yaw_v.shape)
                # # print("example target_yaw_v: ", target_yaw_v[0])

                # # conditioning_random = conditioning_random.repeat_interleave(self.train_set.batch_size, dim=0)


                # imgs = imgs / 255.

                # if self.prediction_model_name == 'yolov5':
                #     imgs = torch.nn.functional.interpolate(imgs, size=(320, 640), mode='bilinear', align_corners=False)
                # # imgs = next(iter(self.train_set))[0].to(device) / 255.0  # normalize to [0, 1]
                # # print("imgs shape: ", imgs.shape, " min: ", torch.min(imgs), " max: ", torch.max(imgs))

                # # appy all sampled patches to all images in the batch
                # # manipulated_images_all = []
                # # for i in range(imgs.shape[0]):
                # manipulated_images = project_patch(
                #     patches=denoised_patches,
                #     T_matrices=T_matrices,
                #     images=imgs
                # )
                # # manipulated_images_all.append(manipulated_images)

                # # manipulated_images = torch.stack(manipulated_images_all).permute(1, 0, 2, 3, 4)

                # # add random noise to manipulated images for robustness
                # # manipulated_images = manipulated_images + torch.randn_like(manipulated_images) * 0.1

                # manipulated_images.clamp_(0., 1.)

                # # print("manipulated_images shape: ", manipulated_images.shape, " min: ", torch.min(manipulated_images), " max: ", torch.max(manipulated_images))

                # # loss_per_sample = []
                # # for image_batch in manipulated_images:
                # if self.prediction_model_name == 'frontnet':
                #     x, y, z, yaw = self.prediction_model(manipulated_images*255.)
                #     # print("x, y, z, yaw:", x, y, z, yaw)
                #     prediction = torch.stack([x, y, z, yaw])
                #     prediction = prediction.squeeze(2).mT
                # elif self.prediction_model_name == 'yolov5':
                   
                #     manipulated_images = manipulated_images.repeat_interleave(3, dim=1)
                #     # target_xyz = conditioning_random[:, 3:6].cpu().numpy()
                #     # # batch target_box
                #     # # try:
                #     # #     target_boxes = torch.stack([torch.tensor([bb_from_xyz(self.prediction_model.cam.camera_intrinsic, self.prediction_model.cam.camera_extrinsic, xyz, self.prediction_model.cam.radius.cpu().item())], device=device, dtype=torch.float32) for xyz in target_xyz], dim=0)
                #     # # except TypeError:
                #     # target_boxes = None
                #     # while target_boxes is None:
                #     #     try:
                #     #         target_boxes = torch.stack([torch.tensor([bb_from_xyz(self.prediction_model.cam.camera_intrinsic, self.prediction_model.cam.camera_extrinsic, xyz, self.prediction_model.cam.radius.cpu().item())], device=device, dtype=torch.float32) for xyz in target_xyz], dim=0)
                #     #     except TypeError:
                #     #         x = np.random.uniform(0., 1.5, 1,)
                #     #         y = np.random.uniform(-1, 1, 1,)
                #     #         z = np.random.uniform(-0.5, 0.5, 1,)
                #     #         yaw = np.random.uniform(-0.3, 0.3, 1,)

                #     #         conditioning_random = torch.tensor(np.stack((sf, tx, ty, x, y, z, yaw)).T, dtype=torch.float32).to(device)
                #     #         target_xyz = conditioning_random[:, 3:6].cpu().numpy()

                #     # # target_box = torch.tensor([bb_from_xyz(self.prediction_model.cam.camera_intrinsic, self.prediction_model.cam.camera_extrinsic, target_xyz, self.prediction_model.cam.radius)], device=device, dtype=torch.float32)
                #     # # scale to image of size 320 x 640
                #     # scaled_target_boxes = target_boxes.clone().squeeze(1)
                #     # scaled_target_boxes[:, [0, 2]] *= (640.0 / 160.)  # x coords
                #     # scaled_target_boxes[:, [1, 3]] *= (320.0 / 96.)   # y coords

                #     # # predicted_boxes_list = []
                #     # # predicted_scores_list = []
                #     # loss_boxes = 0.
                #     # loss_confidence = 0.
                #     # for scaled_box, manipulated_images, condi in zip(scaled_target_boxes, manipulated_images, conditioning_random):
                #     #     # print("scaled_box: ", scaled_box)
                #     #     # print("manipulated_image shape before resize: ", manipulated_image.shape)
                #     #     manipulated_image_resized = torch.nn.functional.interpolate(manipulated_images.unsqueeze(0), size=(320, 640), mode='bilinear', align_corners=False)
                #     #     # print("manipulated_image shape after resize: ", manipulated_image_resized.shape
                        


                #     #     center_target_box = torch.tensor([(scaled_box[0] + scaled_box[2]) / 2,
                #     #                                 (scaled_box[1] + scaled_box[3]) / 2], device=device)
                #     #     predicted_boxes, predicted_scores = self.prediction_model(manipulated_image_resized, target_anchor=center_target_box)  # yolo expects images in range [0, 1], out (B, 4)
                #     #     loss_confidence += F.mse_loss(predicted_scores, torch.ones_like(predicted_scores).to(device))
                #     #     if predicted_boxes.shape[0] > 1:
                #     #         # choose box with highest confidence
                #     #         scores = F.softmax(predicted_scores, dim=0)
                #     #         predicted_box = (predicted_boxes * scores.unsqueeze(-1)).sum(dim=0, keepdim=True)
                #     #         # predicted_scores = scores.sum(dim=0, keepdim=True)
                #     #     else:
                #     #         predicted_box = predicted_boxes
                #     #     # predicted_boxes_list.append(predicted_box)
                #     #     # predicted_scores_list.append(predicted_scores)
                #     #     loss_boxes += F.mse_loss(predicted_box, scaled_box.unsqueeze(0))

                #     # loss_boxes = loss_boxes / manipulated_images.shape[0]
                #     # loss_confidence = loss_confidence / manipulated_images.shape[0]

                #     actual_predictions = []
                #     for condi in conditioning_random:
                #         predicted_boxes, predicted_scores = self.prediction_model(manipulated_images, target_anchor=None)
                    
                #         if predicted_boxes.shape[0] > 1:
                #             # choose box with highest confidence
                #             scores = F.softmax(predicted_scores, dim=0)
                #             predicted_box = (predicted_boxes * scores.unsqueeze(-1)).sum(dim=0, keepdim=True)
                #             # predicted_scores = scores.sum(dim=0, keepdim=True)
                #         else:
                #             predicted_box = predicted_boxes

                #         scaled_box = predicted_box.clone()
                #         # scale back to 160x96
                #         scaled_box[:, [0, 2]] *= (160.0 / 640.)
                #         scaled_box[:, [1, 3]] *= (96.0 / 320.)   # y coords

                #         recovered_xyzyaw = self.prediction_model.cam.batch_xyz_from_boxes(scaled_box)
                #         actual_predictions.append(recovered_xyzyaw[0]) # prediction values
                #         # # print("recovered_xyzyaw: ", recovered_xyzyaw, recovered_xyzyaw.shape)
                #         # T_rec_pred_in_drone = T_matrix(recovered_xyzyaw[0])
                #         # # print("T_rec_pred_in_drone: ", T_rec_pred_in_drone)
                #         # print("condi: ", condi)
                #         # print(condi[-4:])
                #         # print(condi[3:6])
                #         # T_rec_pred_in_world = T_rec_pred_in_drone @ T_matrix(condi[-4:])
                #         # target_yaw = normalize_yaw_t(recovered_xyzyaw[0, 3])
                #         # T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
                #         # T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
                #         # T_setpoint_world = T_direction_world @ T_rec_pred_in_world
                #         # setpoint_yaw = torch.zeros_like(target_yaw).to(device)

                #         # actual_predictions.append(torch.stack([*T_setpoint_world[:3, 3], setpoint_yaw]).to(device)) # prediction values
                #     prediction = torch.stack(actual_predictions, dim=0)
                #     # predicted_boxes, predicted_scores = model(manipulated_image_resized, target_anchor=center_target_box)  # yolo expects images in range [0, 1], out (B, 4)
                #     # loss_confidence = F.mse_loss(predicted_scores, torch.ones_like(predicted_scores).to(device))

                #     # if predicted_boxes.shape[0] > 1:
                #     #     # choose box with highest confidence
                #     #     scores = F.softmax(predicted_scores, dim=0)
                #     #     predicted_boxes = (predicted_boxes * scores.unsqueeze(-1)).sum(dim=0, keepdim=True)
                #     #     # predicted_scores = scores.sum(dim=0, keepdim=True)

                    
                #     # loss_boxes = F.mse_loss(predicted_boxes, scaled_target_box)

                #     # predicted_boxes, _ = model(manipulated_image_resized, target_anchor=None)
                #     # # print(predicted_boxes, predicted_boxes.shape)
                #     # scaled_box = predicted_boxes.clone()
                #     # # scale back to 160x96
                #     # scaled_box[:, [0, 2]] *= (160.0 / 640.)  # x coords
                #     # scaled_box[:, [1, 3]] *= (96.0 / 320.)   # y coords

                #     # recovered_xyzyaw = self.prediction_model.cam.batch_xyz_from_boxes(scaled_box)
                #     # T_rec_pred_in_drone = T_matrix(recovered_xyzyaw[0])
                #     # T_rec_pred_in_world = T_drone_in_world @ T_rec_pred_in_drone
                #     # target_yaw = normalize_yaw_t(recovered_xyzyaw[0, 3])
                #     # T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
                #     # T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
                #     # # print("Direction in world within loop:")
                #     # # print(T_direction_world)

                #     # T_setpoint_world = T_direction_world @ T_rec_pred_in_world
                #     # setpoint_yaw = torch.zeros_like(target_yaw).to(device)
                #     # #yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
                #     # #setpoint_yaw = normalize_yaw_t(yaw)

                #     # prediction = torch.stack([*T_setpoint_world[:3, 3], setpoint_yaw]).to(device) # prediction values

                    
                #     # # print("prediction from yolov5: ", prediction[0])

                #     # #scale back to 160x96
                #     # prediction[:, [0, 2]] *= (160.0 / 640.0)  # x coords
                #     # prediction[:, [1, 3]] *= (96.0 / 320.0)   # y coords

                #     # # print("YOLOv5 prediction before scaling to 160x96:", prediction)
                #     # # print("Scaled bounding box for 160x96 image: ", prediction)

                #     # prediction = self.prediction_model.cam.batch_xyz_from_boxes(scaled_box) #  xyzyaw from bounding box

                # # print("prediction shape: ", prediction.shape, " example prediction: ", prediction[0])
                # # print("target shape: ", conditioning_random[:, 3:6].shape, " example target: ", conditioning_random[0, 3:6])
                # yaw_v = prediction[:, 3]

                # target_xyz = conditioning_random[:, 3:6]
                # target_yaw = conditioning_random[:, -1]


                # mse_losses = F.mse_loss(prediction[:, :3], target_xyz, reduction='none').mean(dim=1)
                # angular_losses = 1 - torch.cos(normalize_yaw_t(prediction[:, 3]) - normalize_yaw_t(target_yaw))
                
                # # Total Adversarial Loss
                # adv_loss = mse_losses + angular_losses

                # # --- 6. Total Variation Loss (Smoothing) ---
                # # Pure adversarial optimization creates high-frequency noise. 
                # # TV loss forces the patch to be smooth/natural.
                # # tv_loss = torch.sum(torch.abs(denoised_patches[:, :, :, :-1] - denoised_patches[:, :, :, 1:])) + \
                # #           torch.sum(torch.abs(denoised_patches[:, :, :-1, :] - denoised_patches[:, :, 1:, :]))
                # # tv_loss = tv_loss / (n_samples * self.patch_size[0] * self.patch_size[1])

                # # Weight the loss by Sigma? 
                # # We want the model to be accurate at Low noise (sigma small) and High noise (sigma large).
                # # Usually we weight high noise less because it's harder.
                # weight = (sigmas ** 2 + self.sigma_data ** 2) / (sigmas * self.sigma_data) ** 2
                # weighted_adv_loss = adv_loss * weight.flatten()
                
                # # Combine
                # # loss = (adv_loss * weight.flatten()).mean() #+ (0.1 * tv_loss)


                # # # loss_per_sample.append(prediction_loss.mean())
                # # sorted_prediction_losses = torch.sort(prediction_loss, descending=True)[0]
                # # top5_prediction_losses = sorted_prediction_losses[:5]
                # # prediction_losses.append(torch.mean(prediction_loss).detach().cpu().numpy())
                # # all_prediction_losses.append(torch.mean(prediction_loss).detach().cpu().numpy())
                # adv_losses.append(adv_loss.mean().detach().cpu().numpy())
                # # tv_losses.append(tv_loss.detach().cpu().numpy())
                # sorted_adv_losses = torch.sort(weighted_adv_loss, descending=True)[0]
                # top5_adv_losses = sorted_adv_losses[:5]
                # loss = top5_adv_losses.mean()
                


                # loss = top5_prediction_losses.mean()#prediction_loss.mean()
                # loss = combined_loss
            loss = reconstruction_loss
            losses.append(loss.detach().cpu().numpy())
            all_losses.append(loss.detach().cpu().numpy())
            

            # Bwd pass
            loss.backward()
            optimizer.step()
            scheduler.step()

                # mean_losses = np.mean(np.array(losses)[-10:])
                # print(f"Epoch {epoch+1} mean loss over last 10 batches: {mean_losses}")

            if (epoch+1) % 1 == 0:
                mean_loss = np.mean(np.array(losses))
                # mean_reco_loss = np.mean(np.array(reco_losses))
                # mean_prediction_loss = np.mean(np.array(prediction_losses))
                mean_adv_loss = np.mean(np.array(adv_losses))
                # mean_tv_loss = np.mean(np.array(tv_losses))
                adv_losses = []
                tv_losses = []
                losses = []
                reco_losses = []
                prediction_losses = []
                print("Epoch %d,\t Loss %f \t Adv Loss %f" % (epoch+1, mean_loss, mean_adv_loss)) #mean_tv_loss))
                # print("Epoch %d,\t Loss %f \t Reconstruction Loss %f \t Prediction Loss %f" % (epoch+1, mean_loss, mean_reco_loss, mean_prediction_loss))
                
                
                # save some of the manipulated images for visualization
                # os.makedirs(f'results/diffusion_training/epoch_images/epoch{epoch+1}', exist_ok=True)
                # for i in range(min(5, manipulated_images.shape[0])):
                #     img = manipulated_images[i].detach().to('cpu').numpy()
                #     fig, ax = plt.subplots(1, 1)
                #     ax.imshow(img.transpose(1, 2, 0), cmap='gray')
                #     ax.set_title(f'epoch_{epoch+1}_img_{i}')
                #     ax.set_axis_off()
                #     fig.savefig(f'results/diffusion_training/epoch_images/epoch{epoch+1}/img_{i}.png', dpi=200)
                #     plt.close(fig)

            if (epoch+1) % 1 == 0:
                print("Saving checkpoint...")
                os.makedirs(f'diffusion/results/diffusion_training/{self.prediction_model_name}/{0}/checkpoints/', exist_ok=True)
                model.save(f'diffusion/results/diffusion_training/{self.prediction_model_name}/{0}/checkpoints/checkpoint_epoch_{epoch+1}.pth')

                test_sample_losses = self.eval(num_samples=10, data_loader=self.test_set, device=device)
                print(f"Test sample losses at epoch {epoch+1}: {test_sample_losses}, mean: {np.mean(test_sample_losses)}")

        return all_losses

    @torch.no_grad()
    def eval(self, num_samples: int, data_loader: torch.utils.data.DataLoader, device: torch.device):
        self.model.eval()
        sf = np.random.uniform(0.5, 1.1, num_samples)
        if self.prediction_model_name == 'frontnet':
            tx = np.random.uniform(-20., 140., num_samples)
            ty = np.random.uniform(-10., 80., num_samples)
        elif self.prediction_model_name == 'yolov5':
            tx = np.random.uniform(-20., 630., num_samples)
            ty = np.random.uniform(-10., 310., num_samples)
        x = np.random.uniform(0., 1.5, num_samples)
        y = np.random.uniform(-1, 1, num_samples)
        z = np.random.uniform(-0.5, 0.5, num_samples)
        yaw = np.random.uniform(-0.3, 0.3, num_samples)

        r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z, yaw)).T, dtype=torch.float32).to(device)
        test_samples = self.sample(num_samples, r_targets, device, n_steps=25).to(device)

        test_dataset_losses = []
        for i, (test_patch, conditioning) in enumerate(zip(test_samples, r_targets)):
            test_patch = test_patch.unsqueeze(0)  # add batch dimension
            # print("Test patch shape: ", test_patch.shape)
            conditioning = conditioning  # add batch dimension
            T = construct_T_matrix(conditioning[0], conditioning[1], conditioning[2]).unsqueeze(0).to(device)
            batch_patch_losses = []
            for imgs in data_loader:
                imgs = imgs[0].to(device) / 255.0  # normalize to [0, 1]

                if self.prediction_model_name == 'yolov5':

                    imgs = torch.nn.functional.interpolate(imgs, size=(320, 640), mode='bilinear', align_corners=False)
                              
                manipulated_image = project_patch(
                    patches=test_patch,
                    T_matrices=T,
                    images=imgs
                )
                manipulated_image.clamp_(0., 1.)

                if self.prediction_model_name == 'frontnet':
                    x_pred, y_pred, z_pred, yaw_pred = self.prediction_model(manipulated_image*255.)
                    prediction = torch.stack([x_pred, y_pred, z_pred, yaw_pred])
                    prediction = prediction.squeeze(2).mT
                elif self.prediction_model_name == 'yolov5':
                   
                    
                    # gray to rgb
                    manipulated_image = manipulated_image.repeat_interleave(3, dim=1)

                    prediction = self.prediction_model(manipulated_image)  # yolo expects images in range [0, 1]
                    prediction[:, [0, 2]] *= (160.0 / 640.0)  # x coords
                    prediction[:, [1, 3]] *= (96.0 / 320.0)   # y coords

                    # print("YOLOv5 prediction before scaling to 160x96:", prediction)
                    # print("Scaled bounding box for 160x96 image: ", prediction)

                    prediction = self.prediction_model.cam.batch_xyz_from_boxes(prediction) #  xyzyaw from bounding box
                    yaw_pred = prediction[:, 3]
                
                target = conditioning[-4:].unsqueeze(0)  # last 4 values are the target x, y, z, yaw
                mse_losses = torch.stack([F.mse_loss(tar, pre) for tar, pre in zip(target[:, :3], prediction[:, :3])]) # calc mse for each of the predictions of each patch
                angular_losses = 1 - torch.cos(normalize_yaw_t(yaw_pred) - normalize_yaw_t(target[:, 3]))  # angular loss for yaw

                batch_loss = torch.mean(mse_losses + angular_losses)
                batch_patch_losses.append(batch_loss.detach().cpu().numpy())
            mean_patch_loss = np.mean(np.array(batch_patch_losses), axis=0)  # mean over all images in data_loader
            test_dataset_losses.append(mean_patch_loss)
        self.model.train()
        return np.array(test_dataset_losses)  # shape (num_samples,)



    # @torch.no_grad()
    def sample(self, n_samples: int, targets: torch.tensor, device: torch.device, patch_size: tuple[int, int] | None = None, n_steps: int=1_000):
        """Alg 2 from the DDPM paper."""
        # self.model.eval()
        # make sure that targets is a tensor
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        # and of shape (n_samples, 3)
        if len(targets.shape) == 1 or targets.shape[0] == 1:
            targets = targets.repeat(n_samples, 1)

        sigma_max = 80
        sigma_min = 0.002
        rho = 7

        # use provided patch_size or fall back to model's configured patch size
        if patch_size is None:
            patch_size = self.patch_size

        x_t = torch.randn((n_samples, 1, *patch_size)).to(device)
        targets = targets.to(device)
        step_indices = torch.arange(n_steps, dtype=torch.float64, device=device)
        sigmas = (sigma_max ** (1 / rho) + step_indices / (n_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
        sigmas = sigmas.float().to(device)
        x_t = torch.randn(n_samples, 1, *patch_size).to(device) * sigmas[0]
        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas, sigmas[1:])):
            model_prediction = self.denoised_prediction(x_t, targets, sigma_cur)
            d_cur = (x_t - model_prediction) / sigma_cur
            x_t = x_t + (sigma_next - sigma_cur) * d_cur

        # x_t is now [-1, 1], convert to [0, 1]
        x_t = torch.stack([(x + 1) / 2 for x in x_t]).clamp(min=0,max=1) # normalize
        #x_t is somewhere between [-1, 1], first normalize to [-1, 1], then to [0, 1], and clamp to [0, 1]
        # x_t = torch.stack([(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in x_t]).clamp(min=0, max=1)
        return x_t


    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    
if __name__ == '__main__':

    import pickle
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet', help='Prediction model to use for training.')
    parser.add_argument('--datasets', type=str, nargs='+', help='Paths to the dataset pickle files.')
    parser.add_argument('--epochs', type=int, default=1_000, help='Number of epochs to train.')
    parser.add_argument('--output', type=str, default='trained_model.pth', help='Path to save the model.')
    parser.add_argument('--corpus_size', type=int, default=1000, help='Size of the training corpus.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.autograd.set_detect_anomaly(True)

    data = np.load("temp_pid_364256/sample_4.npz", allow_pickle=True)
    # print(data)
    
    print("Keys in the loaded data:", data.keys())
    patch = data['patch']  # shape (45, 80)
    target = data['target']  # shape (4,) -> x, y, z, yaw
    condition = data['condition']  # shape (3, ) -> sf, tx, ty

    print("patch shape: ", patch.shape)
    print("target shape: ", target.shape)
    print("condition shape: ", condition.shape)
    print("target: ", target)
    print("condition: ", condition)

    # data = []
    # for dataset_path in args.datasets:
    #     with open(dataset_path, 'rb') as f:
    #         data += pickle.load(f)


    # patches = []
    # targets = []
    # positions = []

    # for i in range(len(data)):
    #     patches.append(data[i][0])
    #     targets.append(data[i][1])
    #     positions.append(data[i][2])

    
    # patches = np.array(patches) # shape (N, 45, 80)
    # targets = np.array(targets) # shape (N, 1, 4) -> x, y, z, yaw, x in range [0, 1.5], [-1, 1], [-0.5, 0.5], [-0.3, 0.3]
    # positions = np.array(positions) # shape (N, 1, 3), sf in range [0.31, 0.99], tx, ty in range [0, 85]



    # print("DEBUGGING")
    # print("patches shape: ", patches.shape)
    # print("targets shape: ", targets.shape)
    # print("positions shape: ", positions.shape)

    # print("targets[0]: ", targets[0])
    # print("positions[0]: ", positions[0])

    if args.model == 'frontnet':
        patch_size = (45, 80)
    elif args.model == 'yolov5':
        patch_size = (80, 320)

    # patches = np.array([(patch - np.min(patch)) / (np.max(patch) - np.min(patch)) for patch in patches]) # normalize
    # patches = torch.tensor(patches).unsqueeze(1)
    # targets = torch.tensor(targets, dtype=torch.float).squeeze(1)
    # positions = torch.tensor(positions, dtype=torch.float).squeeze(1)

    # conditioning = torch.cat((positions, targets), dim=1)

    # print(patches.shape)

    # Define dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = torch.utils.data.TensorDataset(patches, conditioning)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # model = UNet(in_size=1, out_size=1, device=device)
    # model.to(device)

    model = DiffusionModel(device, lr=1e-5, patch_size=patch_size, prediction_model_name=args.model)

    # training
    # print("Start training..")
    # model.load(f'results/diffusion_training/{args.output}')
    all_losses = model.train(patch, target, condition, device, nepochs=args.epochs)

    os.makedirs(f'results/diffusion_training/{args.model}/{100}', exist_ok=True)
    model.save(f'results/diffusion_training/{args.model}/{100}/{args.output}')
    
    n_samples = 5
    sf = np.random.uniform(0.5, 1.1, n_samples)
    if args.model == 'frontnet':
        tx = np.random.uniform(-20., 140., n_samples)
        ty = np.random.uniform(-10., 80., n_samples)
    elif args.model == 'yolov5':
        tx = np.random.uniform(-20., 630., n_samples)
        ty = np.random.uniform(-10., 310., n_samples)
    x = np.random.uniform(0., 1.5, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    z = np.random.uniform(-0.5, 0.5, n_samples)
    yaw = np.random.uniform(-0.3, 0.3, n_samples)

    r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z, yaw)).T, dtype=torch.float32)

    samples = model.sample(n_samples, r_targets, device, patch_size=patch_size, n_steps=25).detach().to('cpu').numpy()
    

    fig = plt.figure(constrained_layout=True, figsize=(n_samples*5, 5))
    axs_samples = fig.subplots(1, n_samples)
    for i, sample in enumerate(samples):
        ax = axs_samples[i] if n_samples > 1 else axs_samples
        ax.imshow(sample[0], cmap='gray')
        ax.set_title(f'sample {i}')
        ax.set_axis_off()
    fig.savefig(f'results/diffusion_training/{args.model}/{100}/samples1.png', dpi=200)
    plt.show()