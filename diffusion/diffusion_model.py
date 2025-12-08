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
from attack_minimal_single import project_patch, normalize_yaw_t

# source for UNet: https://github.com/jbergq/simple-diffusion-model/

def construct_T_matrix(sf, tx, ty):
    T_matrix = torch.zeros((3, 3), device=sf.device)
    scale_T = torch.eye(2, device=sf.device) * sf
    T_matrix[:2, :2] = scale_T
    T_matrix[0, 2] = tx
    T_matrix[1, 2] = ty
    T_matrix[2, 2] = 1.0
    return T_matrix


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
    def __init__(self, patch_size: tuple[int, int], target_size: int = 7, embed_channels: int = 1):
        super().__init__()

        # the whole purpose of this is to learn to encode the target 
        # and bring it into a shape that is easy to concatenate with the
        # image before processing it

        self.embed_channels = embed_channels
        # ensure patch_size ordering is (height, width)
        self.h, self.w = patch_size
        self.linear = nn.Linear(target_size, 512)   # 4 for target only, 7 for target+position
        # convolution that projects the linear embedding to a feature map
        # padding chosen to keep reasonable receptive field for non-square patches
        # Project the linear embedding (b, 512, 1, 1) into a feature map of shape
        # (b, embed_channels, h, w). Using ConvTranspose2d with kernel_size=(h,w)
        # maps a 1x1 spatial input to an h x w output.
        self.conv = nn.ConvTranspose2d(512, self.embed_channels, kernel_size=(self.h, self.w), stride=1, padding=0)

    def forward(self, target: Tensor) -> Tensor:
        # print("DEBUGGING")
        # print("target shape in TargetEncoding: ", target.shape)
        out = self.linear(target).unsqueeze(2).unsqueeze(2)
        out = self.conv(out).view(target.shape[0], self.embed_channels, self.h, self.w)
        return out

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
        features_start: int = 64,
        t_emb_size: int = 512,
        max_time_steps: int = 1000,
        patch_size: tuple[int, int] = (45, 80),
        target_emb : bool = True
    ) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
            PositionalEncoding(max_time_steps, t_emb_size, device), nn.Linear(t_emb_size, t_emb_size)
        )

        # default target embedding uses the project's patch size (45,80)
        # keep flexibility to pass a different patch size when constructing
        # the UNet externally
        if target_emb:
            self.target_embedding = TargetEncoding(patch_size)

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        self.num_layers = num_layers

        if target_emb:
            self.conv_in = nn.Sequential(ConvBlock(in_size + 1, features_start), ConvBlock(features_start, features_start))
        else:
            self.conv_in = nn.Sequential(ConvBlock(in_size, features_start), ConvBlock(features_start, features_start))

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
        # print(x.shape, target.shape, t.shape)
        t_emb = self.t_embedding(t.flatten()) # shape is (b, 512)
        # print("t_emb shape: ", t_emb.shape)
        target_emb = self.target_embedding(target)
        x = torch.concat((x, target_emb), dim=1)

    # print("DEBUGGING")
    # print("x shape before conv_in: ", x.shape)

        x = self.conv_in(x)

    # print("x shape after conv_in: ", x.shape)

        # Store hidden states for U-net skip connections.
        x_i = [x]

        # Encoder stage.
        for layer in self.layers[: self.num_layers - 1]:
            # print("x_i[-1] shape before encoder layer: ", x_i[-1].shape)
            # print("t_emb shape before encoder layer: ", t_emb.shape)
            x_i.append(layer(x=x_i[-1], t_emb=t_emb))

        # Decoder stage.
        for i, layer in enumerate(self.layers[self.num_layers - 1 :]):
            # print("x_i[-1] shape before decoder layer: ", x_i[-1].shape)
            # print("x_i[-2 - i] shape before decoder layer: ", x_i[-2 - i].shape)
            # print("t_emb shape before decoder layer: ", t_emb.shape)
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
        self.model = UNet(in_size=self.in_size, out_size=self.out_size, patch_size=self.patch_size, device=self.device).to(device)

        self.prediction_model_name = prediction_model_name
        if self.prediction_model_name == 'yolov5':
            from yolo_bounding import YOLOBox
            self.prediction_model = YOLOBox()  
        
        elif self.prediction_model_name == 'frontnet':
            self.prediction_model = load_model(f"{project_root}/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
            self.prediction_model.eval()

        self.train_set = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = 32, shuffle = True, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)
        self.test_set = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = 32, shuffle = True, drop_last = True, num_workers = 1, train=False, train_set_size=0.9, IMRC=True)

    def denoised_prediction(self, x, conditioning, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, conditioning, c_noise)
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def train(self, data_loader: torch.utils.data.DataLoader, device: torch.device, nepochs: int = 10):
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

        # batch_size = 64
        n_samples = 1

        for epoch in trange(nepochs):

            for imgs, _ in tqdm(self.train_set):
            # for [patches, conditioning] in data_loader:
            #     patches = (2 * patches.to(device)) - 1
            #     conditioning = conditioning.to(device)
                optimizer.zero_grad()

                sf = np.random.uniform(0.5, 1.1, 1,)
                if self.prediction_model_name == 'frontnet':
                    tx = np.random.uniform(-20., 140., 1,)
                    ty = np.random.uniform(-10., 80., 1,)
                elif self.prediction_model_name == 'yolov5':
                    tx = np.random.uniform(-20., 630., 1,)
                    ty = np.random.uniform(-10., 310., 1,)
                x = np.random.uniform(0., 1.5, 1,)
                y = np.random.uniform(-1, 1, 1,)
                z = np.random.uniform(-0.5, 0.5, 1,)
                yaw = np.random.uniform(-0.3, 0.3, 1,)

                # print("conditioning random shape: ", (sf.shape, tx.shape, ty.shape, x.shape, y.shape, z.shape, yaw.shape))

                conditioning_random = torch.tensor(np.stack((sf, tx, ty, x, y, z, yaw)).T, dtype=torch.float32).to(device)
                sampled_patches = self.sample(n_samples=1, targets=conditioning_random, device=device, n_steps=25).to(device)
                # print("sampled_patches shape: ", sampled_patches.shape, " min: ", torch.min(sampled_patches), " max: ", torch.max(sampled_patches))

                # print("DEBUGGING")
                # print("positions shape: ", positions.shape)
                # print("example position: ", positions[0])
                T_matrix = construct_T_matrix(conditioning_random[0, 0], conditioning_random[0, 1], conditioning_random[0, 2]).unsqueeze(0).to(device)
                    # print("T_matrices shape: ", T_matrices.shape)
                # T_matrices = torch.stack([construct_T_matrix(s, tx, ty) for s, tx, ty in zip(conditioning_random[:, 0], conditioning_random[:, 1], conditioning_random[:, 2])]).to(device)



                conditioning_random = conditioning_random.repeat_interleave(self.train_set.batch_size, dim=0)
                sampled_patches = sampled_patches.repeat_interleave(self.train_set.batch_size, dim=0)
                T_matrices = T_matrix.repeat_interleave(self.train_set.batch_size, dim=0)


                # print("conditioning_random shape after repeat: ", conditioning_random.shape)
                # print("sampled_patches shape after repeat: ", sampled_patches.shape)
                # print("T_matrices shape after repeat: ", T_matrices.shape)

                target_yaw_v = conditioning_random[:, -1] # of shape [n_samples,]
                # target_yaw_v = target_yaw_v.repeat_interleave(self.train_set.batch_size, dim=0)
                #target_yaw_v = target_yaw_noisy
                # print("target_yaw_v shape: ", target_yaw_v.shape)
                # print("example target_yaw_v: ", target_yaw_v[0])

                # conditioning_random = conditioning_random.repeat_interleave(self.train_set.batch_size, dim=0)


                imgs = imgs / 255.

                if self.prediction_model_name == 'yolov5':
                    imgs = torch.nn.functional.interpolate(imgs, size=(320, 640), mode='bilinear', align_corners=False)
                # imgs = next(iter(self.train_set))[0].to(device) / 255.0  # normalize to [0, 1]
                # print("imgs shape: ", imgs.shape, " min: ", torch.min(imgs), " max: ", torch.max(imgs))

                # appy all sampled patches to all images in the batch
                # manipulated_images_all = []
                # for i in range(imgs.shape[0]):
                manipulated_images = project_patch(
                    patches=sampled_patches,
                    T_matrices=T_matrices,
                    images=imgs
                )
                # manipulated_images_all.append(manipulated_images)

                # manipulated_images = torch.stack(manipulated_images_all).permute(1, 0, 2, 3, 4)

                # add random noise to manipulated images for robustness
                manipulated_images = manipulated_images + torch.randn_like(manipulated_images) * 0.01

                manipulated_images.clamp_(0., 1.)

                # print("manipulated_images shape: ", manipulated_images.shape, " min: ", torch.min(manipulated_images), " max: ", torch.max(manipulated_images))

                # loss_per_sample = []
                # for image_batch in manipulated_images:
                if self.prediction_model_name == 'frontnet':
                    x, y, z, yaw = self.prediction_model(manipulated_images*255.)
                    # print("x, y, z, yaw:", x, y, z, yaw)
                    prediction = torch.stack([x, y, z, yaw])
                    prediction = prediction.squeeze(2).mT
                elif self.prediction_model_name == 'yolov5':
                   
                    manipulated_images = manipulated_images.repeat_interleave(3, dim=1)

                    prediction = self.prediction_model(manipulated_images)  # yolo expects images in range [0, 1]
                    # print("prediction from yolov5: ", prediction[0])

                    #scale back to 160x96
                    prediction[:, [0, 2]] *= (160.0 / 640.0)  # x coords
                    prediction[:, [1, 3]] *= (96.0 / 320.0)   # y coords

                    # print("YOLOv5 prediction before scaling to 160x96:", prediction)
                    # print("Scaled bounding box for 160x96 image: ", prediction)

                    prediction = self.prediction_model.cam.batch_xyz_from_boxes(prediction) #  xyzyaw from bounding box


                yaw_v = prediction[:, 3]


                mse_losses = torch.stack([F.mse_loss(tar, pre) for tar, pre in zip(conditioning_random[:, 3:6], prediction[:, :3])]) # calc mse for each of the predictions of each patch
                angular_losses = 1 - torch.cos(normalize_yaw_t(yaw_v) - normalize_yaw_t(target_yaw_v))  # angular loss for yaw
                # print(angular_losses)
                #prediction_loss = torch.mean(mse_losses + angular_losses)
                prediction_loss = mse_losses + angular_losses
                # loss_per_sample.append(prediction_loss.mean())
                # sorted_prediction_losses = torch.sort(prediction_loss, descending=True)[0]
                # top5_prediction_losses = sorted_prediction_losses[:5]
                prediction_losses.append(torch.mean(prediction_loss).detach().cpu().numpy())
                all_prediction_losses.append(torch.mean(prediction_loss).detach().cpu().numpy())


                loss = prediction_loss.mean()
                # loss = combined_loss
                losses.append(loss.detach().cpu().numpy())
                all_losses.append(loss.detach().cpu().numpy())
                

                # Bwd pass
                loss.backward()
                optimizer.step()
                scheduler.step()

            if (epoch+1) % 1 == 0:
                mean_loss = np.mean(np.array(losses))
                # mean_reco_loss = np.mean(np.array(reco_losses))
                mean_prediction_loss = np.mean(np.array(prediction_losses))
                losses = []
                reco_losses = []
                prediction_losses = []
                print("Epoch %d,\t Loss %f \t Prediction Loss %f" % (epoch+1, mean_loss, mean_prediction_loss))
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
                os.makedirs(f'diffusion/results/diffusion_training/{self.prediction_model_name}/{100}/checkpoints/', exist_ok=True)
                model.save(f'diffusion/results/diffusion_training/{self.prediction_model_name}/{100}/checkpoints/checkpoint_epoch_{epoch+1}.pth')

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
            T_matrix = construct_T_matrix(conditioning[0], conditioning[1], conditioning[2]).unsqueeze(0).to(device)
            batch_patch_losses = []
            for imgs in data_loader:
                imgs = imgs[0].to(device) / 255.0  # normalize to [0, 1]

                if self.prediction_model_name == 'yolov5':

                    imgs = torch.nn.functional.interpolate(imgs, size=(320, 640), mode='bilinear', align_corners=False)
                              
                manipulated_image = project_patch(
                    patches=test_patch,
                    T_matrices=T_matrix,
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

        x_t = torch.stack([(x + 1) / 2 for x in x_t]).clamp(min=0,max=1) # normalize
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
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Paths to the dataset pickle files.')
    parser.add_argument('--epochs', type=int, default=1_000, help='Number of epochs to train.')
    parser.add_argument('--output', type=str, default='trained_model.pth', help='Path to save the model.')
    parser.add_argument('--corpus_size', type=int, default=1000, help='Size of the training corpus.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.autograd.set_detect_anomaly(True)

    data = []
    for dataset_path in args.datasets:
        with open(dataset_path, 'rb') as f:
            data += pickle.load(f)


    patches = []
    targets = []
    positions = []

    for i in range(len(data)):
        patches.append(data[i][0])
        targets.append(data[i][1])
        positions.append(data[i][2])

    
    patches = np.array(patches) # shape (N, 45, 80)
    targets = np.array(targets) # shape (N, 1, 4) -> x, y, z, yaw, x in range [0, 1.5], [-1, 1], [-0.5, 0.5], [-0.3, 0.3]
    positions = np.array(positions) # shape (N, 1, 3), sf in range [0.31, 0.99], tx, ty in range [0, 85]



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

    patches = np.array([(patch - np.min(patch)) / (np.max(patch) - np.min(patch)) for patch in patches]) # normalize
    patches = torch.tensor(patches).unsqueeze(1)
    targets = torch.tensor(targets, dtype=torch.float).squeeze(1)
    positions = torch.tensor(positions, dtype=torch.float).squeeze(1)

    conditioning = torch.cat((positions, targets), dim=1)

    # print(patches.shape)

    # Define dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.utils.data.TensorDataset(patches, conditioning)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # model = UNet(in_size=1, out_size=1, device=device)
    # model.to(device)

    model = DiffusionModel(device, lr=1e-4, patch_size=patch_size, prediction_model_name=args.model)

    # training
    # print("Start training..")
    # model.load(f'results/diffusion_training/{args.output}')
    all_losses = model.train(loader, device, nepochs=args.epochs)

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