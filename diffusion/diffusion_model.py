import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
import numpy as np

from tqdm import trange

# source for UNet: https://github.com/jbergq/simple-diffusion-model/

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
    def __init__(self, patch_size: tuple[int, int], embed_channels: int = 1):
        super().__init__()

        # the whole purpose of this is to learn to encode the target 
        # and bring it into a shape that is easy to concatenate with the
        # image before processing it

        self.embed_channels = embed_channels
        self.h, self.w = patch_size
        self.linear = nn.Linear(6, 512)   # 3 for target only, 6 for target+position
        self.conv = nn.Conv2d(512, 64, kernel_size=2, stride=1, padding=5)

    def forward(self, target: Tensor) -> Tensor:
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
        x = self.up(x)

        # Concatenate with encoder skip connection.
        if x_skip is not None:
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
        target_emb : bool = True
    ) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
            PositionalEncoding(max_time_steps, t_emb_size, device), nn.Linear(t_emb_size, t_emb_size)
        )

        if target_emb:
            self.target_embedding = TargetEncoding([80, 80])

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

        x = self.conv_in(x)

        # Store hidden states for U-net skip connections.
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
    def __init__(self, device, in_size=1, out_size=1, lr=1e-4):
        self.in_size = in_size    # number of channels (1 -> grayscale)
        self.out_size = out_size  # number of channels
        self.lr = lr
        self.sigma_data = 0.5

        self.device = device

        self.model = UNet(in_size=self.in_size, out_size=self.out_size, device=self.device).to(device)

    def denoised_prediction(self, x, conditioning, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, conditioning, c_noise)
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def train(self, data_loader: torch.utils.data.DataLoader, device: torch.device, nepochs: int = 10):
        P_mean = 1.2
        P_std = 1.2
        p_unconditioned = 0.1

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-4)
        all_losses = []

        losses = []
        for epoch in trange(nepochs):
            for [patches, conditioning] in data_loader:
                patches = (2 * patches.to(device)) - 1
                conditioning = conditioning.to(device)
                optimizer.zero_grad()
                sigmas = P_std * torch.randn(patches.shape[0], 1, 1, 1, device=patches.device) + P_mean
                sigmas = sigmas.exp()

                noise = torch.randn_like(patches) * sigmas # Sample DIFFERENT random noise for each datapoint
                
                model_in = patches + noise # Noise corrupt the data 
                out = self.denoised_prediction(model_in, conditioning, sigmas)
                weight = (sigmas ** 2 + self.sigma_data ** 2) / (sigmas * self.sigma_data) ** 2
                loss = torch.mean(weight * (patches - out)**2) # Compute loss on predictio
                losses.append(loss.detach().cpu().numpy())
                all_losses.append(loss.detach().cpu().numpy())

                # Bwd pass
                loss.backward()
                optimizer.step()

            if (epoch+1) % 10 == 0:
                mean_loss = np.mean(np.array(losses))
                losses = []
                print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

        return all_losses

    @torch.no_grad()
    def sample(self, n_samples: int, targets: torch.tensor, device: torch.device, patch_size: tuple[int, int], n_steps: int=1_000):
        """Alg 2 from the DDPM paper."""
        self.model.eval()
        # make sure that targets is a tensor
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        # and of shape (n_samples, 3)
        if len(targets.shape) == 1 or targets.shape[0] == 1:
            targets = targets.repeat(n_samples, 1)

        sigma_max = 80
        sigma_min = 0.002
        rho = 7

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
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Paths to the dataset pickle files.')
    parser.add_argument('--epochs', type=int, default=1_000, help='Number of epochs to train.')
    parser.add_argument('--output', type=str, default='trained_model.pth', help='Path to save the model.')
    args = parser.parse_args()

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

    
    patches = np.array(patches) # shape (N, 80, 80)
    targets = np.array(targets) # shape (N, 1, 3)
    positions = np.array(positions) # shape (N, 1, 3), sf in range [0.4, 0.8], tx, ty in range [0, 1]

    patch_size = patches.shape[-2:]

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

    model = DiffusionModel(device)

    # training
    # print("Start training..")
    # model.load(f'results/diffusion_training/{args.output}')
    all_losses = model.train(loader, device, nepochs=args.epochs)
    
    os.makedirs('results/diffusion_training', exist_ok=True)
    model.save(f'results/diffusion_training/{args.output}')
    
    n_samples = 5
    sf = np.random.uniform(0.4, 0.8, n_samples)
    tx = np.random.uniform(0., 1., n_samples)
    ty = np.random.uniform(0., 1., n_samples)
    x = np.random.uniform(0,2,n_samples)
    y = np.random.uniform(-1,1,n_samples,)
    z = np.random.uniform(-0.5,0.5,n_samples,)

    r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z)).T, dtype=torch.float32)

    samples = model.sample(n_samples, r_targets, device, patch_size=patch_size, n_steps=25).detach().to('cpu').numpy()
    

    fig = plt.figure(constrained_layout=True, figsize=(n_samples*5, 5))
    axs_samples = fig.subplots(1, n_samples)
    for i, sample in enumerate(samples):
        ax = axs_samples[i] if n_samples > 1 else axs_samples
        ax.imshow(sample[0], cmap='gray')
        ax.set_title(f'sample {i}')
        ax.set_axis_off()
    fig.savefig(f'results/diffusion_training/samples1.png', dpi=200)
    plt.show()