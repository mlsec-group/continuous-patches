import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional

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
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class PositionalEncoding(nn.Module):
    """Transformer sinusoidal positional encoding."""

    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        """Constructs the PositionalEncoding.

        Args:
            max_time_steps (int): Number of timesteps that can be uniquely represented by encoding.
            embedding_size (int): Size of returned time embedding.
            n (int, optional): User-defined scalar. Defaults to 10000.
        """
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        # Pre-compute the embedding vector for each possible time step.
        # Store in 2D tensor indexed by time step `t` along 0th axis, with embedding vectors along 1st axis.
        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False).to(device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t: Tensor) -> Tensor:
        """Returns embedding encoding time step `t`.

        Args:
            t (Tensor): Time step.

        Returns:
            Tensor: Returned position embedding.
        """
        return self.pos_embeddings[t, :]


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
            t_emb = self.t_proj(t_emb)
            x = t_emb.unsqueeze(2).unsqueeze(2) + x #rearrange(t_emb, "b c -> b c 1 1") + x

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
        num_layers: int = 5,
        features_start: int = 64,
        t_emb_size: int = 512,
        max_time_steps: int = 1000,
    ) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
            PositionalEncoding(max_time_steps, t_emb_size), nn.Linear(t_emb_size, t_emb_size)
        )

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        self.num_layers = num_layers

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

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        if t is not None:
            # Create time embedding using positional encoding.
            t_emb = self.t_embedding(t)

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


def train(model: nn.Module, data_loader: torch.utils.data.DataLoader, nepochs: int = 10, denoising_steps: int = 100):
  """Alg 1 from the DDPM paper"""
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  alpha_bars, _ = get_alpha_betas(denoising_steps)      # Precompute alphas

  all_losses = []

  losses = []
  print("Start training...")
  for epoch in trange(nepochs):
    for [data] in data_loader:
    #   data = data.to(device)
      optimizer.zero_grad()
      # print("batch", data.shape)
      # Fwd pass
      t = torch.randint(denoising_steps, size=(data.shape[0],))  # sample timesteps - 1 per datapoint
      # print("t", t.shape)
      alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)    # Get the alphas for each timestep
      # print("alpha_t", alpha_t**0.5, (alpha_t**.5).shape)

      noise = torch.randn(*data.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
      model_in = alpha_t**.5 * data + noise*(1-alpha_t)**.5   # Noise corrupt the data (eq14)
      out = model(model_in, t.unsqueeze(1).to(device))
      loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
      losses.append(loss.detach().cpu().numpy())
      all_losses.append(loss.detach().cpu().numpy())

      # Bwd pass
      loss.backward()
      optimizer.step()

    if (epoch+1) % 500 == 0:
        mean_loss = np.mean(np.array(losses))
        losses = []
        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

  return model, all_losses


def sample(model: nn.Module, patch_size: int=3*3, n_samples: int = 50, n_steps: int=100):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, patch_size)).to(device)
    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1)).to(device)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(device)  # Tile the alpha to the number of samples
        z = (torch.randn((n_samples, patch_size)) if t > 1 else torch.zeros((n_samples, patch_size))).to(device)
        model_prediction = model(x_t, ts)
        x_t = 1 / alphas[t]**.5 * (x_t - betas[t]/(1-ab_t)**.5 * model_prediction)
        x_t += betas[t]**0.5 * z

    return x_t

if __name__ == '__main__':
    import pickle
    import numpy as np
    import torch

    with open('/home/hanfeld/flying_adversarial_patch/80x80patches.pickle', 'rb') as f:
        data = pickle.load(f)

    print(data.shape)
    patch_size = data.shape[1:]
    # print(patch_size)
    # # data = data.reshape(data.shape[0], -1)
    # # print(data.shape)
    data = np.array(data)
    data_t = torch.Tensor(data).unsqueeze(1)
    print(data_t.shape)

    # Define dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.utils.data.TensorDataset(data_t.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


    model = UNet(in_size=1, out_size=1)
    model.to(device)
    
    [batch] = next(iter(loader))
    print(batch.shape)

    t = torch.randint(100, size=(batch.shape[0],))

    out = model(batch, t)
    print(out.shape)