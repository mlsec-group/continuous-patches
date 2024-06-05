# Base code by Cornelius Braun 

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange


# Define net
class Net(nn.Module):
  def __init__(self, nhidden: int = 256):
    super().__init__()
    layers = [nn.Linear(3*3+1, nhidden)] # Change this to 6 if you want to use the fourier embeddings of t
    for _ in range(5):
      layers.append(nn.Linear(nhidden, nhidden))
    layers.append(nn.Linear(nhidden, 3*3))
    self.linears = nn.ModuleList(layers)

    #Iinit using kaiming
    for layer in self.linears:
      nn.init.kaiming_uniform_(layer.weight)

  def forward(self, x, t):
    # Optional: Use Fourier feature embeddings for t, cf. transformers
    #t = torch.concat([t - 0.5, torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t), -torch.cos(4*torch.pi*t)], axis=1)
    x = torch.concat([x, t], axis=-1)
    for l in self.linears[:-1]:
      x = nn.ReLU()(l(x))
    return self.linears[-1](x)
  
def get_alpha_betas(N: int):
  """Schedule from the original paper. Commented out is sigmoid schedule from:

  'Score-Based Generative Modeling through Stochastic Differential Equations.'
   Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
   Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
  """
  beta_min = 0.1
  beta_max = 20.
  betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
  #betas = np.random.uniform(10e-4, .02, N)  # schedule from the 2020 paper
  alpha_bars = np.cumprod(1 - betas)
  return alpha_bars, betas

def train(nepochs: int = 10, denoising_steps: int = 100):
  """Alg 1 from the DDPM paper"""
  model = Net()
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  alpha_bars, _ = get_alpha_betas(denoising_steps)      # Precompute alphas

  all_losses = []

  losses = []
  print("Start training...")
  for epoch in trange(nepochs):
    for [data] in loader:
      data = data.to(device)
      optimizer.zero_grad()

      # Fwd pass
      t = torch.randint(denoising_steps, size=(data.shape[0],))  # sample timesteps - 1 per datapoint
      alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)    # Get the alphas for each timestep
      noise = torch.randn(*data.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
      model_in = alpha_t**.05 * data + noise*(1-alpha_t)**.05   # Noise corrupt the data (eq14)
      out = model(model_in, t.unsqueeze(1).to(device))
      loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
      losses.append(loss.detach().cpu().numpy())
      all_losses.append(loss.detach().cpu().numpy())

      # Bwd pass
      loss.backward()
      optimizer.step()

    if (epoch+1) % 5_000 == 0:
        mean_loss = np.mean(np.array(losses))
        losses = []
        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

  return model, all_losses


def sample(model: nn.Module, n_samples: int = 50, n_steps: int=100):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, 3*3)).to(device)
    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1)).to(device)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(device)  # Tile the alpha to the number of samples
        z = (torch.randn((n_samples, 3*3)) if t > 1 else torch.zeros((n_samples, 3*3))).to(device)
        model_prediction = model(x_t, ts)
        x_t = 1 / alphas[t]**.5 * (x_t - betas[t]/(1-ab_t)**.5 * model_prediction)
        x_t += betas[t]**0.5 * z

    return x_t

# # Get data from a circle
# thetas = np.random.uniform(0, 2*np.pi, 50)
# x = np.cos(thetas) + np.random.normal(0, 3e-2, 50)
# y = np.sin(thetas) + np.random.normal(0, 3e-2, 50)
# data = np.vstack((x, y)).T
# print(data.shape)

# plt.figure(figsize=(5, 5))
# plt.scatter(x, y)
# plt.show()

data = np.random.rand(3,3)
plt.figure(figsize=(5, 5))
plt.imshow(data, cmap='gray')
plt.show()

# Define dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torch.utils.data.TensorDataset(torch.Tensor(data.flatten()).unsqueeze(0))
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# training
trained_model, all_losses = train(40_000)
trained_model = trained_model.eval()

fig, ax = plt.subplots(1,1)
ax.plot(all_losses)
ax.set_title('Train loss')
ax.set_x('training steps')
ax.set_y('MSE')
fig.savefig('train_loss.png', dpi=200)

# inference
samples = sample(trained_model, n_samples=3).detach().cpu().numpy()
print(samples.shape)
fig, axs = plt.subplots(1, 4)
axs[0].set_title('ground truth')
axs[0].imshow(data, cmap='gray')
for i, sample in enumerate(samples):
  axs[i+1].set_title(f'diffusion sample {i}')
  axs[i+1].imshow(sample.reshape(3,3), cmap='gray')
# plt.scatter(x, y)
# plt.scatter(*(samples.T))
fig.savefig('samples.png', dpi=200)
plt.show()