import torch
import torch.nn as nn
import numpy as np

from diffusion.example_unet import UNet
from simulators.cf_frontnet_pt import CFSim

from eval_gt_patches import calc_loss, get_targets, gen_T

def get_alpha_betas(N: int):
  """Schedule from the original paper. Commented out is sigmoid schedule from:

  'Score-Based Generative Modeling through Stochastic Differential Equations.'
   Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
   Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
  """
  beta_min = 0.1
  beta_max = 20.
  #betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
  betas = np.random.uniform(10e-4, .02, N)  # schedule from the 2020 paper
  alpha_bars = np.cumprod(1 - betas)
  return alpha_bars, betas


def sample(model: nn.Module, targets: torch.tensor, device: torch.device, patch_size: (int, int), n_samples: int = 50, n_steps: int=100):
    """Alg 2 from the DDPM paper."""
    x_t = torch.randn((n_samples, 1, *patch_size)).to(device)
    targets = targets.to(device)
    alpha_bars, betas = get_alpha_betas(n_steps)
    alphas = 1 - betas
    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1), dtype=torch.int32).to(device)
        ab_t = alpha_bars[t] * torch.ones((n_samples, 1), dtype=torch.int32).to(device)  # Tile the alpha to the number of samples
        z = (torch.randn((n_samples, 1, *patch_size)) if t > 1 else torch.zeros((n_samples, 1, *patch_size))).to(device)
        model_prediction = model(x_t, targets, ts.squeeze(1))
        x_t = 1 / alphas[t]**.5 * (x_t - (betas[t]/(1-ab_t)**.5).unsqueeze(2).unsqueeze(2) * model_prediction)
        x_t += betas[t]**0.5 * z

    return x_t


def loss_dataset(gt_patch, diffusion_patch, random_patch, targets, position, dataset, sim):
    T = gen_T(position)

    gt_losses = []
    diffusion_losses = []
    random_losses = []

    for data in dataset:
        img, _ = data

        gt_losses.append(calc_loss([targets], gt_patch, [T], img, sim))
        diffusion_losses.append(calc_loss([targets], diffusion_patch, [T], img, sim))
        random_losses.append(calc_loss([targets], random_patch, [T], img, sim))

    return np.array(gt_losses), np.array(diffusion_losses), np.array(random_losses)

if __name__ == '__main__':
    from pathlib import Path
    import pickle

    np.random.seed(2562)
    torch.manual_seed(2562)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = 100

    model = UNet(in_size=1, out_size=1, device=device)
    model.to(device)

    model.load_state_dict(torch.load('conditioned_unet_80x80_1000_3256i_255.pth', map_location=device))
    model.eval()

    with open('data/FAP_combined.pickle', 'rb') as f:
        data = pickle.load(f)
    gt_patches = []
    gt_targets = []
    gt_positions = []
    for i in range(len(data)):
        gt_patches.append(data[i][0])
        gt_targets.append(data[i][1])
        gt_positions.append(data[i][2])
    
    random_idx = np.random.choice(len(gt_patches), size=50, replace=False)
    gt_patches = np.array(gt_patches)[random_idx]
    gt_targets = np.array(gt_targets)[random_idx]
    gt_positions = np.array(gt_positions)[random_idx]


    # print(random_idx[:10])
    # print(gt_patches.shape)

    with torch.no_grad():
        diffusion_patches = sample(model, torch.tensor(gt_targets), device, [80, 80], n_samples=n_samples).detach().cpu() * 255.

    print(diffusion_patches.shape)
    # print(torch.min(diffusion_patches), torch.max(diffusion_patches))
    diffusion_patches = (diffusion_patches - torch.min(diffusion_patches)) / (torch.max(diffusion_patches) - torch.min(diffusion_patches)) # normalize
    diffusion_patches *= 255.
    print(torch.min(diffusion_patches), torch.max(diffusion_patches))
    print(diffusion_patches.shape)

    np.save(f'eval/diffusion_patches__{n_samples}.npy', diffusion_patches.numpy())

    import matplotlib.pyplot as plt
    fig = plt.figure(constrained_layout=True)
    n_cols = min(n_samples, 5)
    n_rows = int(np.ceil(n_samples / 5))
    axs_samples = fig.subplots(n_rows, n_cols).flatten()
    for i, sample in enumerate(diffusion_patches):
        axs_samples[i].imshow(sample[0], cmap='gray')
        axs_samples[i].axis('off')
        # axs_samples[i].set_title(f'sample {i}')
    fig.savefig(f'eval/diffusion_patches_{n_samples}.png', dpi=200)
    plt.show()

    sim = CFSim()

    # # generate random patches
    random_patches = torch.rand_like(diffusion_patches) * 255.

    dataset_path = "simulators/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    _, dataset = sim.load_dataset(dataset_path, batch_size=1, train_set_size=0.9)

    all_gt = []
    all_diffusion = []
    all_random = []
    for idx, (gt_patch, diffusion_patch, random_patch) in enumerate(zip(torch.tensor(gt_patches), diffusion_patches, random_patches)):
        gt_loss, diffusion_loss, random_loss = loss_dataset(gt_patch, diffusion_patch[0], random_patch[0], torch.tensor(gt_targets[idx]), gt_positions[idx], dataset, sim)
        all_gt.append(gt_loss)
        all_diffusion.append(diffusion_loss)
        all_random.append(random_loss)

    all_gt = np.array(all_gt)
    all_diffusion = np.array(all_diffusion)
    all_random = np.array(all_random)

    print(all_gt.shape)
    print(all_diffusion.shape)
    print(all_random.shape)

    np.save(f'eval/comparison_gt_{n_samples}.npy', all_gt)
    np.save(f'eval/comparison_diffusion_{n_samples}.npy', all_diffusion)
    np.save(f'eval/comparison_random_{n_samples}.npy', all_random)

    # all_gt = np.load('eval/comparison_gt.npy')
    # all_diffusion = np.load('eval/comparison_diffusion.npy')
    # all_random = np.load('eval/comparison_random.npy')


    print(f"Ground truth patches mean loss: {np.mean(all_gt)}, std: {np.std(all_gt)}")
    # print(f"Per patch, mean: {np.mean(all_gt, axis=1)}, std: {np.std(all_gt, axis=1)}")
    print()
    print(f"Random patches mean loss: {np.mean(all_random)}, std: {np.std(all_random)}")
    # print(f"Per patch, mean: {np.mean(all_random, axis=1)}, std: {np.std(all_random, axis=1)}")
    print()
    print(f"Diffusion patches mean loss: {np.mean(all_diffusion)}, std: {np.std(all_diffusion)}")
    # print(f"Per patch, mean: {np.mean(all_diffusion, axis=1)}, std: {np.std(all_diffusion, axis=1)}")
    print()

    data = np.array([np.mean(all_gt.T, axis=1), np.mean(all_random.T, axis=1), np.mean(all_diffusion.T, axis=1)])
    print(data.shape)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1, layout='constrained')
    axs.boxplot(data.T, tick_labels=['ground truth', 'random', 'diffusion']) 
    axs.set_ylabel('MSE(target, prediction)')

    fig.savefig(f'eval/eval_boxplot__{n_samples}.png', dpi=200)
    
    