import torch
import torch.nn as nn
import numpy as np

from diffusion.example_unet import UNet
from eval_diffusion import get_alpha_betas, sample

from torchsummary import summary

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
    # model.eval()

    summary(model, [(1, 80, 80), (3,), (1,)])

    # with open('data/FAP_combined.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # gt_patches = []
    # gt_targets = []
    # #gt_positions = []
    # for i in range(len(data)):
    #     gt_patches.append(data[i][0])
    #     gt_targets.append(data[i][1])
    #     #gt_positions.append(data[i][2])

    # gt_patches = np.array(gt_patches)[:n_samples]
    # gt_targets = np.array(gt_targets)[:n_samples]


    # # running into memory issues with this sample function! fix: don't compute gradients
    # with torch.no_grad():
    #     samples = sample(model, torch.tensor(gt_targets), device, n_samples=n_samples, patch_size=[80, 80], n_steps=1_000).detach().cpu().numpy()
    # print(samples.shape)

    # samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples)) # normalize

    # print(gt_patches.max(), samples.max())

    # from skimage.metrics import structural_similarity as ssim

    # all_ssim = []
    # for gt, sample in zip(gt_patches, samples):
    #     all_ssim.append(ssim(gt, sample[0], data_range=sample.max() - sample.min()))

    # all_ssim = np.array(all_ssim)
    # np.save(f'eval/ssim_{n_samples}.npy', all_ssim)

    # print(all_ssim.shape)

    # print("SSIM mean: ", np.mean(all_ssim), ", std: ", np.std(all_ssim))
    # print("SSIM first 4: ", all_ssim[:4])

    
    # # print(np.min(samples), np.max(samples))

    # import matplotlib.pyplot as plt
    # plt.rcParams.update({
    #             "text.usetex": True,
    #             "font.family": "sans-serif",
    #             "font.sans-serif": "Helvetica",
    #             "font.size": 12,
    #             "figure.figsize": (5, 3),
    #             "mathtext.fontset": 'stix'
    # })

    # # random_idx = np.random.choice(len(gt_patches), size=2, replace=False)

    # # print(gt_targets[random_idx])
    # fig, axs = plt.subplots(2, 4, layout='constrained')
    # for i in range(2):
    #     for j in range(2):
    #         idx = i * 2 + j

    #         axs[i, j * 2].imshow(gt_patches[idx], cmap='gray')
    #         axs[i, j * 2].axis('off')
    #         axs[i, j * 2].set_title(f'FAP {idx + 1}')

    #         axs[i, j * 2 +1].imshow(samples[idx][0], cmap='gray')
    #         axs[i, j * 2 +1].axis('off')
    #         axs[i, j * 2 +1].set_title(f'diffusion {idx + 1}')

    # # fig, axs = plt.subplots(2, 4, layout='constrained')
    # # for i in range(4):
    # #     axs[i][0]
    # #     axs[i][1].imshow(samples[i], cmap='gray')
    # # axs[0][0].imshow(gt_patches[random_idx[0]], cmap='gray')
    # # axs[1][0].imshow(gt_patches[random_idx[1]], cmap='gray')
    # # axs[0][1].imshow(diffusion_patches.detach().cpu().numpy()[random_idx[0]][0], cmap='gray')
    # # axs[1][1].imshow(diffusion_patches.detach().cpu().numpy()[random_idx[1]][0], cmap='gray')

    # # plt.tight_layout()
    # fig.savefig(f'eval/eval_visual.pdf', dpi=200)