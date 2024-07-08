import torch
import numpy as np

from diffusion.example_unet import UNet, sample
from simulators.cf_frontnet_pt import CFSim

from eval_gt_patches import calc_loss, get_targets, get_Ts


def loss_dataset(gt_patch, diffusion_patch, random_patch, patch_path, dataset, sim):
    targets = get_targets(patch_path)
    Ts = get_Ts(patch_path)

    gt_losses = []
    diffusion_losses = []
    random_losses = []

    for data in dataset:
        img, _ = data
        # print(img.shape)
        gt_losses.append(calc_loss(targets, gt_patch, Ts, img, sim))
        diffusion_losses.append(calc_loss(targets, diffusion_patch, Ts, img, sim))
        random_losses.append(calc_loss(targets, random_patch, Ts, img, sim))

    return np.array(gt_losses), np.array(diffusion_losses), np.array(random_losses)

if __name__ == '__main__':
    from pathlib import Path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = 3

    model = UNet(in_size=1, out_size=1)
    model.to(device)

    model.load_state_dict(torch.load('unet_80x80_1000_v2.pth', map_location=device))

    samples = sample(model, device, [80, 80], n_samples=n_samples).detach().cpu() * 255.
    del model  # running into memory issues, hence have to delete the diffusion model

    print(samples.shape)

    sim = CFSim()

    # generate random patches
    random_patches = torch.rand_like(samples) * 255.

    # load 3 patches
    path = Path(f'/home/hanfeld/flying_adversarial_patch/results/dataset80x80/')
    patches_paths = list(path.glob('[0-9]*/patches.npy'))
    patches_paths.sort(key=lambda path: int(path.parent.name))
    patches = np.array([np.load(file_path)[-1][0][0]*255. for file_path in patches_paths[:n_samples]])



    dataset_path = "/home/hanfeld/flying_adversarial_patch/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    _, dataset = sim.load_dataset(dataset_path, batch_size=1, train_set_size=0.9)

    all_gt = []
    all_diffusion = []
    all_random = []
    for idx, (gt_patch, diffusion_patch, random_patch) in enumerate(zip(patches, samples, random_patches)):
        gt_loss, diffusion_loss, random_loss = loss_dataset(gt_patch, diffusion_patch[0], random_patch[0], patches_paths[idx], dataset, sim)
        all_gt.append(gt_loss)
        all_diffusion.append(diffusion_loss)
        all_random.append(random_loss)

    all_gt = np.array(all_gt)
    all_diffusion = np.array(all_diffusion)
    all_random = np.array(all_random)

    print(all_gt.shape)
    print(all_diffusion.shape)
    print(all_random.shape)

    np.save('eval/comparison_gt.npy', all_gt)
    np.save('eval/comparison_diffusion.npy', all_diffusion)
    np.save('eval/comparison_random.npy', all_random)

    print(f"Ground truth patches mean loss: {np.mean(all_gt)}, std: {np.std(all_gt)}")
    print(f"Per patch, mean: {np.mean(all_gt, axis=1)}, std: {np.std(all_gt, axis=1)}")
    print()
    print(f"Random patches mean loss: {np.mean(all_random)}, std: {np.std(all_random)}")
    print(f"Per patch, mean: {np.mean(all_random, axis=1)}, std: {np.std(all_random, axis=1)}")
    print()
    print(f"Diffusion patches mean loss: {np.mean(all_diffusion)}, std: {np.std(all_diffusion)}")
    print(f"Per patch, mean: {np.mean(all_diffusion, axis=1)}, std: {np.std(all_diffusion, axis=1)}")
    print()

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, layout='constrained')
    axs[0].boxplot(all_gt.T, tick_labels=[f'gt {i}' for i in range(len(all_gt))]) 
    axs[0].set_ylabel('MSE(target, prediction)')

    axs[1].boxplot(all_random.T, tick_labels=[f'rnd {i}' for i in range(len(all_random))]) 

    axs[2].boxplot(all_diffusion.T, tick_labels=[f'dif {i}' for i in range(len(all_diffusion))]) 

    fig.savefig('eval/eval_boxplot_all.png', dpi=200)
    
    