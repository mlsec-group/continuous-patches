import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import linprog

from diffusion.diffusion_model import DiffusionModel
from util import scale_tx_ty
from attacks import calc_eval_loss, get_transformation
from util import load_model, load_dataset

N = 1000
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    test_set = load_dataset(
        path=dataset_path,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        train=False,
        num_workers=0
    )
    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    model = load_model(path=model_path, device=DEVICE, config=model_config)
    model.eval()

    with open("/shares/datasets/continuous_patches/frontnet1k.pickle", "rb") as f:
        patch_dataset = pickle.load(f)
    
    diffusion_model = DiffusionModel(DEVICE)
    diffusion_model.load(f'results/diffusion_training.bak/trained_model.pth')

    print(f"Successfully loaded, using device {DEVICE}")
    patches = torch.as_tensor(np.array([patch for patch, target, transformation in patch_dataset])).float()
    targets = torch.as_tensor([target.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
    transformations = torch.as_tensor([transformation.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
    combineds = torch.hstack([targets, transformations])

    print(targets.shape, transformations.shape)

    def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))
    
    def closest_distance(target, transformation):
        combined = torch.hstack([target, transformation]).squeeze()
        distances = dist(combined, combineds)
        order = torch.argsort(distances)
        combined = combined.numpy()
        return distances.min().item()
    
    def lookup_patch(target, transformation):
        combined = torch.hstack([target, transformation]).squeeze()
        distances = dist(combined, combineds)
        closest = torch.argmin(distances)
        return patches[closest]

    def interpolated_patch(target, transformation):
        combined = torch.hstack([target, transformation]).squeeze()
        distances = dist(combined, combineds)
        order = torch.argsort(distances)
        combined = combined.numpy()
        ordered_combined = combineds[order].numpy()
        for n in range(1, len(ordered_combined)):
            result = linprog(
                bounds=[(0,1)]*n,
                c=np.ones(n),
                A_eq=ordered_combined[:n].T,
                b_eq=combined,
            )
            if result.success and result.fun <= 1:
                coeffs = torch.as_tensor(result.x)
                candidate = (patches[order][:n].permute(1, 2, 0) * coeffs[None, None, :]).sum(axis=-1)
                return candidate.float()
        return patches[order[0]] # backup: closest patch

    def generated_patch(target, transformation):
        r_targets = torch.hstack([transformation, target])
        samples = diffusion_model.sample(1, r_targets, DEVICE, patch_size=(80, 80), n_steps=100).detach()
        return samples.squeeze()

    xs = np.random.uniform(0., 2., N)
    ys = np.random.uniform(-1., 1., N)
    zs = np.random.uniform(-.5, .5, N)
    sfs = np.random.uniform(.4, .8, N)
    txs = np.random.uniform(0., 1., N)
    tys = np.random.uniform(0., 1., N)

    eval_targets = torch.as_tensor(np.stack([xs, ys, zs]).T).float()
    eval_transformations = torch.as_tensor(np.stack([sfs, txs, tys]).T).float()

    approaches = {
        "Corpus": lookup_patch,
        "Interpolation": interpolated_patch,
        "Generation": generated_patch,
    }
    approach_colors = {
        "Corpus": "b",
        "Interpolation": "r",
        "Generation": "g",
    }
    losses = {a: [] for a in approaches.keys()}
    distances = []
    for target, transformation in tqdm(zip(eval_targets, eval_transformations), total=len(eval_targets)):
        sf, tx, ty = transformation
        distances.append(closest_distance(target, transformation))
        loss_target = target.to(DEVICE)
        scale_tx, scale_ty = scale_tx_ty(sf, tx, ty, 80)
        transformation_matrix = get_transformation(sf, scale_tx[None], scale_ty[None]).to(DEVICE)
        
        for approach_name, approach_fn in approaches.items():
            patch = approach_fn(target, transformation)
            patch = patch.to(DEVICE)
            loss = calc_eval_loss(test_set, patch[None, None], transformation_matrix, model, loss_target, model_name="frontnet", quantized=False)
            losses[approach_name].append(loss.detach().cpu().item())
    distances = np.asarray(distances)
    losses = {a: np.asarray(l) for a, l in losses.items()}

    for approach_name, approach_losses in losses.items():
        print(f"{approach_name}: {approach_losses.mean()} (std of mean: {np.std(approach_losses)/N**0.5})")

    fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax2.scatter(distances, losses)
    # ax.hist(distances, bins=10)

    # mean line
    edges = np.histogram_bin_edges(distances, bins=10)
    masks = [(left <= distances) & (distances < right) for left, right in zip(edges, edges[1:])]
    midpoints = [(left + right) / 2 for left, right, mask in zip(edges, edges[1:], masks) if np.sum(mask) > 0]
    
    for approach_name, approach_losses in losses.items():
        means = np.asarray([np.mean(approach_losses[mask]) for mask in masks if np.sum(mask) > 0])
        stds = np.asarray([1.64*np.std(approach_losses[mask])/np.sqrt(np.sum(mask)) for mask in masks if np.sum(mask) > 0])
        ax.plot(midpoints, means, c=approach_colors[approach_name], label=approach_name)
        ax.fill_between(midpoints, means-stds, means+stds, color=approach_colors[approach_name], alpha=0.2)
    fig.savefig("single_step_results.png")