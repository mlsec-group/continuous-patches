import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import linprog

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

    with open("/shares/datasets/continuous_patches/frontnet80x80.pickle", "rb") as f:
        patch_dataset = pickle.load(f)
    
    print(f"Successfully loaded, using device {DEVICE}")
    patches = torch.as_tensor(np.array([patch for patch, target, transformation in patch_dataset])).float()
    targets = torch.as_tensor([target.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
    transformations = torch.as_tensor([transformation.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
    combineds = torch.hstack([targets, transformations])

    def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))

    def interpolated_patch(target, transformation):
        combined = torch.hstack([target, transformation]).squeeze()
        distances = dist(combined, combineds)
        order = torch.argsort(distances)
        combined = combined.numpy()
        ordered_combined = combineds[order].numpy()
        for n in range(1, max(100, len(order))):
            result = linprog(
                bounds=[(0,1)]*n,
                c=np.ones(n),
                A_eq=ordered_combined[:n].T,
                b_eq=combined,
            )
            if result.success and result.fun <= 1:
                coeffs = torch.as_tensor(result.x)
                candidate = (patches[order][:n].permute(1, 2, 0) * coeffs[None, None, :]).sum(axis=-1)
                return candidate.float(), combineds[order[0]]
        return patches[order[0]], combineds[order[0]]

    xs = np.random.uniform(0., 2., N)
    ys = np.random.uniform(-1., 1., N)
    zs = np.random.uniform(-.5, .5, N)
    sfs = np.random.uniform(.4, .8, N)
    txs = np.random.uniform(0., 1., N)
    tys = np.random.uniform(0., 1., N)

    eval_targets = torch.as_tensor(np.stack([xs, ys, zs]).T).float()
    eval_transformations = torch.as_tensor(np.stack([sfs, txs, tys]).T).float()

    losses = []
    distances = []
    pbar = tqdm(zip(eval_targets, eval_transformations), total=len(eval_targets))
    for target, transformation in pbar:
        sf, tx, ty = transformation
        patch, patch_combined = interpolated_patch(target, transformation)
        distances.append(dist(patch_combined, torch.hstack([target, transformation]).squeeze()))
        patch = patch.to(DEVICE)
        target = target.to(DEVICE)
        scale_tx, scale_ty = scale_tx_ty(sf, tx, ty, 80)
        transformation_matrix = get_transformation(sf, scale_tx[None], scale_ty[None]).to(DEVICE)
        loss = calc_eval_loss(test_set, patch[None, None], transformation_matrix, model, target, model_name="frontnet", quantized=False)
        losses.append(loss.detach().cpu().item())
        pbar.set_description(f"{sum(losses)/len(losses):.3f}")
    distances = np.asarray(distances)
    losses = np.asarray(losses)

    print(f"mean over {N} samples: {losses.mean()} (ci: {np.std(losses)/len(losses)**0.5})")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    # ax.scatter(distances, losses)
    ax.hist(distances, bins=10)

    # mean line
    edges = np.histogram_bin_edges(distances, bins=10)
    masks = [(left <= distances) & (distances < right) for left, right in zip(edges, edges[1:])]
    midpoints = [(left + right) / 2 for left, right, mask in zip(edges, edges[1:], masks) if np.sum(mask) > 0]
    means = np.asarray([np.mean(losses[mask]) for mask in masks if np.sum(mask) > 0])
    stds = np.asarray([1.64*np.std(losses[mask])/np.sqrt(np.sum(mask)) for mask in masks if np.sum(mask) > 0])
    ax2.plot(midpoints, means, c="b")
    ax2.fill_between(midpoints, means-stds, means+stds, color="b", alpha=0.2)
    fig.savefig("interpolation_results.png")