import numpy as np
import yaml
from pathlib import Path

import torch

from simulators.cf_frontnet_pt import CFSim

def gen_T(coeffs):
    T = np.zeros((3,3))
    T[0, 0] = T[1, 1] = coeffs[0] # sf
    T[0, 2] = coeffs[1] # tx
    T[1, 2] = coeffs[2] # ty
    T[2, 2] = 1.

    return T

def get_targets(patch_path):
    parent_folder = patch_path.parent
    with open(parent_folder / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).float()
    return targets

def get_Ts(patch_path):
    parent_folder = patch_path.parent
    T_coeffs = np.load(parent_folder / 'positions_norm.npy') # shape: [sf, tx, ty], epochs, num_targets, 1, 1
    # # should shape: num_targets, [sf, tx, ty]
    T_coeffs = T_coeffs[:, -1, :, 0, 0].T

    Ts = [gen_T(coeffs) for coeffs in T_coeffs]
    return Ts

def _calc_loss(target, patch, T, img, sim):
    mod_img_pt = sim.pt_project_patch(patch, T, img) 
    x, y, z, yaw = sim.pose_estimator(mod_img_pt)
    predicted_pose = torch.hstack((x, y, z))[0].detach().cpu()

    mse = torch.nn.functional.mse_loss(target, predicted_pose).item()
    return mse

def calc_loss(targets, patch, Ts, img, sim):
    losses = np.array([_calc_loss(target, patch, T, img, sim) for target, T in zip(targets, Ts)])
    return np.mean(losses)

def loss_dataset(patch, patch_path, dataset, sim):
    targets = get_targets(patch_path)
    Ts = get_Ts(patch_path)

    losses = []
    for data in dataset:
        img, _ = data
        losses.append(calc_loss(targets, patch, Ts, img, sim))

    return losses

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    sim = CFSim()

    dataset_path = "/home/hanfeld/flying_adversarial_patch/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    _, dataset = sim.load_dataset(dataset_path, batch_size=1, train_set_size=0.9)

    # base_img, gt = dataset.dataset.__getitem__(0)

    # with open('dataset.yaml') as f:
    #     settings = yaml.load(f, Loader=yaml.FullLoader)
    
    # patch_size = settings['patch']['size']
    path = Path(f'/home/hanfeld/flying_adversarial_patch/results/dataset80x80/')

    patches_paths = list(path.glob('[0-9]*/patches.npy'))
    patches_paths.sort(key=lambda path: int(path.parent.name))

    patches = np.array([np.load(file_path)[-1][0][0]*255. for file_path in patches_paths])

    # losses = np.array([loss_dataset(patch, path, dataset, sim) for patch, path in zip(patches, patches_paths)])
    losses = []
    from tqdm import trange
    for idx in trange(len(patches)):
        losses.append(loss_dataset(patches[idx], patches_paths[idx], dataset, sim))
        if idx % 10 == 0:
            np.save('eval/gtFAP_losses.npy', np.array(losses))
    
    losses = np.array(losses)

    fig, ax = plt.subplots(1, 1)
    ax.boxplot(losses.flatten(), tick_labels=['gt FAP'])
    ax.set_ylabel('MSE(target, prediction)')
    fig.savefig('eval/eval_boxplot_FAPs.png', dpi=200)