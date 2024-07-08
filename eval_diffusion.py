import numpy as np
import yaml
from pathlib import Path

import torch

from simulators.cf_frontnet_pt import CFSim

def gen_T(coeffs):
    print(coeffs)
    T = np.zeros((3,3))
    T[0, 0] = T[1, 1] = coeffs[0] # sf
    T[0, 2] = coeffs[1] # tx
    T[1, 2] = coeffs[2] # ty
    T[2, 2] = 1.

    return T

def get_targets(patch_path, device=torch.device('cpu')):
    parent_folder = patch_path.parent
    with open(parent_folder / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).float().to(device)

    return targets

def get_Ts(patch_path):
    parent_folder = patch_path.parent
    T_coeffs = np.load(parent_folder / 'positions_norm.npy') # shape: [sf, tx, ty], epochs, num_targets, 1, 1
    # # should shape: num_targets, [sf, tx, ty]
    T_coeffs = T_coeffs[:, -1, :, 0, 0].T

    Ts = [gen_T(coeffs) for coeffs in T_coeffs]
    return Ts

def calc_loss(target, patch, T, img, sim):
    mod_img_pt = sim.pt_project_patch(patch*255., T, img) 
    x, y, z, yaw = sim.pose_estimator(mod_img_pt)
    predicted_pose = torch.hstack((x, y, z))[0]

    mse = torch.nn.functional.mse_loss(target, predicted_pose).detach().cpu().item()
    return mse


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    sim = CFSim()

    dataset_path = "/home/hanfeld/flying_adversarial_patch/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    dataset = sim.load_dataset(dataset_path, train_set_size=1)

    base_img, gt = dataset.dataset.__getitem__(0)

    # with open('dataset.yaml') as f:
    #     settings = yaml.load(f, Loader=yaml.FullLoader)
    
    # patch_size = settings['patch']['size']
    path = Path(f'/home/hanfeld/flying_adversarial_patch/results/dataset80x80/')

    patches_paths = list(path.glob('[0-9]*/patches.npy'))
    patches_paths.sort(key=lambda path: int(path.parent.name))

    patches = np.array([np.load(file_path)[-1][0][0] for file_path in patches_paths[:1]])

    targets = get_targets(patches_paths[0], device)
    print(targets)

    Ts = get_Ts(patches_paths[0])
    print(Ts)

    loss = calc_loss(targets[0], patches[0], Ts[0], base_img, sim)
    print(loss)

    # print(base_img.shape)



    # print(T)

    # # mod_img_cv = sim.project_patch(patches[0], T, base_img[0]/255.)
    # # print(mod_img_cv.shape)

    # # plt.imshow(mod_img, cmap='gray')
    # # plt.savefig('eval_cv.png', dpi=200)

    # mod_img_pt = sim.pt_project_patch(patches[0], T, base_img/255.)
    # print(mod_img_pt.shape)

    # 

    # 

    # print(predicted_pose)



    # np.testing.assert_almost_equal(mod_img_cv, mod_img_pt, decimal=1)

    # plt.imshow(mod_img_pt[0][0], cmap='gray')
    # plt.savefig('eval_pt.png', dpi=200)

