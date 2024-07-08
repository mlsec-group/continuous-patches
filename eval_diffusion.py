import numpy as np
import yaml
from pathlib import Path

import torch

from simulators.cf_frontnet_pt import CFSim


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sim = CFSim()

    # with open('dataset.yaml') as f:
    #     settings = yaml.load(f, Loader=yaml.FullLoader)
    
    # patch_size = settings['patch']['size']
    path = Path(f'/home/hanfeld/flying_adversarial_patch/results/dataset80x80/')

    patches_paths = list(path.glob('[0-9]*/patches.npy'))
    patches_paths.sort(key=lambda path: int(path.parent.name))

    opt_positions_paths = list(path.glob('[0-9]*/positions_norm.npy'))
    opt_positions_paths.sort(key=lambda path: int(path.parent.name))

    patches = np.array([np.load(file_path)[-1][0][0] for file_path in patches_paths[:1]])

    positions = np.load(opt_positions_paths[0]) # shape: [sf, tx, ty], epochs, num_targets, 1, 1
    # should shape: num_targets, [sf, tx, ty]
    positions = positions[:, -1, :, 0, 0].T
    print(positions)

    dataset_path = "/home/hanfeld/flying_adversarial_patch/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    dataset = sim.load_dataset(dataset_path, train_set_size=1)

    base_img, gt = dataset.dataset.__getitem__(0)
    print(base_img.shape)

    T = np.zeros((3,3))
    T[0, 0] = T[1, 1] = positions[0][0] # sf
    T[0, 2] = positions[0][1] # tx
    T[1, 2] = positions[0][2] # ty
    T[2, 2] = 1.

    print(T)

    mod_img_cv = sim.project_patch(patches[0], T, base_img[0]/255.)
    print(mod_img_cv.shape)

    # plt.imshow(mod_img, cmap='gray')
    # plt.savefig('eval_cv.png', dpi=200)

    mod_img_pt = sim.pt_project_patch(patches[0], T, base_img/255.)[0][0].numpy()
    print(mod_img_pt.shape)

    np.testing.assert_almost_equal(mod_img_cv, mod_img_pt, decimal=1)

    # plt.imshow(mod_img_pt[0][0], cmap='gray')
    # plt.savefig('eval_pt.png', dpi=200)