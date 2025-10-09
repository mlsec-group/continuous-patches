import numpy as np
# import matplotlib as mpl
# mpl.use('pgf')
from matplotlib import pyplot as plt

from pathlib import Path


exp_path = Path('results/change_y/interpolated/frontnet/')

img_folders = list(exp_path.glob('image_[0-9]*/'))

for img_folder in img_folders:
        drone_poses_files = list(img_folder.glob('simulation_[0-9]*/drone_poses.npy'))
        all_drone_poses = [np.load(file) for file in drone_poses_files]
        target_poses = np.load(img_folder / 'simulation_0/target_poses.npy')      

# path = Path('results/diffusion/image_72/')
# # results/diffusion/image_0/simulation_2/drone_poses.npy

# # get all files with name drone_poses.npy from results/diffusion/image_0/simulation_[0-9]/
# files = list(path.glob('simulation_[0-9]*/drone_poses.npy'))
# print(len(files))

# target_poses = np.load(path / 'simulation_0/target_poses.npy')
# print(target_poses.shape)

# all_drone_poses = [np.load(file) for file in files]
# print(len(all_drone_poses))
# print(all_drone_poses[0].shape)


        fig, axs = plt.subplots(2, 1)
        for i in range(len(all_drone_poses)):
            drone_poses = all_drone_poses[i]
            axs[0].plot(drone_poses[:, 0], drone_poses[:, 1])
            axs[1].plot(drone_poses[:, 0], drone_poses[:, 2])
        # axs[2].plot(drone_poses[:, 0], drone_poses[:, 3])

        axs[0].plot(target_poses[:, 0], target_poses[:, 1], label='Target', linestyle='dotted', color='black')
        axs[1].plot(target_poses[:, 0], target_poses[:, 2], label='Target', linestyle='dotted', color='black')
        #axs[2].plot(target_poses[:, 0], target_poses[:, 3], label='Target', linestyle='dotted', color='black')

        axs[0].set_ylabel('x [m]')
        axs[1].set_ylabel('y [m]')
        #axs[2].set_ylabel('z [m]')
        axs[1].set_xlabel('Time [s]')
        axs[0].legend()

        axs[0].set_ylim(-1, 1)
        axs[1].set_ylim(-1, 1)
        # axs[2].set_ylim(-1, 1)

        fig.tight_layout()
        fig.savefig(img_folder / 'result.png', dpi=300)
        fig.savefig(img_folder / 'result.pdf', dpi=300)
        plt.close(fig)