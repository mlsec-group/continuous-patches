import torch
import numpy as np
from diffusion.diffusion_model import DiffusionModel

from cf_simulator import CFSim, SimulatorThread

import time

from util import scale_tx_ty

import cv2

from threading import Thread
from collections import deque

class DiffusionThread(Thread):
    def __init__(self, diffusion_model_path):
        super().__init__()
        # self.camera_image_queue = camera_image_queue
        # self.project_patch = project_patch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = DiffusionModel(self.device, lr=1e-5)
        self.diffusion_model.load(diffusion_model_path)

        self.patch_queue = deque(maxlen=1)

        self._stay_alive = True

    def run(self):
        while self._stay_alive:
            if not self.camera_image_queue.empty():
                # camera_image = self.camera_image_queue.popleft()
                # print(camera_image.shape, camera_image.min(), camera_image.max())

                sf = np.random.uniform(0.4,0.8,n_samples)
                tx = np.random.uniform(0.,1.,n_samples)
                ty = np.random.uniform(0.,1.,n_samples)
                #TODO: generate target based on target trajectory
                x = np.random.uniform(0,2,n_samples)
                y = np.random.uniform(-1,1,n_samples,)
                z = np.random.uniform(-0.5,0.5,n_samples,)

                r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z)).T, dtype=torch.float32)

                samples = diffusion_model.sample(n_samples, r_targets, device, patch_size=[80,80], n_steps=1_000).detach().to('cpu').numpy()
                patch = samples[0, 0] * 255.

                scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80)
                self.patch_queue.append((patch, sf, scaled_tx, scaled_ty))

                # T = np.zeros((3, 3))
                # T[0, 0] = sf
                # T[1, 1] = sf
                # T[0, 2] = scaled_tx
                # T[1, 2] = scaled_ty
                # T[2, 2] = 1

                # mod_img = self.project_patch(patch, T, camera_image)
                # print(mod_img.shape, mod_img.min(), mod_img.max())

                # self.camera_image_queue.append(mod_img)
                # time.sleep(0.01)

    def close(self):
        self._stay_alive = False
        self.join()

class CameraThread(Thread):
    def __init__(self, dataset):
        super().__init__()
        self.camera_image_queue = deque(maxlen=1)
        self.dataset = dataset
        self._stay_alive = True

    def run(self):
        i = 0
        while self._stay_alive:
            # rnd_idx = np.random.randint(0, len(self.dataset))
            # new_img, _ = self.dataset.dataset.__getitem__(rnd_idx)

            new_img, _ = self.dataset.dataset.__getitem__(i)
            i += 1
            if i >= len(self.dataset):
                i = 0
            self.camera_image_queue.append(new_img[0])
            time.sleep(0.1)

    def close(self):
        self._stay_alive = False
        self.join()    

class ManipulatorThread(Thread):
    def __init__(self, camera_image_queue, patch_queue, project_patch):
        super().__init__()
        self.camera_image_queue = camera_image_queue
        self.patch_queue = patch_queue
        self.project_patch = project_patch
        self._stay_alive = True

    def run(self):
        while self._stay_alive:
            if self.camera_image_queue and self.patch_queue:
                camera_image = self.camera_image_queue[0]
                patch, sf, scaled_tx, scaled_ty = self.patch_queue[0]
            # if not self.camera_image_queue.empty() and not self.patch_queue.empty():
            #     camera_image = self.camera_image_queue.popleft()
            #     patch, sf, scaled_tx, scaled_ty = self.patch_queue.popleft()

                T = np.zeros((3, 3))
                T[0, 0] = sf
                T[1, 1] = sf
                T[0, 2] = scaled_tx
                T[1, 2] = scaled_ty
                T[2, 2] = 1
                mod_img = self.project_patch(patch, T, camera_image)
                print(mod_img.shape, mod_img.min(), mod_img.max())

                self.camera_image_queue.appendleft(mod_img)
                #visualize with cv2
                # cv2.imshow('modified image', mod_img[0, 0].cpu().numpy())
                # cv2.waitKey(0)
                time.sleep(0.01)

    def close(self):
        self._stay_alive = False
        # self.join()


if __name__ == "__main__":

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # diffusion_model = DiffusionModel(device, lr=1e-5)
    # diffusion_model.load('results/diffusion_training/trained_model.pth')

    model_path = 'results/diffusion_training/trained_model.pth'


    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    model_type = 'frontnet'
    cf_sim = CFSim(model_type, dataset_path)

    base_img = cf_sim.base_img[0]
    print(base_img.shape, base_img.min(), base_img.max())

    camera_thread = CameraThread(cf_sim.dataset)
    camera_thread.start()

    sim_thread = SimulatorThread(cf_sim.sim_new_pose, camera_thread.camera_image_queue)
    sim_thread.start()

    time_start = time.time()

    while time.time() - time_start < 5:
        # wait for 20 seconds
        time.sleep(0.1)

    sim_thread.close()
    camera_thread.close()

    # print(sim_thread.all_poses)

    all_poses = np.array(sim_thread.all_poses)
    # print(all_poses.shape)

    # plot 3d positions stored in all_poses (consists of (timestamp, pose))  with matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(all_poses[:, 1], all_poses[:, 2], all_poses[:, 3])
    # as trajectory
    ax.plot(all_poses[:, 1], all_poses[:, 2], all_poses[:, 3])
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    fig.savefig('3d_positions.png')


    # sim_thread.start()

    # sim_thread.update(cf_sim.base_img[0])

    # timestamp_1 = time.time()
    # n_samples = 1
    # sf = np.random.uniform(0.4,0.8,n_samples)
    # tx = np.random.uniform(0.,1.,n_samples)
    # ty = np.random.uniform(0.,1.,n_samples)
    # x = np.random.uniform(0,2,n_samples)
    # y = np.random.uniform(-1,1,n_samples,)
    # z = np.random.uniform(-0.5,0.5,n_samples,)

    # r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z)).T, dtype=torch.float32)

    # samples = diffusion_model.sample(n_samples, r_targets, device, patch_size=[80,80], n_steps=1_000).detach().to('cpu').numpy()
    # print(samples.min(), samples.max(), samples.shape)

    # patch = samples[0, 0] * 255.

    # print("Sampling time: ", time.time() - timestamp_1)


    # scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80)
    
    # T = np.zeros((3, 3))
    # T[0, 0] = sf
    # T[1, 1] = sf
    # T[0, 2] = scaled_tx
    # T[1, 2] = scaled_ty
    # T[2, 2] = 1

    # mod_img = cf_sim.project_patch(patch, T, base_img).to(cf_sim.device)
    # mod_img = mod_img.unsqueeze(0).unsqueeze(0).float()
    # print(mod_img.shape, mod_img.dtype)

    # # plot the modified image
    # from matplotlib import pyplot as plt
    # plt.imshow(mod_img.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    # plt.savefig("patched_image.png")


    # predicted_pose = cf_sim.sim_new_pose(mod_img)
    # print(predicted_pose)