import torch
import numpy as np
from diffusion.diffusion_model import DiffusionModel

from cf_simulator import CFSim, SimulatorThread

import time

from util import scale_tx_ty

import cv2

from threading import Thread
from collections import deque

import os


from matplotlib import pyplot as plt

class DiffusionThread(Thread):
    def __init__(self, diffusion_model_path, target_queue):
        super().__init__()
        # self.camera_image_queue = camera_image_queue
        # self.project_patch = project_patch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = DiffusionModel(self.device, lr=1e-5)
        self.diffusion_model.load(diffusion_model_path)

        self.patch_queue = deque(maxlen=1)

        self.target_queue = target_queue

        self._stay_alive = True

    def run(self):
        while self._stay_alive:
            if not self.target_queue:
                time.sleep(0.01)
                continue
            # sf = np.random.uniform(0.4,0.8,1)
            # tx = np.random.uniform(0.,1.,1)
            # ty = np.random.uniform(0.,1.,1)

            sf = np.array([0.6])
            tx = np.array([0.5])
            ty = np.array([0.5])


            #TODO: generate target based on target trajectory
            # x = np.random.uniform(0,2,1)
            # y = np.random.uniform(-1,1,1,)
            # z = np.random.uniform(-0.5,0.5,1,)
            # x = np.array([1.])
            # y = np.array([0.])
            # z = np.array([0.])

            x, y, z = self.target_queue[0]

            r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z)).T, dtype=torch.float32)

            samples = self.diffusion_model.sample(1, r_targets, self.device, patch_size=[80,80], n_steps=1_000).detach().to('cpu').numpy()
            patch = samples[0, 0] * 255.

            scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80)
            self.patch_queue.append((patch, sf, scaled_tx, scaled_ty))
            print("Bing new patch!")

           
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

            # TODO: sticking to first image for now
            new_img, _ = self.dataset.dataset.__getitem__(i)
            # print(new_img.shape, new_img.min(), new_img.max())
            # new_img = torch.ones((1, 96, 160), dtype=torch.float32) * 255.
            # i += 1
            # if i >= len(self.dataset):
            #     i = 0
            self.camera_image_queue.append(new_img[0])
            time.sleep(5)

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
        # i = 0
        while self._stay_alive:
            if self.camera_image_queue and self.patch_queue:
                camera_image = self.camera_image_queue[0]
                patch, sf, scaled_tx, scaled_ty = self.patch_queue[0]

                T = np.zeros((3, 3))
                T[0, 0] = sf[0] # shape is (1,) TODO!
                T[1, 1] = sf[0]
                T[0, 2] = scaled_tx[0]
                T[1, 2] = scaled_ty[0]
                T[2, 2] = 1
                mod_img = self.project_patch(patch, T, camera_image)
                # print(mod_img.shape, mod_img.min(), mod_img.max())

                self.camera_image_queue.appendleft(mod_img)
                
                # plt.imshow(mod_img.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
                # plt.savefig(f"results/simulation/patched_image_{i:04d}.png")
                # i += 1
                #visualize with cv2
                # cv2.imshow('modified image', mod_img[0, 0].cpu().numpy())
                # cv2.waitKey(0)
                time.sleep(0.01)

    def close(self):
        self._stay_alive = False
        # self.join()


class AttackerPolicyThread(Thread):
    def __init__(self, drone_pose, target_trajectory):
        super().__init__()
        self.drone_pose = drone_pose
        self.target_trajectory = target_trajectory
        self.current_target = deque(maxlen=1)
        self.index_reached = 0
        self._stay_alive = True

    def run(self):
        while self._stay_alive:
            if self.index_reached < len(self.target_trajectory):
                target_position = self.target_trajectory[self.index_reached]

                # check if the drone pose is close to the target
                if np.linalg.norm(self.drone_pose - target_position) < 0.1:
                    self.index_reached += 1
                    print("Target reached: ", target_position)
                    target_position = self.target_trajectory[self.index_reached]
                    print("Moving towards: ", target_position)
                
                # to keep x constant -> target x should be 1.
                # to lower x -> target x should be > 1.
                # to increase x -> target x should be < 1.
                # to move left -> target y should be > 0.
                # to move right -> target y should be < 0.
                # to move up -> target z should be > 0.
                # to move down -> target z should be < 0.

                change_necessary = target_position - self.drone_pose[0]
                # print("Change necessary: ", change_necessary)

                # calculate target x based on change_necessary[0]
                # keep in range [0, 2]
                target_x = np.maximum(np.minimum(1. - change_necessary[0], 2.), 0.0)
                # calculate target y based on change_necessary[1]
                # keep in range [-1, 1]
                target_y = np.maximum(np.minimum(0. - change_necessary[1], 1.), -1.0)

                # calculate target z based on change_necessary[2]
                # keep in range [-0.5, 0.5]
                target_z = np.maximum(np.minimum(0. - change_necessary[2], 0.5), -0.5)

                # print("Target: ", target_x, target_y, target_z)

                self.current_target.append(([target_x], [target_y], [target_z]))
                time.sleep(0.1)

            else:
                print("All targets reached")
                self._stay_alive = False
                break

    def close(self):
        self._stay_alive = False
        self.join()

if __name__ == "__main__":

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # diffusion_model = DiffusionModel(device, lr=1e-5)
    # diffusion_model.load('results/diffusion_training/trained_model.pth')

    os.makedirs('results/simulation', exist_ok=True)

    model_path = 'results/diffusion_training/frontnet1k_1kds_2kepochs.pth'


    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    model_type = 'frontnet'
    cf_sim = CFSim(model_type, dataset_path)

    camera_thread = CameraThread(cf_sim.dataset)
    # camera_thread.start()


    from camera import Camera
    cam = Camera('camera_calibration.yaml')

    sim_thread = SimulatorThread(cf_sim.sim_new_pose, camera_thread.camera_image_queue, cam.point_from_xyz)
    # sim_thread.start()


    target_trajectory = np.array([[0.0, 0.25, 1., 0.0],
                                  [0.0, 0.50, 1., 0.0],
                                  [0.0, 0.75, 1., 0.0],
                                  [0.0, 1.00, 1., 0.0],
                                  [0.0, 0.75, 1., 0.0],
                                  [0.0, 0.50, 1., 0.0],
                                  [0.0, 0.25, 1., 0.0],
                                  [0.0, 0.00, 1., 0.0],
                                  [0.0, -0.25, 1., 0.0],
                                  [0.0, -0.50, 1., 0.0],
                                  [0.0, -0.75, 1., 0.0],
                                  [0.0, -1.00, 1., 0.0],
                                  [0.0, -0.75, 1., 0.0],
                                  [0.0, -0.50, 1., 0.0],
                                  [0.0, -0.25, 1., 0.0],
                                  [0.0, 0.00, 1., 0.0]])

    attacker_policy = AttackerPolicyThread(sim_thread.drone_pose, target_trajectory)
    attacker_policy.start()


    diffusion_thread = DiffusionThread(model_path, attacker_policy.current_target)
    diffusion_thread.start()

    while not diffusion_thread.patch_queue:
        time.sleep(0.1)

    camera_thread.start()
    time.sleep(0.01)
    sim_thread.start()

    manipulator_thread = ManipulatorThread(camera_thread.camera_image_queue, diffusion_thread.patch_queue, cf_sim.project_patch)
    manipulator_thread.start()

    time_start = time.time()

    while time.time() - time_start < 30:
        # wait for 20 seconds
        time.sleep(0.1)

    print("Stopping threads...")
    sim_thread.close()
    camera_thread.close()
    manipulator_thread.close()
    diffusion_thread.close()
    attacker_policy.close()

    # print(sim_thread.all_poses)

    all_poses = np.array(sim_thread.all_poses)
    # print(all_poses.shape)

    # plot 3d positions stored in all_poses (consists of (timestamp, pose))  with matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(all_poses[:, 1], all_poses[:, 2], all_poses[:, 3])
    ax.plot(all_poses[:, 1], all_poses[:, 2], all_poses[:, 3])
    ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], color='red', linestyle='dotted') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
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