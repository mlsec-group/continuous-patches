import torch
import pickle
import numpy as np

from scipy.optimize import linprog

from diffusion.diffusion_model import DiffusionModel

from cf_simulator import CFSim, SimulatorThread

import time

from util import scale_tx_ty, bb2camera, line_plane_intersection

from pathlib import Path
import argparse

import cv2
import rowan

from threading import Thread
from collections import deque

import os

import sys
sys.path.insert(0,'uav_trajectories/scripts')
from uav_trajectory import Trajectory

# import matplotlib as mpl
from matplotlib import pyplot as plt
# mpl.use('pgf')

class DiffusionThread(Thread):
    def __init__(self, diffusion_model_path, target_queue):
        super().__init__()
        # self.camera_image_queue = camera_image_queue
        # self.project_patch = project_patch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = DiffusionModel(self.device, lr=1e-5)
        self.diffusion_model.load(diffusion_model_path)

        # gt_dataset = pickle.load("frontnet1k.pickle")
        # self.patches = torch.as_tensor(np.array([patch for patch, target, transformation in patch_dataset])).float()
        # targets = torch.as_tensor([target.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        # transformations = torch.as_tensor([transformation.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        # self.combined = torch.hstack([targets, transformations])


        self.patch_queue = deque(maxlen=1)

        self.target_queue = target_queue

        self._stay_alive = True

    def run(self):
        while self._stay_alive:
            if not self.target_queue:
                time.sleep(0.01)
                continue

            sf, tx, ty, patch_ul_x, patch_ul_y, projector_height, projector_width, x, y, z  = self.target_queue[0]
            # print(sf, tx, ty, x, y, z)
            

            condiditions = torch.tensor(np.stack(([sf], [tx], [ty], [x], [y], [z])).T, dtype=torch.float32)
            # print(condiditions.shape)

            samples = self.diffusion_model.sample(1, condiditions, self.device, patch_size=[80,80], n_steps=25).detach().to('cpu').numpy()
            patch = samples[0, 0] * 255.

            scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80, (projector_height, projector_width))

            tx_img = patch_ul_x + scaled_tx
            ty_img = patch_ul_y + scaled_ty

            self.patch_queue.append((patch, sf, tx_img, ty_img))
            # print("Bing new patch!")

           
            # time.sleep(0.05)

    def close(self):
        self._stay_alive = False
        self.join()


class ClosestPatchThread(Thread):
    def __init__(self, patches_path, target_queue):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = DiffusionModel(self.device, lr=1e-5)
        with open(patches_path, "rb") as f:
            patch_dataset = pickle.load(f)

        self.patches = np.array([patch for patch, target, transformation in patch_dataset], dtype=np.float32)
        targets = torch.as_tensor([target.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        transformations = torch.as_tensor([transformation.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        self.combineds = torch.hstack([targets, transformations])


        self.patch_queue = deque(maxlen=1)
        self.target_queue = target_queue
        self._stay_alive = True

    @staticmethod
    def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))

    def run(self):
        while self._stay_alive:
            if not self.target_queue:
                time.sleep(0.01)
                continue

            sf, tx, ty, patch_ul_x, patch_ul_y, projector_height, projector_width, x, y, z  = self.target_queue[0]
            combined = torch.tensor(np.stack(([x], [y], [z], [sf], [tx], [ty])).T, dtype=torch.float32)
            distances = ClosestPatchThread.dist(combined, self.combineds)
            closest = torch.argmin(distances)
            patch = self.patches[closest] * 255.

            scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80, (projector_height, projector_width))

            tx_img = patch_ul_x + scaled_tx
            ty_img = patch_ul_y + scaled_ty

            self.patch_queue.append((patch, sf, tx_img, ty_img))
            # print("Bing new patch!")
            # time.sleep(0.05)

    def close(self):
        self._stay_alive = False
        self.join()


class InterpolatedPatchThread(Thread):
    def __init__(self, patches_path, target_queue):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = DiffusionModel(self.device, lr=1e-5)
        with open(patches_path, "rb") as f:
            patch_dataset = pickle.load(f)

        self.patches = torch.as_tensor(np.array([patch for patch, target, transformation in patch_dataset])).float()
        targets = torch.as_tensor([target.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        transformations = torch.as_tensor([transformation.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        self.combineds = torch.hstack([targets, transformations])


        self.patch_queue = deque(maxlen=1)
        self.target_queue = target_queue
        self._stay_alive = True

    @staticmethod
    def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))

    def run(self):
        while self._stay_alive:
            if not self.target_queue:
                time.sleep(0.01)
                continue

            sf, tx, ty, patch_ul_x, patch_ul_y, projector_height, projector_width, x, y, z  = self.target_queue[0]
            combined = torch.tensor(np.stack(([x], [y], [z], [sf], [tx], [ty])).T, dtype=torch.float32)
            distances = InterpolatedPatchThread.dist(combined, self.combineds)
            order = torch.argsort(distances)
            ordered_combined = self.combineds[order].numpy()
            patch = self.patches[order[0]].numpy()
            for n in range(1, len(order)):
                result = linprog(
                    bounds=[(0,1)]*n,
                    c=np.ones(n),
                    A_eq=ordered_combined[:n].T,
                    b_eq=combined,
                )
                if result.success and result.fun <= 1:
                    coeffs = torch.as_tensor(result.x)
                    candidate = (self.patches[order][:n].permute(1, 2, 0) * coeffs[None, None, :]).sum(axis=-1)
                    patch = candidate.float().numpy()
                    break
            patch *= 255.
            patch = np.clip(patch, 0, 255.)
            
            scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80, (projector_height, projector_width))

            tx_img = patch_ul_x + scaled_tx
            ty_img = patch_ul_y + scaled_ty

            self.patch_queue.append((patch, sf, tx_img, ty_img))
            # print("Bing new patch!")
            # time.sleep(0.05)
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
            time.sleep(0.2)

    def close(self):
        self._stay_alive = False
        self.join()    

class ManipulatorThread(Thread):
    def __init__(self, camera_image_queue, patch_queue, project_patch, dataset, background_idx=0):
        super().__init__()
        self.camera_image_queue = camera_image_queue#deque(maxlen=1)
        # self.camera_image_queue.append(torch.ones((1, 96, 160), dtype=torch.float32) * 255.)
        self.dataset = dataset
        self.patch_queue = patch_queue
        self.project_patch = project_patch
        self._stay_alive = True

        self.background_idx = background_idx

    def run(self):
        # i = 0
        while self._stay_alive:
            # if self.camera_image_queue and self.patch_queue:
            if self.patch_queue:
                # print("Background image idx: ", self.background_idx)
                camera_image = self.dataset.dataset.__getitem__(self.background_idx)[0][0]#self.camera_image_queue[0]
                # camera_image = torch.ones((96, 160), dtype=torch.float32) * 255.
                patch, sf, scaled_tx, scaled_ty = self.patch_queue[0]

                if sf > 0.:
                    T = np.zeros((3, 3))
                    T[0, 0] = sf 
                    T[1, 1] = sf
                    T[0, 2] = scaled_tx
                    T[1, 2] = scaled_ty
                    T[2, 2] = 1
                    mod_img = self.project_patch(patch, T, camera_image)
                    # print(mod_img.shape, mod_img.min(), mod_img.max())

                    self.camera_image_queue.appendleft(mod_img)
                else:
                    self.camera_image_queue.appendleft(camera_image)
                
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
    def __init__(self, drone_pose, target_trajectory, camera_data, path):
        super().__init__()

        self.path = path
        self.drone_pose = drone_pose
        self.target_trajectory = target_trajectory
        self.current_target = deque(maxlen=1)
        self.index_reached = 0

        self.all_drone_poses = []
        self.all_target_poses = []

        self.projector_world = np.array([[2, 1, 2.2],   # ul
                                        [2, -1.5, 2.2], #ur
                                        [2, 1, 0.8], # ll
                                        [2, -1.5, 0.8]]) #lr
        

        self.camera_intrinsic = camera_data.camera_intrinsic
        self.camera_distortion = camera_data.distortion_coeffs
        self.camera_extrinsic = camera_data.camera_extrinsic
        
        self.patch_size = 80

        self.image_bb = (0, 0, 160, 96)  # camera image bounding box

        self._stay_alive = True

    def run(self):
        start_time = time.time()
        self.all_drone_poses.append((0., *self.drone_pose[0]))
        e = self.target_trajectory.eval(0.)
        target_position = np.array([e.pos[0], e.pos[1], e.pos[2], e.yaw])
        self.all_target_poses.append((0., *target_position))
        
        while self._stay_alive:
            # if self.index_reached < len(self.target_trajectory):
            #     target_position = self.target_trajectory[self.index_reached]

            #     # check if the drone pose is close to the target
            #     distance = np.linalg.norm(self.drone_pose[0][:2] - target_position[:2])
            #     # print("current distance: ", distance)
            #     if distance < 0.2:
            #         if self.index_reached < len(self.target_trajectory):
            #             self.index_reached += 1
            #             print("Target reached: ", target_position)
            #             target_position = self.target_trajectory[self.index_reached]
            #             print("Moving towards: ", target_position)
            #         else: 
            #             print("All targets reached")
            #             self._stay_alive = False
            #             break
                time_elapsed = np.clip(time.time() - start_time, 0., self.target_trajectory.duration)
                e = self.target_trajectory.eval(time_elapsed)
                target_position = np.array([e.pos[0], e.pos[1], e.pos[2], e.yaw])
                # to keep x constant -> target x should be 1.
                # to lower x -> target x should be > 1.
                # to increase x -> target x should be < 1.
                # to move left -> target y should be > 0.
                # to move right -> target y should be < 0.
                # to move up -> target z should be > 0.
                # to move down -> target z should be < 0.

                # print("Drone pose: ", self.drone_pose[0])
                # print("Checkpoint position world: ", target_position)

                quats_checkpoint = rowan.from_euler(0., 0., target_position[3], convention='xyz') # returns qw, qx, qy, qz
                rot_matrix_checkpoint = rowan.to_matrix(quats_checkpoint)
                T_checkpoint = np.eye(4)
                T_checkpoint[:3, :3] = rot_matrix_checkpoint
                T_checkpoint[:3, 3] = target_position[:3]

                T_direction = np.eye(4)
                T_direction[:3, 3] = [1. * np.cos(np.pi), 1. * np.sin(np.pi), 0.]   # TODO: Calculate the angle, such that the drone will be facing the center of the monitor

                
                T_prediction_world = np.linalg.inv(T_direction) @ T_checkpoint


               
                quats = rowan.from_euler(0., 0., self.drone_pose[0][3], convention='xyz') # returns qw, qx, qy, qz
                # TODO: yaw might need to change once we switch to having the monitor at fixed position

                rot_matrix = rowan.to_matrix(quats)
                T_drone_world = np.eye(4)
                T_drone_world[:3, :3] = rot_matrix
                T_drone_world[:3, 3] = self.drone_pose[0][:3]


                T_prediction_drone = np.linalg.inv(T_drone_world) @ T_prediction_world

                # print("Checkpoint position frontnet in drone frame: ", T_prediction_drone[:3, 3])
                target_yaw = rowan.to_euler(rowan.from_matrix(T_prediction_drone[:3, :3]), convention='xyz')[2]

                target_x, target_y, target_z = T_prediction_drone[:3, 3].tolist()

                # print("Target for Frontnet: ", target_x, target_y, target_z)


                # possible position decision:
                # if target_x closer 2.: sf should be closer to 0.4
                # if target_x closer 0.: sf should be closer to 0.8
                # if target_y closer 1.: tx should be closer to 0
                # if target_y closer -1.: tx should be closer to 1
                # if target_z closer 0.5: ty should be closer to 0
                # if target_z closer -0.5: ty should be closer to 1


                # sf = 0.8 - 0.4 * self.sigmoid(target_x, x0=1.0, k=5.0)
                # tx = self.sigmoid(target_y, x0=0.0, k=-5.0)
                # ty = 1 - self.sigmoid(target_z, x0=0.0, k=10.0)

                # TODO: change back to smoother/different tx decision
                # tx = np.random.uniform(0., 1.)#1. if target_y < 0 else 0.
                tx = 1. if target_y < 0 else 0.
                ty = 0.5

                # ty = np.random.uniform(0., 1.)

                # print("Position patch: ", sf, tx, ty)

                # sf = np.clip(sf, 0.4, 0.8)
                # tx = np.clip(tx, 0.0, 1.0)
                # ty = np.clip(ty, 0.0, 1.0)

                # print("Position patch: ", sf, tx, ty)

                possible_sf, patch_ul, visible_projector_dim = self.get_possible_scale_factor(T_drone_world)
                # if possible_sf is None:
                #     print("No possible transformation found, skipping...")
                #     time.sleep(0.1)
                #     continue

                #sf = np.random.uniform(0.4, possible_sf)
                sf = possible_sf


                # print("Scale factor: ", sf, "Patch upper left: ", patch_ul, "Visible projector dim: ", visible_projector_dim)
                # print("tx, ty: ", tx, ty)


                # print(possible_sf, tx, ty, *patch_ul, *visible_projector_dim, target_x, target_y, target_z)

                self.current_target.append(np.array([sf, tx, ty, *patch_ul, *visible_projector_dim, target_x, target_y, target_z]))
                
                
                
                self.all_drone_poses.append((time_elapsed, *self.drone_pose[0]))
                self.all_target_poses.append((time_elapsed, *target_position))
                time.sleep(0.1)

    def sigmoid(self, x, x0=0.0, k=10.0):
        """Sigmoid function centered at x0 with steepness k"""
        return 1 / (1 + np.exp(-k * (x - x0)))
    

    def get_possible_scale_factor(self, T_drone_world):
        projector_area_image = self.get_projector_area(T_drone_world)
        if projector_area_image[0] == 0.:
            print("No overlap in image coordinates, cannot project patch area to image.")
            return 0., (0., 0.), (0., 0.)
        

        projector_ul_x, projector_ul_y, projector_height, projector_width = projector_area_image
        max_dim = np.min((projector_height, projector_width))
        

        possible_sf = np.clip(max_dim / self.patch_size, 0.4, 0.8)  # scale factor should be between 0.4 and 0.8

        return possible_sf, (projector_ul_x, projector_ul_y), (projector_height, projector_width)

    def get_projector_area(self, T_drone_world):

        projector_drone = np.array([np.linalg.inv(T_drone_world) @ np.array([*projector_coords, 1.]) for projector_coords in self.projector_world])
        projector_camera = np.array([(self.camera_extrinsic @ patch_coords)[:3] for patch_coords in projector_drone])  # camera_extrinsic is T_drone_camera
        projector_image = np.array([(self.camera_intrinsic @ patch_coords) for patch_coords in projector_camera])         # camera_intrinsic is T_camera_image

        projector_image_ul, projector_image_ur, projector_image_ll, projector_image_lr = projector_image
        projector_image_ul = np.array([projector_image_ul[0] / projector_image_ul[2], projector_image_ul[1] / projector_image_ul[2]])
        projector_image_ur = np.array([projector_image_ur[0] / projector_image_ur[2], projector_image_ur[1] / projector_image_ur[2]])
        projector_image_ll = np.array([projector_image_ll[0] / projector_image_ll[2], projector_image_ll[1] / projector_image_ll[2]])
        projector_image_lr = np.array([projector_image_lr[0] / projector_image_lr[2], projector_image_lr[1] / projector_image_lr[2]])


        # check for overlap with image
        # x => width
        # y => height
        
        intersection_min_x = np.max((projector_image_ul[0], self.image_bb[0]))
        intersection_min_y = np.max((projector_image_ul[1], self.image_bb[1]))
        intersection_max_x = np.min((projector_image_lr[0], self.image_bb[2]))
        intersection_max_y = np.min((projector_image_ll[1], self.image_bb[3]))

        if intersection_min_x < intersection_max_x and intersection_min_y < intersection_max_y:
            height = intersection_max_y - intersection_min_y
            width = intersection_max_x - intersection_min_x
            return np.array((intersection_min_x, intersection_min_y, height, width))
        else:
            print("No overlap with image bounding box")
            return np.array((0., 0., 0., 0.))

    def close(self):
        self._stay_alive = False
        self.join()

        drone_poses = np.array(self.all_drone_poses)
        target_poses = np.array(self.all_target_poses)

        np.save(self.path / 'drone_poses.npy', drone_poses)
        np.save(self.path / 'target_poses.npy', target_poses)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(drone_poses[:, 0], drone_poses[:, 1], label='Drone')
        axs[0].plot(target_poses[:, 0], target_poses[:, 1], label='Target', linestyle='dotted')

        axs[1].plot(drone_poses[:, 0], drone_poses[:, 2], label='Drone')
        axs[1].plot(target_poses[:, 0], target_poses[:, 2], label='Target', linestyle='dotted')

        axs[2].plot(drone_poses[:, 0], drone_poses[:, 3], label='Drone')
        axs[2].plot(target_poses[:, 0], target_poses[:, 3], label='Target', linestyle='dotted')

        axs[2].set_xlabel('Time (s)')
        axs[0].set_ylabel('X Position (m)')
        axs[1].set_ylabel('Y Position (m)')
        axs[2].set_ylabel('Z Position (m)')

        axs[2].legend()
        fig.tight_layout()
        fig.savefig(self.path / 'drone_target_poses.png', dpi=300)
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate a drone attack using diffusion models.')
    parser.add_argument('--trajectory', type=str, default='change_y', choices=['change_y', 'change_x']) # TODO: include rectangle, figure8
    parser.add_argument('--mode', type=str, default='diffusion', choices=['diffusion', 'closest', 'interpolated']) # TODO: include gt, interpolation, random, black/white?
    parser.add_argument('--model', type=str, default='frontnet', choices=['frontnet', 'yolov5'])
    # parser.add_argument('--diffusion_model_path', type=str, default='results/diffusion_training/edm_frontnet_1k_25ds_2ke.pth')
    parser.add_argument('--n_images', type=int, default=1, help='Index of the image to use for the simulation.')
    parser.add_argument('--n_sim_runs', type=int, default=1, help='Number of simulation runs to perform.')
    
    # set a seed for reproducibility
    torch.manual_seed(424242)
    np.random.seed(424242)

    model_path = 'results/diffusion_training/edm_frontnet_1k_25ds_2ke.pth'

    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    model_type = 'frontnet'
    cf_sim = CFSim(model_type, dataset_path)

    args = parser.parse_args()
    n_sim_runs = args.n_sim_runs

    len_dataset = len(cf_sim.dataset.dataset)
    background_image_indices = np.random.choice(len_dataset, args.n_images, replace=False)

    print(f"Running {n_sim_runs} simulations for each of the {args.n_images} background images.")
    print(f"Background image indices: {background_image_indices}")

    for background_image_idx in background_image_indices:
        for i in range(n_sim_runs):

            save_directory = Path(f'results/{args.trajectory}/{args.mode}/{args.model}/image_{background_image_idx}/simulation_{i}/')

            os.makedirs(save_directory, exist_ok=True)

            # camera_thread = CameraThread(cf_sim.dataset)
            # camera_thread.start()

   
            target_trajectory = Trajectory()
            target_trajectory.loadcsv(f"uav_trajectories/traj_{args.trajectory}.csv")
            target_trajectory.stretchtime(10)

            from camera import Camera
            cam = Camera('camera_calibration.yaml')

            camera_image_queue = deque(maxlen=1)

            sim_thread = SimulatorThread(cf_sim.sim_new_pose, camera_image_queue, cam, path=save_directory, trajectory=args.trajectory)

            attacker_policy = AttackerPolicyThread(sim_thread.drone_pose, target_trajectory, camera_data=cam, path=save_directory)


            match args.mode:
                case 'diffusion':
                    patch_gen_thread = DiffusionThread(model_path, attacker_policy.current_target)
                case 'closest':
                    patch_gen_thread = ClosestPatchThread("frontnet1k.pickle", attacker_policy.current_target)
                case 'interpolated':
                    patch_gen_thread = InterpolatedPatchThread("frontnet1k.pickle", attacker_policy.current_target)

            manipulator_thread = ManipulatorThread(camera_image_queue, patch_gen_thread.patch_queue, cf_sim.project_patch, cf_sim.dataset, background_idx=background_image_idx)




            # sim_thread = SimulatorThread(cf_sim.sim_new_pose, camera_thread.camera_image_queue, cam.point_from_xyz)
            # sim_thread.start()

            attacker_policy.start()


            patch_gen_thread.start()

            while not patch_gen_thread.patch_queue:
                time.sleep(0.1)

            # camera_thread.start()
            manipulator_thread.start()
            time.sleep(0.01)
            sim_thread.start()

            time_start = time.time()

            while time.time() - time_start < target_trajectory.duration:
                # wait for 20 seconds
                time.sleep(0.1)
                if not attacker_policy._stay_alive:
                    print("Attacker policy thread finished")
                    break

            print("Stopping threads...")
            sim_thread.close()
            # camera_thread.close()
            manipulator_thread.close()
            patch_gen_thread.close()
            attacker_policy.close()

            # print(sim_thread.all_poses)

            all_poses = np.array(sim_thread.all_poses)
            # print(all_poses.shape)

            # plot 3d positions stored in all_poses (consists of (timestamp, pose))  with matplotlib
            # from mpl_toolkits.mplot3d import Axes3D
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.scatter(all_poses[:, 1], all_poses[:, 2], all_poses[:, 3])
            # ax.plot(all_poses[:, 1], all_poses[:, 2], all_poses[:, 3])
            # ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], color='red', linestyle='dotted') 
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # fig.savefig('3d_positions.png')


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