import torch
import numpy as np
from diffusion.diffusion_model import DiffusionModel

from cf_simulator import CFSim, SimulatorThread

import time

from util import scale_tx_ty, bb2camera, line_plane_intersection

import cv2
import rowan

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
            

            r_targets = torch.tensor(np.stack(([sf], [tx], [ty], [x], [y], [z])).T, dtype=torch.float32)
            # print(r_targets.shape)

            samples = self.diffusion_model.sample(1, r_targets, self.device, patch_size=[80,80], n_steps=25).detach().to('cpu').numpy()
            patch = samples[0, 0] * 255.

            scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80, (projector_height, projector_width))

            tx_img = patch_ul_x + scaled_tx
            ty_img = patch_ul_y + scaled_ty

            self.patch_queue.append((patch, sf, tx_img, ty_img))
            # print("Bing new patch!")

           
            time.sleep(0.05)

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
    def __init__(self, camera_image_queue, patch_queue, project_patch, dataset):
        super().__init__()
        self.camera_image_queue = camera_image_queue#deque(maxlen=1)
        # self.camera_image_queue.append(torch.ones((1, 96, 160), dtype=torch.float32) * 255.)
        self.dataset = dataset
        self.patch_queue = patch_queue
        self.project_patch = project_patch
        self._stay_alive = True

    def run(self):
        # i = 0
        while self._stay_alive:
            # if self.camera_image_queue and self.patch_queue:
            if self.patch_queue:
                camera_image = self.dataset.dataset.__getitem__(0)[0][0]#self.camera_image_queue[0]
                # camera_image = torch.ones((96, 160), dtype=torch.float32) * 255.
                patch, sf, scaled_tx, scaled_ty = self.patch_queue[0]

                T = np.zeros((3, 3))
                T[0, 0] = sf 
                T[1, 1] = sf
                T[0, 2] = scaled_tx
                T[1, 2] = scaled_ty
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
    def __init__(self, drone_pose, target_trajectory, camera_data):
        super().__init__()
        self.drone_pose = drone_pose
        self.target_trajectory = target_trajectory
        self.current_target = deque(maxlen=1)
        self.index_reached = 0


        self.projector_world = np.array([[2, 1, 2.2],   # ul
                                        [2, -1.5, 2.2], #ur
                                        [2, 1, 0.8], # ll
                                        [2, -1.5, 0.8]]) #lr
        

        self.camera_intrinsic = camera_data.camera_intrinsic
        self.camera_distortion = camera_data.distortion_coeffs
        self.camera_extrinsic = camera_data.camera_extrinsic
        
        self.patch_size = 80

        self._stay_alive = True

    def run(self):
        while self._stay_alive:
            if self.index_reached < len(self.target_trajectory):
                target_position = self.target_trajectory[self.index_reached]

                # check if the drone pose is close to the target
                distance = np.linalg.norm(self.drone_pose[0][:2] - target_position[:2])
                # print("current distance: ", distance)
                if distance < 0.2:
                    if self.index_reached < len(self.target_trajectory):
                        self.index_reached += 1
                        print("Target reached: ", target_position)
                        target_position = self.target_trajectory[self.index_reached]
                        print("Moving towards: ", target_position)
                    else: 
                        print("All targets reached")
                        self._stay_alive = False
                        break
                
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


                # possible position decision:
                # if target_x closer 2.: sf should be closer to 0.4
                # if target_x closer 0.: sf should be closer to 0.8
                # if target_y closer 1.: tx should be closer to 0
                # if target_y closer -1.: tx should be closer to 1
                # if target_z closer 0.5: ty should be closer to 0
                # if target_z closer -0.5: ty should be closer to 1


                # sf = 0.8 - 0.4 * self.sigmoid(target_x, x0=1.0, k=5.0)
                tx = self.sigmoid(target_y, x0=0.0, k=-5.0)
                ty = 1 - self.sigmoid(target_z, x0=0.0, k=10.0)

                # print("Position patch: ", sf, tx, ty)

                # sf = np.clip(sf, 0.4, 0.8)
                tx = np.clip(tx, 0.0, 1.0)
                ty = np.clip(ty, 0.0, 1.0)

                # print("Position patch: ", sf, tx, ty)

                possible_sf, patch_ul, visible_projector_dim = self.get_possible_scale_factor(T_drone_world)
                if possible_sf is None:
                    print("No possible transformation found, skipping...")
                    time.sleep(0.1)
                    continue

                # print(possible_sf, tx, ty, *patch_ul, *visible_projector_dim, target_x, target_y, target_z)

                self.current_target.append(np.array([possible_sf, tx, ty, *patch_ul, *visible_projector_dim, target_x, target_y, target_z]))
                time.sleep(0.1)

    def sigmoid(self, x, x0=0.0, k=10.0):
        """Sigmoid function centered at x0 with steepness k"""
        return 1 / (1 + np.exp(-k * (x - x0)))
    

    def get_possible_scale_factor(self, T_drone_world):
        patch_area_image = self.get_possible_patch_area_image(T_drone_world)
        if patch_area_image is None:
            print("No overlap in image coordinates, cannot project patch area to image.")
            return None, None, (None, None)
        
        
        patch_image_ul, patch_image_ur, patch_image_ll, patch_image_lr = patch_area_image
        max_width = np.abs(patch_image_ur[0] - patch_image_ul[0])
        max_height = np.abs(patch_image_ll[1] - patch_image_ul[1])
        max_dim = np.min((max_width, max_height))

        possible_sf = np.clip(max_dim / self.patch_size, 0.4, 0.8)  # scale factor should be between 0.4 and 0.8
        #scaled_tx, scaled_ty = scale_tx_ty(possible_sf, desired_tx, desired_ty, self.patch_size, (max_height, max_width))

        # T_patch = np.zeros((3, 3))
        # T_patch[0, 0] = possible_sf
        # T_patch[1, 1] = possible_sf
        # T_patch[0, 2] = scaled_tx + patch_image_ul[0]
        # T_patch[1, 2] = scaled_ty + patch_image_ul[1]
        # T_patch[2, 2] = 1


        return possible_sf, patch_image_ul, (max_height, max_width)
    
    def get_possible_patch_area_image(self, T_drone_world):
        patch_area_world = self.get_possible_patch_area_world(T_drone_world)
        if patch_area_world is None:
            print("No overlap in world coordinates, cannot project patch area to image.")
            return None
        
        patch_area_drone =  np.array([(np.linalg.inv(T_drone_world) @ np.array([*patch_coords, 1.])) for patch_coords in patch_area_world]) 
        # element-wise matrix multiplication to get each point in drone frame (in homogeneous coords)

        patch_area_camera = np.array([(self.camera_extrinsic @ patch_coords)[:3] for patch_coords in patch_area_drone])  # camera_extrinsic is T_drone_camera

        patch_area_image = np.array([(self.camera_intrinsic @ patch_coords) for patch_coords in patch_area_camera])      # camera_intrinsic is T_camera_image

        patch_image_ul, patch_image_ur, patch_image_ll, patch_image_lr = patch_area_image

        patch_image_ul = np.array([patch_image_ul[0] / patch_image_ul[2], patch_image_ul[1] / patch_image_ul[2]])
        patch_image_ur = np.array([patch_image_ur[0] / patch_image_ur[2], patch_image_ur[1] / patch_image_ur[2]])
        patch_image_ll = np.array([patch_image_ll[0] / patch_image_ll[2], patch_image_ll[1] / patch_image_ll[2]])
        patch_image_lr = np.array([patch_image_lr[0] / patch_image_lr[2], patch_image_lr[1] / patch_image_lr[2]])

        return np.array([patch_image_ul, patch_image_ur, patch_image_ll, patch_image_lr])


    def get_possible_patch_area_world(self, T_drone_world):
        # projecting the corners of the camera image to world coordinates
        camera_image_bounding_box = (0, 0, 160, 96)
        camera_center, ul_camera, ur_camera, ll_camera, lr_camera, _ = bb2camera(camera_image_bounding_box, self.camera_intrinsic, self.camera_distortion)

        camera_center_drone = (np.linalg.inv(self.camera_extrinsic) @ np.array([*camera_center, 1.]))[:3]
        ul_drone = (np.linalg.inv(self.camera_extrinsic) @ np.array([*ul_camera, 1.]))[:3]
        ur_drone = (np.linalg.inv(self.camera_extrinsic) @ np.array([*ur_camera, 1.]))[:3]
        ll_drone = (np.linalg.inv(self.camera_extrinsic) @ np.array([*ll_camera, 1.]))[:3]
        lr_drone = (np.linalg.inv(self.camera_extrinsic) @ np.array([*lr_camera, 1.]))[:3]

        camera_center_world = (T_drone_world @ np.array([*camera_center_drone, 1.]))[:3]
        ul_world = (T_drone_world @ np.array([*ul_drone, 1.]))[:3]
        ur_world = (T_drone_world @ np.array([*ur_drone, 1.]))[:3]
        ll_world = (T_drone_world @ np.array([*ll_drone, 1.]))[:3]
        lr_world = (T_drone_world @ np.array([*lr_drone, 1.]))[:3]

        plane_normal = np.cross(self.projector_world[1] - self.projector_world[0], self.projector_world[2] - self.projector_world[0])
        plane_point = self.projector_world[0]

        intersection_ul = line_plane_intersection(plane_normal, plane_point, ul_world - camera_center_world, camera_center_world)
        intersection_ur = line_plane_intersection(plane_normal, plane_point, ur_world - camera_center_world, camera_center_world)
        intersection_ll = line_plane_intersection(plane_normal, plane_point, ll_world - camera_center_world, camera_center_world)
        intersection_lr = line_plane_intersection(plane_normal, plane_point, lr_world - camera_center_world, camera_center_world)

        # find overlap with projector area in world space

        max_y = np.min((intersection_ul[1], intersection_ll[1]))
        min_y = np.max((intersection_ur[1], intersection_lr[1]))

        max_z = np.min((intersection_ul[2], intersection_ur[2]))
        min_z = np.max((intersection_ll[2], intersection_lr[2]))


        if min_y <= self.projector_world[0, 1] and max_y >= self.projector_world[1, 1] and min_z <= self.projector_world[1, 2] and max_z >= self.projector_world[3, 2]:
            patch_min_y = np.max((min_y, self.projector_world[0, 1]))
            patch_max_y = np.min((max_y, self.projector_world[1, 1]))

            patch_min_z = np.max((min_z, self.projector_world[3, 2]))
            patch_max_z = np.min((max_z, self.projector_world[1, 2]))


            patch_space_world = np.array([[2., patch_min_y, patch_max_z],
                                        [2., patch_max_y, patch_max_z],
                                        [2., patch_min_y, patch_min_z],
                                        [2., patch_max_y, patch_min_z]])
            
            # print(patch_space_world)

            return patch_space_world
        else:
            print("no overlap")
            return None

    def close(self):
        self._stay_alive = False
        self.join()

if __name__ == "__main__":

    # set a seed for reproducibility
    torch.manual_seed(4242)
    np.random.seed(4242)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # diffusion_model = DiffusionModel(device, lr=1e-5)
    # diffusion_model.load('results/diffusion_training/trained_model.pth')

    os.makedirs('results/simulation', exist_ok=True)

    model_path = 'results/diffusion_training/edm_frontnet_1k_25ds_2ke.pth'


    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    model_type = 'frontnet'
    cf_sim = CFSim(model_type, dataset_path)

    # camera_thread = CameraThread(cf_sim.dataset)
    # camera_thread.start()

    target_trajectory = np.array([#[0.0, 0.25, 1., 0.0],
                            [0.0, 0.50, 1., 0.0],
                            #[0.0, 0.75, 1., 0.0],
                            [0.0, 1.00, 1., 0.0],
                            #[0.0, 0.75, 1., 0.0],
                            [0.0, 0.50, 1., 0.0],
                            #[0.0, 0.25, 1., 0.0],
                            [0.0, 0.00, 1., 0.0],
                            #[0.0, -0.25, 1., 0.0],
                            [0.0, -0.50, 1., 0.0],
                            #[0.0, -0.75, 1., 0.0],
                            [0.0, -1.00, 1., 0.0],
                            #[0.0, -0.75, 1., 0.0],
                            [0.0, -0.50, 1., 0.0],
                            #[0.0, -0.25, 1., 0.0],
                            [0.0, 0.00, 1., 0.0]])

    # t = np.linspace(0, 2 * np.pi, 20)
    # x = 0.5 * np.sin(2 * t)  # Horizontal figure 8
    # y = 1.5 * np.sin(t)  # Vertical figure 8
    # z = np.ones_like(t)  # Constant height at 1
    # yaw = np.zeros_like(t)  # Constant yaw
    # target_trajectory = np.column_stack((x, y, z, yaw))

    from camera import Camera
    cam = Camera('camera_calibration.yaml')

    camera_image_queue = deque(maxlen=1)

    sim_thread = SimulatorThread(cf_sim.sim_new_pose, camera_image_queue, cam.point_from_xyz)

    attacker_policy = AttackerPolicyThread(sim_thread.drone_pose, target_trajectory, camera_data=cam)

    diffusion_thread = DiffusionThread(model_path, attacker_policy.current_target)


    manipulator_thread = ManipulatorThread(camera_image_queue, diffusion_thread.patch_queue, cf_sim.project_patch, cf_sim.dataset)




    # sim_thread = SimulatorThread(cf_sim.sim_new_pose, camera_thread.camera_image_queue, cam.point_from_xyz)
    # sim_thread.start()

    attacker_policy.start()


    diffusion_thread.start()

    while not diffusion_thread.patch_queue:
        time.sleep(0.1)

    # camera_thread.start()
    manipulator_thread.start()
    time.sleep(0.01)
    sim_thread.start()

    time_start = time.time()

    while time.time() - time_start < 120.:
        # wait for 20 seconds
        time.sleep(0.1)
        if not attacker_policy._stay_alive:
            print("Attacker policy thread finished")
            break

    print("Stopping threads...")
    sim_thread.close()
    # camera_thread.close()
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