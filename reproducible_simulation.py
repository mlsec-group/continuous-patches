import os
import time
import pickle
import argparse

from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import cv2
import torch
import rowan

import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from yolo_bounding import YOLOBox
from diffusion.diffusion_model import DiffusionModel
from util import scale_tx_ty, load_model, load_dataset

import sys
sys.path.insert(0,'uav_trajectories/scripts')
from uav_trajectory import Trajectory


EPSILON = 1e-12


class VirtualThread(ABC):
    name: str
    @abstractmethod
    def step(environment, t: int) -> int:
        pass


class TimedState:
    def __init__(self, init):
        self.states = [(0, init)]
    
    def set(self, state, t):
        self.states.append((t, state))
        self.states.sort(key=lambda tupl: tupl[0])
    
    def get(self, t):
        if len(self.states) == 0:
            return None
        i = 0
        while (i + 1 < len(self.states)) and t - EPSILON >= self.states[i + 1][0]:
            i += 1
        return self.states[i][1]


@dataclass
class Patch:
    sf: float = None
    tx: float = None
    ty: float = None
    image: np.array = None


class Environment:
    def __init__(self, background_image):
        self.drone_pose = TimedState(np.array([0., 0., 1., 0.]))
        self.drone_velocity = TimedState(np.array([0., 0., 0.]))
        self.drone_set_point = TimedState(np.zeros(4))
        self.patch = TimedState(Patch())
        self.background_image = background_image


class Attacker(VirtualThread):
    name = "Attacker"

    def __init__(self, trajectory_waypoints, camera_data):
        super().__init__()
        self.trajectory_waypoints = trajectory_waypoints

        self.target_trajectory = target_trajectory
        self.index_reached = 0

        self.projector_world = np.array([[2, 1, 2.2],   # ul
                                        [2, -1.5, 2.2], #ur
                                        [2, 1, 0.8], # ll
                                        [2, -1.5, 0.8]]) #lr
        

        self.camera_intrinsic = camera_data.camera_intrinsic
        self.camera_distortion = camera_data.distortion_coeffs
        self.camera_extrinsic = camera_data.camera_extrinsic
        
        self.patch_size = 80
        self.waypoint_threshold = 0.2 # 20cm maximum distance to waypoint
        self.image_bb = (0, 0, 160, 96)  # camera image bounding box
    
    def step(self, environment: Environment, t: int) -> int:
        start_time = time.time()
        drone_pose = environment.drone_pose.get(t)
        target_pose = self.target_trajectory[self.index_reached]
        if self.index_reached + 1 < len(self.target_trajectory):
            distance = np.linalg.norm(drone_pose[:2] - target_pose[:2])
            if distance < self.waypoint_threshold:
                self.index_reached += 1
                if self.index_reached + 1 < len(self.target_trajectory):
                    print("Target reached: ", target_pose)
                else: 
                    print("All targets reached")
        target_pose = self.target_trajectory[self.index_reached]

        quats_checkpoint = rowan.from_euler(0., 0., target_pose[3], convention='xyz') # returns qw, qx, qy, qz
        rot_matrix_checkpoint = rowan.to_matrix(quats_checkpoint)
        T_checkpoint = np.eye(4)
        T_checkpoint[:3, :3] = rot_matrix_checkpoint
        T_checkpoint[:3, 3] = target_pose[:3]

        T_direction = np.eye(4)
        # TODO: Calculate the angle, such that the drone will be facing the center of the monitor
        T_direction[:3, 3] = [1. * np.cos(np.pi), 1. * np.sin(np.pi), 0.]
        T_prediction_world = np.linalg.inv(T_direction) @ T_checkpoint

        quats = rowan.from_euler(0., 0., drone_pose[3], convention='xyz') # returns qw, qx, qy, qz
        rot_matrix = rowan.to_matrix(quats)
        T_drone_world = np.eye(4)
        T_drone_world[:3, :3] = rot_matrix
        T_drone_world[:3, 3] = drone_pose[:3]

        T_prediction_drone = np.linalg.inv(T_drone_world) @ T_prediction_world

        target_x, target_y, target_z = T_prediction_drone[:3, 3].tolist()
        tx = 1. if target_y < 0 else 0.
        ty = 0.5
        possible_sf, patch_ul, visible_projector_dim = self.get_possible_scale_factor(T_drone_world)

        current_target = np.array([possible_sf, tx, ty, *patch_ul, *visible_projector_dim, target_x, target_y, target_z])
        time_elapsed = time.time() - start_time
        if possible_sf > 0.:
            patch = self._generate_patch(current_target)
            time_elapsed = time.time() - start_time
            environment.patch.set(patch, t + time_elapsed)
        # TODO: remove
        time_elapsed = max(time_elapsed, 0.05)
        return t + time_elapsed

    def get_possible_scale_factor(self, T_drone_world):
        projector_area_image = self.get_projector_area(T_drone_world)
        if projector_area_image[2] <= 0. or projector_area_image[3] <= 0.:
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

        intersection_ul = (max(projector_image_ul[0], self.image_bb[0]), 
                           max(projector_image_ul[1], self.image_bb[1]))
        intersection_ur = (min(projector_image_ur[0], self.image_bb[2]),
                           max(projector_image_ur[1], self.image_bb[1]))
        intersection_ll = (max(projector_image_ll[0], self.image_bb[0]),
                           min(projector_image_ll[1], self.image_bb[3]))
        intersection_lr = (min(projector_image_lr[0], self.image_bb[2]),
                           min(projector_image_lr[1], self.image_bb[3]))

        height = min(intersection_ll[1], intersection_lr[1]) - max(intersection_ur[1], intersection_ul[1])
        width = min(intersection_ur[0], intersection_lr[0]) - max(intersection_ul[0], intersection_ll[0])

        return np.array((intersection_ul[0], intersection_ul[1], height, width))

    @abstractmethod
    def _generate_patch(self, target: np.array):
        pass


class ClosestPatchAttacker(Attacker):
    def __init__(self, patches_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(patches_path, "rb") as f:
            patch_dataset = pickle.load(f)

        self.patches = np.array([patch for patch, target, transformation in patch_dataset], dtype=np.float32)
        targets = torch.as_tensor([target.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        transformations = torch.as_tensor([transformation.tolist() for patch, target, transformation in patch_dataset]).squeeze().float()
        self.combineds = torch.hstack([targets, transformations])

    @staticmethod
    def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))

    def _generate_patch(self, target):
        sf, tx, ty, patch_ul_x, patch_ul_y, projector_height, projector_width, x, y, z  = target
        combined = torch.tensor(np.stack(([x], [y], [z], [sf], [tx], [ty])).T, dtype=torch.float32)
        distances = ClosestPatchAttacker.dist(combined, self.combineds)
        closest = torch.argmin(distances)
        patch = self.patches[closest] * 255.

        scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80, (projector_height, projector_width))

        tx_img = patch_ul_x + scaled_tx
        ty_img = patch_ul_y + scaled_ty

        return Patch(sf=sf, tx=tx_img, ty=ty_img, image=patch)


class DiffusionAttacker(Attacker):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = DiffusionModel(self.device)
        self.diffusion_model.load(model_path)

    @staticmethod
    def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))

    def _generate_patch(self, target):
        sf, tx, ty, patch_ul_x, patch_ul_y, projector_height, projector_width, x, y, z  = target
        condiditions = torch.tensor(np.stack(([sf], [tx], [ty], [x], [y], [z])).T, dtype=torch.float32)

        samples = self.diffusion_model.sample(1, condiditions, self.device, patch_size=[80,80], n_steps=100).detach().to('cpu').numpy()
        patch = samples[0, 0] * 255.

        scaled_tx, scaled_ty = scale_tx_ty(sf, tx, ty, 80, (projector_height, projector_width))
        tx_img = patch_ul_x + scaled_tx
        ty_img = patch_ul_y + scaled_ty

        return Patch(sf=sf, tx=tx_img, ty=ty_img, image=patch)


class PoseTracker(VirtualThread):
    name = "PoseTracker"

    def __init__(self, model="frontnet"):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        if self.model == "frontnet":
            self.pose_estimator = load_model(path='pulp-frontnet/PyTorch/Models/Frontnet160x32.pt', device=device, config='160x32')
            self.pose_estimator.eval()
        elif self.model == "yolov5":
            self.pose_estimator = YOLOBox().to(device).eval()
        else:
            raise ValueError("Model type not supported!")

        self.dt = 0.1

    def step(self, environment: Environment, t: int) -> int:
        background_image = environment.background_image
        patch = environment.patch.get(t)
        camera_image = background_image.clone()
        if patch.image is not None and patch.sf > 0. and patch.sf <= 0.8:
            T = np.zeros((3, 3))
            T[0, 0] = patch.sf
            T[1, 1] = patch.sf
            T[0, 2] = patch.tx
            T[1, 2] = patch.ty
            T[2, 2] = 1
            camera_image = self.project_patch(patch.image, T, camera_image)
        
        device = next(self.pose_estimator.parameters()).device
        image_t = torch.as_tensor(camera_image).to(device)
        if len(image_t.shape) < 3:
            image_t = image_t.unsqueeze(0).unsqueeze(0)
        if len(image_t.shape) == 3:
            image_t = image_t.unsqueeze(1)
        if self.model == 'frontnet':
            x, y, z, yaw = self.pose_estimator(image_t)
            predicted_pose = torch.hstack((x, y, z, yaw)).squeeze()
        elif self.model == 'yolov5':
            predicted_pose = self.pose_estimator(image_t)
            predicted_pose = torch.hstack((
                *predicted_pose,
                torch.zeros(1, device=predicted_pose[0].device) # add a dummy yaw value
            ))

        predicted_pose = predicted_pose.detach().cpu().numpy()
        drone_pose = environment.drone_pose.get(t)
        new_setpoint = self._controller_setpoint(predicted_pose, drone_pose)

        environment.drone_set_point.set(new_setpoint, t + self.dt)

        return t + self.dt
    
    def _controller_setpoint(self, predicted_pose, drone_pose):
        # we manually set the yaw now
        # - frontnet predictions were always faulty
        # - yolo doesn't output an angle (yet)
        predicted_pose[3] = -np.pi

        T_pred_drone = np.eye(4)
        T_pred_drone[:3, :3] = rowan.to_matrix(rowan.from_euler(0., 0., predicted_pose[3], convention='xyz'))
        T_pred_drone[:3, 3] = predicted_pose[:3]

        T_drone_world = np.eye(4)
        T_drone_world[:3, :3] = rowan.to_matrix(rowan.from_euler(0., 0., drone_pose[3], convention='xyz'))
        T_drone_world[:3, 3] = drone_pose[:3]

        T_pred_world = T_drone_world @ T_pred_drone

        T_direction_world = np.eye(4)
        T_direction_world[:3, 3] = self._calc_heading_vec(1., predicted_pose[3])

        T_setpoint_world = T_direction_world @ T_pred_world
        setpoint_yaw = 0.#rowan.to_euler(rowan.from_matrix(T_setpoint_world[:3, :3]), convention='xyz')[2]  # TODO!
        setpoint_x = np.clip(T_setpoint_world[0, 3], -2.5, 2.0)  # limit x to [-2.5, 2.5]
        setpoint_y = np.clip(T_setpoint_world[1, 3], -2.5, 2.5)  # limit y to [-1.5, 1.5]
        setpoint_z = 1.0  # constant height

        return np.array([setpoint_x, setpoint_y, setpoint_z, setpoint_yaw])
    
    def _calc_heading_vec(self, radius, angle):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y, 0.0])

    def project_patch(self, patch, T, image):
        # using cv2 to project the patch instead of FAP place_patch() function,
        # since we don't need to calculate gradients
        width, height = image.shape[:2]
        mask = np.ones_like(patch)

        warped_patch = cv2.warpPerspective(patch, T, (height, width), flags=cv2.INTER_NEAREST)
        mask = cv2.warpPerspective(mask, T, (height, width), flags=cv2.INTER_NEAREST)

        mod_img = image * ~mask.astype(bool)
        mod_img += warped_patch

        return mod_img
    
class Crazyflie(VirtualThread):
    name = "Drone"
    def __init__(self):
        super().__init__()
        self.dt = 0.01
        self.max_a = 1.
        self.max_v = 1.

    def step(self, environment: Environment, t: int) -> int:
        current_setpoint = environment.drone_set_point.get(t)
        current_pose = environment.drone_pose.get(t)
        current_position = current_pose[:3]
        current_velocity = environment.drone_velocity.get(t)

        target_velocity = (current_setpoint[:3] - current_position)
        dv = target_velocity - current_velocity
        dv_magnitude = np.linalg.norm(dv)
        if dv_magnitude > EPSILON:
            dv = dv / dv_magnitude * min(self.max_a, dv_magnitude)
        new_velocity = current_velocity + self.dt * dv
        if np.linalg.norm(new_velocity) > self.max_v:
            new_velocity = new_velocity / np.linalg.norm(new_velocity) * self.max_v
        environment.drone_velocity.set(new_velocity, t)

        new_position = current_position + (current_velocity + new_velocity) / 2 * self.dt
        new_pose = np.array([*new_position, current_pose[3]]) # fixed yaw
        environment.drone_pose.set(new_pose, t + self.dt)

        return t + self.dt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate a drone attack using diffusion models.')
    parser.add_argument('--trajectory', type=str, default='figure8', choices=['change_y', 'change_x', 'figure8']) # TODO: include rectangle, figure8
    parser.add_argument('--mode', type=str, default='closest', choices=['diffusion', 'closest', 'interpolated', 'none'])
    parser.add_argument('--model', type=str, default='frontnet', choices=['frontnet', 'yolov5'])
    parser.add_argument('--diffusion_model_path', type=str, default='results/diffusion_training.bak/trained_model.pth')
    parser.add_argument('--n_images', type=int, default=1, help='Index of the image to use for the simulation.')
    parser.add_argument('--n_sim_runs', type=int, default=1, help='Number of simulation runs to perform.')
    
    # set a seed for reproducibility
    torch.manual_seed(424242)
    np.random.seed(424242)

    args = parser.parse_args()
    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    model_type = args.model
    n_sim_runs = args.n_sim_runs
    camera = Camera('camera_calibration.yaml')

    background_images = load_dataset(dataset_path)
    len_dataset = len(background_images.dataset)
    background_image_indices = np.random.choice(len_dataset, args.n_images, replace=False)

    print(f"Running {n_sim_runs} simulations for each of the {args.n_images} background images.")
    print(f"Background image indices: {background_image_indices}")

    for background_image_idx in background_image_indices:
        for i in range(n_sim_runs):
            save_directory = Path(f'results/{args.trajectory}/{args.mode}/{args.model}/image_{background_image_idx}/simulation_{i}/')
            os.makedirs(save_directory, exist_ok=True)

            background_image = background_images.dataset.__getitem__(background_image_idx)[0][0]
            environment = Environment(background_image)

            t = np.linspace(0, 2 * np.pi, 30)
            x = 0.5 * np.sin(2 * t)  # Horizontal of figure 8
            y = 0.7 * np.sin(t)  # Vertical of figure 8
            z = np.ones_like(t)  # Constant height at 1
            yaw = np.zeros_like(t)  # Constant yaw
            target_trajectory = np.column_stack((x, y, z, yaw))

            attacker = None
            if args.mode == "closest":
                attacker = ClosestPatchAttacker(f"/shares/datasets/continuous_patches/{args.model}1k.pickle", target_trajectory, camera)
            elif args.mode == "diffusion":
                attacker = DiffusionAttacker(args.diffusion_model_path, target_trajectory, camera)

            threads = {
                Crazyflie.name: (0, Crazyflie()),
                PoseTracker.name: (0, PoseTracker(model_type)),
                "Attacker": (0, attacker)
            }
            
            t = -1
            all_poses = []
            last_image_t = 0
            while t < 1000:
                next_t = min(t for t, _ in threads.values())
                all_poses.append(environment.drone_pose.get(t))
                assert next_t > t
                threads_to_run = [name for (name, (t, _)) in threads.items() if t == next_t]
                for thread_name in threads_to_run:
                    _, thread = threads[thread_name]
                    new_thread_t = thread.step(environment, next_t)
                    assert new_thread_t > next_t
                    threads[thread_name] = (new_thread_t, thread)
                t = next_t
                # print(f"(t: {next_t}) ran {threads_to_run}")
                
                if (t - last_image_t) > 1.:
                    print(t)
                    last_image_t = t
                    fig = plt.figure(figsize=(10, 5))

                    # Left subplot: Image with scattered point
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(background_image, cmap='gray')
                    
                    # only plot the dots for the projector if they are within image space:
                    # if projector_image_ul[0] > 0. and  projector_image_ul[1] > 0. and \
                    #    projector_image_lr[0] <= 160. and projector_image_lr[1] <= 96.:
                    #     ax1.scatter(projector_image_ul[0], projector_image_ul[1], color='blue', label='Projector corners')
                    #     ax1.scatter(projector_image_ur[0], projector_image_ur[1], color='blue')
                    #     ax1.scatter(projector_image_ll[0], projector_image_ll[1], color='blue')
                    #     ax1.scatter(projector_image_lr[0], projector_image_lr[1], color='blue')

                    # Right subplot: Drone position in 2d
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.set_title("Drone Position in 2D")
                    current_setpoint = environment.drone_set_point.get(t)
                    all_poses_a = np.asarray(all_poses)
                    ax2.plot(all_poses_a[:, 0], all_poses_a[:, 1], label="Drone Path")
                    ax2.plot(target_trajectory[:, 0], target_trajectory[:, 1], label="Target Trajectory", color='green')
                    ax2.scatter(current_setpoint[0], current_setpoint[1], color='red', label="Current Setpoint")
                    ax2.set_xlabel("X")
                    ax2.set_ylabel("Y")
                    # ax2.set_zlabel("Z")

                    # add current drone position as scatter with arrow for yaw
                    current_pose = environment.drone_pose.get(t)
                    ax2.scatter(current_pose[0], current_pose[1], color='black')
                    ax2.arrow(current_pose[0], current_pose[1],
                              0.3 * np.cos(current_pose[3]), 0.3 * np.sin(current_pose[3]),
                              head_width=0.1, head_length=0.1, fc='black', ec='black')


                    ax2.scatter(2., 1., color='blue', label='Projector corners')
                    ax2.scatter(2., -1.5, color='blue')

                    # set ax2 limits
                    ax2.set_xlim([-2.5, 2.5])
                    ax2.set_ylim([-1.5, 1.5])
                    # ax2.set_zlim([0, 2.5])
                    ax2.legend()

                    plt.tight_layout()
                    plt.savefig(save_directory / f"patched_image_{t}.png", dpi=300)
                    plt.close()