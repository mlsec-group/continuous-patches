import torch
import os, sys
import cv2
import numpy as np


# sys.path.append('simulators/pulp-frontnet/PyTorch/Frontnet')

# from Frontnet import FrontnetModel
# from DataProcessor import DataProcessor
# from Dataset import Dataset
from torch.utils import data

import rowan

from pathlib import Path

from yolo_bounding import YOLOBox
from util import load_model, load_dataset

from threading import Thread
from collections import deque


import time

import matplotlib.pyplot as plt


class CFSim():
    def __init__(self, model='frontnet', dataset_path="pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.load_frontnet_model = load_model
        self.load_dataset = load_dataset
        
        if self.model == 'frontnet':
            self.pose_estimator = self.load_frontnet_model(path='pulp-frontnet/PyTorch/Models/Frontnet160x32.pt', device=self.device, config='160x32')
            self.pose_estimator.eval()
        elif self.model == 'yolov5':
            self.pose_estimator = self.load_yolo_model()
        else:
            raise ValueError("Model type not supported!")
        
        self.pose = np.array([0., 0., 0., 0.]) # x, y, z, yaw
        
        self.current_idx = 0
        # self.target_trajectory = target_trajectory

        # might be deleted later, the dataset is only loaded to get a suitable background image
        self.dataset = self.load_dataset(dataset_path, train=False, shuffle=False) # we need to test on the test set
        base_img, gt = self.dataset.dataset.__getitem__(0)
        self.base_img = base_img#.squeeze(0).numpy()

        # patch stays random for now and inside the simulator for compatibility with current
        # optimize script
        self.patch = np.random.rand(10, 10, 1).astype(np.float32) * 255. # load one of the optimized FAPs instead!
    
    def load_yolo_model(self):
        model = YOLOBox()
        return model
        
    def _perspective_grid(self,
    coeffs: [float], 
    w: int, h: int, 
    ow: int, oh: int, 
    dtype: torch.dtype, 
    device: torch.device,
    center = None,
    ) -> torch.Tensor:
        # source: https://github.com/pytorch/pytorch/issues/100526#issuecomment-1610226058
        # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
        # src/libImaging/Geometry.c#L394

        #
        # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
        # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
        #
        batch_size = coeffs.shape[0]
        theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)

        theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0) # theta2 is a matrix of shape [2, 3], it is the last row of the original transformation matrix repeated 2x
        theta2 = theta2.reshape(batch_size, 2, 3) # reshape from [batch_size*2, 3] to [batch_size, 2, 3] 

        d = 0.5
        base_grid = torch.empty(batch_size, oh, ow, 3, dtype=dtype, device=device)
        x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
        base_grid[..., 0].copy_(x_grid)
        y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
        base_grid[..., 1].copy_(y_grid)
        base_grid[..., 2].fill_(1)

        rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
        shape = (batch_size, oh * ow, 3)
        output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
        output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

        if center is not None:
            center = torch.tensor(center, dtype=dtype, device=device)
        else:
            center = 1.0

        output_grid = output_grid1.div_(output_grid2).sub_(center)
        return output_grid.view(batch_size, oh, ow, 2)
        
    def project_patch(self, patch, T, image):
        # using cv2 to project the patch instead of FAP place_patch() function,
        # since we don't need to calculate gradients
        width, height = image.shape[:2]
        # print(height, width)
        mask = np.ones_like(patch)

        warped_patch = cv2.warpPerspective(patch, T, (height, width), flags=cv2.INTER_NEAREST)
        mask = cv2.warpPerspective(mask, T, (height, width), flags=cv2.INTER_NEAREST)

        mod_img = image * ~mask.astype(bool)
        mod_img += warped_patch

        return mod_img # return a np array instead of jnp array and convert to double

        # sanity check with torch perspective grid
    def pt_project_patch(self, patch, T, base_img):
        patch_t = torch.tensor(patch, dtype=torch.float64, device=self.device).unsqueeze(0).unsqueeze(0)
        img = torch.tensor(base_img, device=self.device)

        p_height, p_width = patch.shape[-2:]
        i_height, i_width = img.shape[-2:]

        mask = torch.ones_like(patch_t, dtype=torch.float64, device=self.device)

        inv_t = np.linalg.inv(T)
        coeffs = torch.tensor(np.array([inv_t.flatten()]), dtype=torch.float64, device=self.device)
        
        grid = self._perspective_grid(coeffs, w=p_width, h=p_height, dtype=torch.float64, ow=i_width, oh=i_height, device=self.device, center = [1., 1.])

        bit_mask = torch.nn.functional.grid_sample(mask, grid, mode='bilinear', align_corners=False, padding_mode='zeros').bool()
        transformed_patch = torch.nn.functional.grid_sample(patch_t, grid, mode='bilinear', align_corners=False, padding_mode='zeros')

        modified_image = img * ~bit_mask.bool()
        modified_image += transformed_patch

        return modified_image
    
    def sim_new_pose(self, image: np.ndarray, drone_pose):
        # make sure images (plural if working with batches) are of correct shape
        assert np.prod(image.shape) % (96 * 160) == 0 
        image_t = torch.tensor(image).to(self.device)
        if len(image_t.shape) < 3:
            image_t = image_t.unsqueeze(0).unsqueeze(0)
        if len(image_t.shape) == 3:
            image_t = image_t.unsqueeze(1)
        
        # frontnet prediction
        if self.model == 'frontnet':
            x, y, z, yaw = self.pose_estimator(image_t)
            # reshape output and turn into numpy array
            predicted_pose = torch.hstack((x, y, z, yaw))
        # yolov5 prediction
        elif self.model == 'yolov5':
            predicted_pose = self.pose_estimator(image_t)

        predicted_pose = predicted_pose.detach().cpu().numpy()

        # print("predicted pose", predicted_pose, predicted_pose.shape)
        
        # calculate controller output
        new_setpoint = self._controller_setpoint(predicted_pose, drone_pose)

        return new_setpoint, predicted_pose

    def _controller_setpoint(self, predicted_poses, drone_pose):
        setpoints = []
        for predicted_pose in predicted_poses:
            # predicted_pose = [1., 0., 0., 0.]
            print("Drone pose in world: ", drone_pose)
            print("Predicted pose in body: ", predicted_pose)
            quats = rowan.from_euler(0., 0., drone_pose[3], convention='xyz') # returns qw, qx, qy, qz
            rotated_desired = rowan.rotate(quats, predicted_pose[:3])
            print("Predicted pose in world: ", rotated_desired)
            target_pos = drone_pose[:3] + rotated_desired
            print("Target pose in world: ", target_pos)

            # predicted yaw angles are discarded since they are very faulty
            global_pos = target_pos - drone_pose[:3]
            target_yaw = np.arctan2(global_pos[1], global_pos[0]) - np.pi

            new_setpoint = target_pos + self._calc_heading_vec(1., target_yaw)
            new_setpoint[2] = 1.
            print("New setpoint in world: ", new_setpoint, target_yaw)

            setpoints.append([*new_setpoint, target_yaw])
        
        return np.array(setpoints)

    def _calc_heading_vec(self, radius, angle):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y, 0.0])

    def _pred_to_numpy(self, prediction):
        x, y, z, yaw = prediction
        x = x.detach().cpu().squeeze(0).squeeze(0).numpy()
        y = y.detach().cpu().squeeze(0).squeeze(0).numpy()
        z = z.detach().cpu().squeeze(0).squeeze(0).numpy()
        yaw = yaw.detach().cpu().squeeze(0).squeeze(0).numpy()

        return np.array([x, y, z, yaw])

    def update(self, pose):
        self.pose = pose
        self.current_idx += 1

    def reset(self):
        self.pose = np.array([0., 0., 0.])
        self.current_idx = 0

    def eval(self, pose, desired_pose):
        l2_distances = np.linalg.norm((pose - desired_pose), ord=2)#, axis=1)
        return l2_distances
    
class SimulatorThread(Thread):
    def __init__(self, sim_new_pose, camera_images, point_from_xyz):
        super().__init__()
        self.simulator = sim_new_pose
        self.drone_pose = deque(maxlen=1)
        self.drone_pose.append(np.array([0., 0., 1., 0.])) # x, y, z, yaw

        self.all_poses = []

        # self.camera_images = deque(maxlen=1)
        self.camera_images = camera_images
        self.current_image = None
        self.point_from_xyz = point_from_xyz

        self._stay_alive = True

        self.dt = 0.1

        # only for debug plots
        self.target_trajectory = np.array([[0.0, 0.25, 1., 0.0],
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

    def run(self):
        i = 0
        while self._stay_alive:
            self.all_poses.append([time.time(), *self.drone_pose[0].tolist()])
            if self.camera_images:
                # self.current_image = self.camera_images.popleft()
            
            # if self.current_image is not None:
                current_setpoint, prediction_relative = self.simulator(self.camera_images[0], self.drone_pose[0])
                print(current_setpoint, current_setpoint.shape, prediction_relative, prediction_relative.shape)
                homogeneous_coords = np.hstack((prediction_relative[0, :3], 1.))
                point = self.point_from_xyz(homogeneous_coords)


                print(self.camera_images[0].shape)


                # print("current setpoint:", current_setpoint)

            if self.drone_pose and not np.allclose(self.drone_pose[0], current_setpoint[0]):
                current_pose = self.drone_pose[0] + ((current_setpoint[0] - self.drone_pose[0]) * 0.1)
                # print("current pose after added setpoint: ", current_pose)
                self.drone_pose.append(current_pose)


            self.all_poses.append([time.time(), *self.drone_pose[0].tolist()])

            if self.camera_images and self.drone_pose:
                fig = plt.figure(figsize=(10, 5))

                # Left subplot: Image with scattered point
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.set_title(f"Frontnet Prediction: {prediction_relative[0, :3]}")
                ax1.imshow(self.camera_images[0], cmap='gray')
                ax1.scatter(point[0], point[1], color='red')

                # Right subplot: Drone position in 3D
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.set_title("Drone Position in 3D")
                drone_positions = np.array(self.all_poses)[:, 1:4]  # Extract x, y, z positions
                ax2.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], label="Drone Path")
                ax2.plot(self.target_trajectory[:, 0], self.target_trajectory[:, 1], self.target_trajectory[:, 2], label="Target Trajectory", color='green')
                ax2.scatter(current_setpoint[0][0], current_setpoint[0][1], current_setpoint[0][2], color='red', label="Current Setpoint")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                ax2.set_zlabel("Z")

                # set ax2 limits
                ax2.set_xlim([-0.5, 2.5])
                ax2.set_ylim([-1.5, 1.5])
                ax2.set_zlim([0, 2.5])
                ax2.legend()

                plt.tight_layout()
                plt.savefig(f"results/simulation/patched_image_{i:04d}.png")
                plt.close()
                i += 1
            time.sleep(0.1)

    def update(self, image):
        self.camera_images.append(image)

    def close(self):
        self._stay_alive = False
        # self.join()


if __name__ == '__main__':

    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"

    model_type = 'frontnet'

    cf_sim = CFSim(model_type, dataset_path)

    patch = np.random.rand(3,3) * 255.
    base_img = cf_sim.base_img[0]

    print(patch.shape)
    print(base_img.shape)

    T = np.eye(3,3)   # basic transformation matrix
    T[0, 0] = T[1, 1] = 10. # scale factor
    # patch upper left corner at center of image
    T[0, 2] = 80.
    T[1, 2] = 48.

    mod_img = cf_sim.project_patch(patch, T, base_img).to(cf_sim.device)
    mod_img = mod_img.unsqueeze(0).unsqueeze(0).float()
    print(mod_img.shape, mod_img.dtype)

    # plot the modified image
    from matplotlib import pyplot as plt
    plt.imshow(mod_img.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    plt.savefig("patched_image.png")


    predicted_pose = cf_sim.sim_new_pose(mod_img)
    print(predicted_pose)

    # sanity check
    # project pose back to point in image frame
    # print(cf_sim.pose_estimator.cam.camera_extrinsic, cf_sim.pose_estimator.cam.camera_extrinsic.shape)
    # homogeneous_coords = np.hstack((predicted_pose[0, :3], 1.))
    # # print(homogeneous_coords, homogeneous_coords.shape)
    # point = cf_sim.pose_estimator.cam.point_from_xyz(homogeneous_coords)
    # print(point)

    # # plot point in mod_img
    # plt.imshow(mod_img.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    # plt.scatter(point[0], point[1], color='red')
    # plt.savefig("pred_pose.png")

    # out_pytorch = torch.hstack(cf_sim.pose_estimator(mod_img.unsqueeze(0).unsqueeze(0)))
    # print("Output frontnet: ", out_pytorch)
    # new_pose = cf_sim.sim_new_pose(mod_img)
    # print("Output controller: ", new_pose)

    # from matplotlib import pyplot as plt
    # plt.imshow(mod_img, cmap='gray')
    # plt.show()