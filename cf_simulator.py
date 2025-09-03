import os
import sys


# Resolve absolute path to Frontnet.py
project_root = os.path.dirname(os.path.abspath(__file__))  # continuous-patches
frontnet_dir = os.path.join(project_root, 'pulp-frontnet', 'PyTorch')
print(frontnet_dir)
sys.path.insert(0, frontnet_dir)  # insert at front so it's prioritized

import torch
import numpy as np

from Frontnet.Frontnet import FrontnetModel
from Frontnet.DataProcessor import DataProcessor
from Frontnet.Dataset import Dataset
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from pathlib import Path


class CFSim():
    def __init__(self, model='frontnet', dataset_path="pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        
        if self.model == 'frontnet':
            self.pose_estimator = self.load_frontnet_model(self.device)
        else:
            raise ValueError("Model type not supported!")
        
        self.pose = torch.tensor([0., 0., 1., 0.], device=self.device, dtype=torch.float32) # x, y, z, yaw
        self.velocity = 0.5 # 1 m/s
        self.omega = 1. # 1.0 rad/s

        self.all_poses = [self.pose.detach().cpu().numpy()]

        self.train_set, self.test_set = self.load_dataset(dataset_path)

        # patch monitor coordinates in homogeneous coordinates in image frame
        monitor_corners = np.array([[48., 20., 1.],     # upper left corner (x, y, 1)
                                    [128., 20., 1.],    # upper right corner
                                    [48., 66., 1.],     # lower left corner
                                    [128., 66., 1]])   # lower right corner
        
        self.max_patch_width = monitor_corners[1, 0] - monitor_corners[0, 0]  # width of patch monitor in pixels
        self.max_patch_height = monitor_corners[2, 1] - monitor_corners[0, 1] # height of patch monitor in pixels

        self.monitor_corners = torch.tensor(monitor_corners, dtype=torch.float32, device=self.device)

        self.score = 1.
        self.current_idx = 1
        t = np.linspace(0, 2 * np.pi, 20)
        x = 0.5 * np.sin(2 * t)  # Horizontal figure 8
        y = 1.5 * np.sin(t)  # Vertical figure 8
        z = np.ones_like(t)  # Constant height at 1
        yaw = np.zeros_like(t)  # Constant yaw
        target_trajectory = np.column_stack((x, y, z, yaw))
        self.target_trajectory = torch.tensor(target_trajectory, dtype=torch.float32, device=self.device)
        
    
    def load_frontnet_model(self, device, model_path="pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", config="160x32"):
        """
        From FAP repo
        Loads a saved Frontnet model from the given path with the set configuration and moves it to CPU/GPU.
        Parameters
            ----------
            path
                The path to the stored Frontnet model
            device
                A PyTorch device (either CPU or GPU)
            config
                The architecture configuration of the Frontnet model. Must be one of ['160x32', '160x16', '80x32']
        """
        assert config in FrontnetModel.configs.keys(), 'config must be one of {}'.format(list(FrontnetModel.configs.keys()))
        
        # get correct architecture configuration
        model_params = FrontnetModel.configs[config]
        # initialize a random model with configuration
        model = FrontnetModel(**model_params).to(device)
        
        # load the saved model 
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model'])
        except RuntimeError:
            print("RuntimeError while trying to load the saved model!")
            print("Seems like the model config does not match the saved model architecture.")
            print("Please check if you're loading the right model for the chosen config!")

        return model.eval()
    
    # only needed during testing
    def load_dataset(self, path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9):
        # From FAP repo
        # load images and labels from the stored dataset
        path = Path(path)
        [images, labels] = DataProcessor.ProcessTestData(path)

        # init RNG for loading the data always with the same key to
        # ensure the same images end up in train and test set respectively
        rng = np.random.default_rng(1749)

        # split dataset into train and test set
        indices = np.arange(len(images))
        rng.shuffle(indices)
        split_idx = int(len(images) * train_set_size)

        train_set = Dataset(images[indices[:split_idx]], labels[indices[:split_idx]])
        test_set = Dataset(images[indices[split_idx:]], labels[split_idx:])

        # for quick and convinient access, create a torch DataLoader with the given parameters
        data_params = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last':drop_last, 'num_workers': num_workers}
        train_loader = DataLoader(train_set, **data_params)
        test_loader = DataLoader(test_set, **data_params)

        return train_loader, test_loader


    def _perspective_grid(self,
    coeffs: list[float], 
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
        
    def project_patch(self, patch, T, img):
        """ Project the patch on the camera image.
        """
        if not torch.is_tensor(patch):
            patch = torch.tensor(patch, dtype=torch.float64, device=self.device)
        # patches should be of shape [B, 1, H, W]
        if len(patch.shape) < 3:
            patch = patch.unsqueeze(0)
        if len(patch.shape) == 3:
            patch = patch.unsqueeze(1)
        
        
        if not torch.is_tensor(img):
            img = torch.tensor(img, device=self.device)
        # img should be of shape [1, 1, H, W]
        while len(img.shape) < 4:
            img = img.unsqueeze(0)

        if not torch.is_tensor(T):
            T = torch.tensor(T, dtype=torch.float32, device=self.device) # shape should be [3, 3]

        p_height, p_width = patch.shape[-2:]
        i_height, i_width = img.shape[-2:]

        mask = torch.ones_like(patch, dtype=torch.float32, device=self.device)

        try:
            inv_t = torch.inverse(T)
        except Exception as e:
            print("Error inverting transformation matrix:", e)
            print("Transformation matrix T:", T)

        coeffs = inv_t.flatten().unsqueeze(0)
        
        grid = self._perspective_grid(coeffs, w=p_width, h=p_height, dtype=torch.float32, ow=i_width, oh=i_height, device=self.device, center = [1., 1.])

        bit_mask = torch.nn.functional.grid_sample(mask, grid, mode='bilinear', align_corners=False, padding_mode='zeros').bool()
        transformed_patch = torch.nn.functional.grid_sample(patch, grid, mode='bilinear', align_corners=False, padding_mode='zeros')

        modified_image = img * ~bit_mask.bool()
        modified_image += transformed_patch

        return modified_image
    
    def sim_new_pose(self, image):
        """ Perform one flight simulation step without updating the simulator state.
        """
        # make sure images (plural if working with batches) are of correct shape
        assert np.prod(image.shape) % (96 * 160) == 0

        if len(image.shape) < 3:
            image = image.unsqueeze(0).unsqueeze(0)
        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        
        x, y, z, yaw = self.pose_estimator(image)
        # All values are in meters and radians and in the coordinate frame relative to the drone
        # x is the distance in meters to the drone, higher x -> further away from the drone
        # y is the distance in meters to the drone, y > 0 -> further to the right of the drone
        # z is the height in meters above the drone, z > 0 -> higher above the drone
        # Don't get confused with the y values!

        # reshape output
        predicted_pose = torch.hstack((x, y, z, yaw)).squeeze(0)
        # print("Predicted pose: ", predicted_pose, predicted_pose.shape)
        
        # calculate controller output
        new_setpoint = self._controller_setpoint(predicted_pose)
        # print("New setpoint: ", new_setpoint, new_setpoint.shape)

        # # update drone pose after 1s
        # delta = new_setpoint[:3] - self.pose[:3]
        # distance = torch.linalg.norm(delta) + 1e-6  # add small value to avoid division by zero
        # unit_direction = delta / distance # normalize direction

        # new_position = self.pose[:3] + unit_direction * self.velocity  # move 0.5 m in the direction of the delta
        
        # delta_yaw = self._normalize_yaw(new_setpoint[3] - self.pose[3])
        # new_yaw = self.pose[3] + delta_yaw * self.omega  # turn 1 rad/s in the direction of the delta yaw

        #new_pose = torch.stack((new_position[0], new_position[1], new_position[2], new_yaw), dim=-1)
        
        # print("New pose: ", new_pose, new_pose.shape)
        #return new_pose
        return new_setpoint

    def _get_T_matrix(self, pose):
        T = torch.eye(4, dtype=torch.float32).to(self.device)
        normalized_yaw = self._normalize_yaw(pose[3])

        sin_yaw = torch.sin(normalized_yaw).to(torch.float32)
        cos_yaw = torch.cos(normalized_yaw).to(torch.float32)
        zero = torch.zeros_like(sin_yaw, device=self.device, dtype=torch.float32)

        rotation_matrix_row1 = torch.stack([cos_yaw, -sin_yaw, zero], dim=-1)
        rotation_matrix_row2 = torch.stack([sin_yaw, cos_yaw, zero], dim=-1)
        rotation_matrix_row3 = torch.tensor([0., 0., 1.], device=self.device, dtype=torch.float32)
        R = torch.stack((rotation_matrix_row1, rotation_matrix_row2, rotation_matrix_row3), dim=0)

        T[:3, :3] = R
        T[:3, 3] = pose[:3]
        return T

    def _controller_setpoint(self, predicted_pose):
        T_pred_in_drone = self._get_T_matrix(predicted_pose)  # predicted pose in drone frame (== relative to the drone)

        T_drone_in_world = self._get_T_matrix(self.pose)    # drone pose in world frame

        # rotate predicted pose from drone into world frame
        T_pred_in_world = T_drone_in_world @ T_pred_in_drone
        
        target_yaw = self._normalize_yaw(predicted_pose[3])
        T_direction_world = torch.eye(4, dtype=torch.float32).to(self.device)
        T_direction_world[:3, 3] = self._calc_heading_vec(1., self._normalize_yaw(target_yaw-torch.pi)).to(self.device)

        T_setpoint_world = T_direction_world @ T_pred_in_world     # rotate the prediction, such that the drone stays 1 m away from it

        return torch.stack([*T_setpoint_world[:3, 3], target_yaw], dim=-1)

        # normalized_yaw = self._normalize_yaw(self.pose[-1])

        # sin_yaw = torch.sin(normalized_yaw).to(torch.float32)
        # cos_yaw = torch.cos(normalized_yaw).to(torch.float32)
        # zero = torch.zeros_like(sin_yaw, device=self.device, dtype=torch.float32)
        # rotation_matrix_row1 = torch.stack([cos_yaw, -sin_yaw, zero], dim=-1)
        # rotation_matrix_row2 = torch.stack([sin_yaw, cos_yaw, zero], dim=-1)
        # rotation_matrix_row3 = torch.tensor([0., 0., 1.], device=self.device, dtype=torch.float32)

        # R = torch.stack((rotation_matrix_row1, rotation_matrix_row2, rotation_matrix_row3), dim=0)
    
        # target_pos = predicted_pose[:3] + R @ predicted_pose[:3]


        # target_yaw = predicted_pose[-1] - np.pi
        # target_yaw = self._normalize_yaw(target_yaw)


        # new_setpoint = target_pos + self._calc_heading_vec(1., target_yaw) # keep a saftey distance of 1 m to predicted human

        # return torch.stack((new_setpoint[0], new_setpoint[1], new_setpoint[2], target_yaw), dim=-1)


    def _calc_heading_vec(self, radius, angle):
        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        zero = torch.zeros_like(x, device=self.device, dtype=torch.float32)
        return torch.stack((x, y, zero), dim=-1)

    def _normalize_yaw(self, yaw):
       return torch.atan2(torch.sin(yaw), torch.cos(yaw)) # normalize angle to [-pi, pi] range

    def _pred_to_numpy(self, prediction):
        x, y, z, yaw = prediction
        x = x.detach().cpu().squeeze(0).squeeze(0).numpy()
        y = y.detach().cpu().squeeze(0).squeeze(0).numpy()
        z = z.detach().cpu().squeeze(0).squeeze(0).numpy()
        yaw = yaw.detach().cpu().squeeze(0).squeeze(0).numpy()

        return np.array([x, y, z, yaw])

    def update(self, pose):
        # self.score *= 1000
        self.pose = pose
        self.all_poses.append(pose.detach().cpu().numpy())

        self.score -= self.eval(pose) * 10.
        self.score = max(self.score, 0.)  # ensure score is not negative
        if self.current_idx <= len(self.target_trajectory):
            self.current_idx += 1
        # self.score /= 1000

    def reset(self):
        self.pose = torch.tensor([0., 0., 1., 0.], device=self.device, dtype=torch.float32)
        self.current_idx = 1

    def eval(self, pose):
        l2_distance = torch.linalg.norm((pose[:3] - self.target_trajectory[self.current_idx, :3]), ord=2)
        angular_loss = 1 - torch.cos(pose[3] - self.target_trajectory[self.current_idx, 3])
        
        error = l2_distance + angular_loss

        return error
    
    def render(self, im_name='drone_trajectory.png'):
        # a bit rudimentary, but you get the idea
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Right subplot: Drone position in 2d
        ax.set_title("Drone Position in 2D")
        drone_positions = np.array(self.all_poses)
        
        ax.plot(drone_positions[:, 0], drone_positions[:, 1], label="Drone Trajectory")
        ax.plot(self.target_trajectory[:, 0], self.target_trajectory[:, 1], label="Target Trajectory", color='green')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax2.set_zlabel("Z")

        # add current drone position as scatter with arrow for yaw
        # print(self.pose)
        pose = self.pose.detach().cpu().numpy()
        ax.scatter(pose[0], pose[1], color='black')
        ax.arrow(pose[0], pose[1],
                    0.3 * np.cos(pose[3]), 0.3 * np.sin(pose[3]),
                    head_width=0.1, head_length=0.1, fc='black', ec='black')

        # set ax2 limits
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-1.5, 1.5])
        # ax2.set_zlim([0, 2.5])
        ax.legend()

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig( f"results/{im_name}", dpi=300)
        plt.close()
