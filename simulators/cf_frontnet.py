import torch
import os, sys
import cv2
import numpy as np
import jax
import jax.numpy as jnp

from copy import deepcopy

from dlpack import asdlpack


sys.path.append('simulators/pulp-frontnet/PyTorch/Frontnet')

from Frontnet import FrontnetModel
from DataProcessor import DataProcessor
from Dataset import Dataset
from torch.utils import data

import rowan

class CFSim():
    def __init__(self, target_trajectory, model_path="simulators/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", dataset_path="simulators/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pose_estimator = self.load_model(model_path, self.device)
        
        self.pose = np.array([0., 0., 0., 0.]) # x, y, z, yaw
        
        self.current_idx = 0
        self.target_trajectory = target_trajectory

        # might be deleted later, the dataset is only loaded to get a suitable background image
        dataset, _ = self.load_dataset(dataset_path)
        base_img, gt = dataset.dataset.__getitem__(0)
        self.base_img = base_img.squeeze(0).numpy()

        # patch stays random for now and inside the simulator for compatibility with current
        # optimize script
        self.patch = np.random.rand(10, 10).astype(np.float32) * 255. # load one of the optimized FAPs instead!

    def load_model(self, path, device, config="160x32"):
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
            model.load_state_dict(torch.load(path, map_location=device)['model'])
        except RuntimeError:
            print("RuntimeError while trying to load the saved model!")
            print("Seems like the model config does not match the saved model architecture.")
            print("Please check if you're loading the right model for the chosen config!")

        return model.eval()
    
    def load_dataset(self, path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9):
        # From FAP repo
        # load images and labels from the stored dataset
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
        train_loader = data.DataLoader(train_set, **data_params)
        test_loader = data.DataLoader(test_set, **data_params)

        return train_loader, test_loader
    
    # additional wrapper to hand over parameters easily
    def _warp_perspective_wrapper(self, patch, T, width, height, flags):
        return cv2.warpPerspective(patch, T, (int(height), int(width)), flags=int(flags)).astype(patch.dtype)
    
    # needed to work with the intermediate values stored in T
    # this is necessary due to:
    # https://jax.readthedocs.io/en/latest/faq.html#how-can-i-convert-a-jax-tracer-to-a-numpy-array
    # https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables
    def warp_perspective_callback(self, patch, T, shape):
        result_shape_dtype = jax.core.ShapedArray(shape, patch.dtype)
        mod_img = jax.pure_callback(self._warp_perspective_wrapper, result_shape_dtype, patch, T, *shape, cv2.INTER_NEAREST)
        return mod_img
    
    def project_patch(self, patch, T, image):
        # using cv2 to project the patch instead of FAP place_patch() function,
        # since we don't need to calculate gradients
        width, height = image.shape[:2]
        # print(height, width)
        mask = np.ones_like(patch)
        copied_t = deepcopy(T)
        # warped_patch = cv2.warpPerspective(patch, jnp.asarray(T.copy(), dtype=np.float64), (width, height), flags=cv2.INTER_NEAREST)
        # mask = cv2.warpPerspective(mask, jnp.asarray(T.copy()), (width, height), flags=cv2.INTER_NEAREST)
        # print(T, T.shape)
        # warped_patch = transforms.apply_transform(patch, T, mask_value=jnp.array([255]), bilinear=False).squeeze(2)
        # mask = transforms.apply_transform(mask, T, mask_value=jnp.array([255]), bilinear=False).squeeze(2)

        warped_patch = self.warp_perspective_callback(patch, copied_t, (width, height))
        mask = self.warp_perspective_callback(mask, copied_t, (width, height))

        mod_img = image * ~mask.astype(bool)
        mod_img += warped_patch

        return mod_img # return a np array instead of jnp array and convert to double
    
    def sim_new_pose(self, params):
        scale_factor, tx, ty = params
        T = jnp.array([[scale_factor, 0, tx],
                 [0, scale_factor, ty],
                 [0, 0, 1.]])
        
        mod_img = asdlpack(self.project_patch(self.patch, T, self.base_img))

        mod_img_t = torch.from_dlpack(mod_img).unsqueeze(0).unsqueeze(0).to(self.device)
        prediction = self.pose_estimator(mod_img_t)
        predicted_pose = self._pred_to_numpy(prediction)
        
        new_setpoint = self._controller_setpoint(predicted_pose)

        return new_setpoint

    def _controller_setpoint(self, predicted_pose):
        quats = rowan.from_euler(0., 0., self.pose[3], convention='xyz') # returns qw, qx, qy, qz
        rotated_desired = rowan.rotate(quats, predicted_pose[:3])
        target_pos = predicted_pose[:3] + rotated_desired

        # predicted yaw angles are discarded since they are very faulty
        global_pos = target_pos - self.pose[:3]
        target_yaw = np.arctan2(global_pos[1], global_pos[0]) - np.pi

        new_setpoint = target_pos + self._calc_heading_vec(1., target_yaw)

        return [*new_setpoint, target_yaw]

    def _calc_heading_vec(self, radius, angle):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y, 0.0])

    def _pred_to_numpy(self, prediction):
        x, y, z, yaw = prediction
        x = x.detach().squeeze(0).squeeze(0).numpy()
        y = y.detach().squeeze(0).squeeze(0).numpy()
        z = z.detach().squeeze(0).squeeze(0).numpy()
        yaw = yaw.detach().squeeze(0).squeeze(0).numpy()

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


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", False)
    model_path = "simulators/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt"
    dataset_path = "simulators/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"

    # won't be necessary later
    sys.path.append('.')
    from util import bezier_curve
    control_points_bezier = np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
    target_trajectory = np.array(bezier_curve(control_points_bezier, 20))

    cf_sim = CFSim(target_trajectory, model_path)

    # patch = np.random.rand(10, 10) * 255.

    sf = 5.
    tx = 80.
    ty = 40.

    T = jnp.array([[sf, 0, tx],
                 [0, sf, ty],
                 [0, 0, 1.]])
    
    import matplotlib.pyplot as plt
    mod_img = cf_sim.project_patch(cf_sim.patch, T, cf_sim.base_img)
    plt.imshow(mod_img, cmap='gray')

    new_pose = cf_sim.sim_new_pose([sf, tx, ty])
    print(new_pose)

    plt.show()