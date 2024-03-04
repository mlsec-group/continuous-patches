import torch
import os, sys
import cv2
import numpy as np

sys.path.append('simulators/pulp-frontnet/PyTorch/Frontnet')

from Frontnet import FrontnetModel
from DataProcessor import DataProcessor
from Dataset import Dataset
from torch.utils import data

def load_dataset(path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9):
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

class CFSim():
    def __init__(self, model_path, target_trajectory):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frontnet = self.load_model(model_path, device)
        
        self.pose = np.array([0., 0., 0.])#, 0.]) # x, y, z, yaw
        
        self.current_idx = 0
        self.target_trajectory = target_trajectory

    def load_model(self, path, device, config="160x32"):
        """
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

        return model
    
    def project_patch(self, patch, T, image):
        # using cv2 to project the patch instead of FAP place_patch() function,
        # since we don't need to calculate gradients
        height, width = image.shape[:2]
        mask = np.ones_like(patch)
        warped_patch = cv2.warpPerspective(patch, T, (width, height), flags=cv2.INTER_NEAREST)
        mask = cv2.warpPerspective(mask, T, (width, height), flags=cv2.INTER_NEAREST)
        mod_img = image * ~mask.astype(bool)
        mod_img += warped_patch

        return mod_img
    
    def sim_new_pose(self):
        # TODO
        pass

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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = load_model("simulators/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device)
    model_path = "simulators/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt"

    # won't be necessary later
    sys.path.append('.')
    from util import bezier_curve
    control_points_bezier = np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
    target_trajectory = np.array(bezier_curve(control_points_bezier, 20))

    cf_sim = CFSim(model_path, target_trajectory)

    dataset, _ = load_dataset("simulators/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle")
    base_img, _ = dataset.dataset.__getitem__(0)
    base_img = torch.squeeze(base_img, 0).numpy()
    # print(base_img.min(), base_img.max())

    patch = np.random.rand(10, 10) * 255.

    sf = 5.
    tx = 80.
    ty = 40.

    T = np.array([[sf, 0, tx],
                 [0, sf, ty],
                 [0, 0, 1.]])
    
    import matplotlib.pyplot as plt
    mod_img = cf_sim.project_patch(patch, T, base_img)

    plt.imshow(mod_img, cmap='gray')
    plt.show()