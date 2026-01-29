import torch
import sys
import os

# Resolve repo-relative path to pulp-frontnet PyTorch so imports work regardless of cwd
project_root = os.path.dirname(os.path.abspath(__file__))  # /home/piha/continous-patches
frontnet_dir = os.path.join(project_root, 'pulp-frontnet', 'PyTorch')
if frontnet_dir not in sys.path:
    sys.path.insert(0, frontnet_dir)

from Frontnet.Frontnet import FrontnetModel

from Frontnet.DataProcessor import DataProcessor
from Frontnet.Dataset import Dataset
from torch.utils import data

import rowan

# import nemo
# from Frontnet.Utils import ModelManager

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
# import random

# import onnx
# from onnx import numpy_helper

import glob

DEBUG_GRAD = False

def normalize_yaw_t(yaw):
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))

def normalize_yaw(yaw):
    return np.atan2(np.sin(yaw), np.cos(yaw))

def dist(c1, c2):
    elementwise = torch.square(c1 - c2)
    # elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
    return torch.sqrt(torch.sum(elementwise, axis=-1))


def bb2camera(bbox, intrinsic, dist_coeffs):
    fx = np.array(intrinsic)[0][0]
    fy = np.array(intrinsic)[1][1]
    ox = np.array(intrinsic)[0][2]
    oy = np.array(intrinsic)[1][2]
    
    ul_image = np.array([bbox[0], bbox[1]], dtype=np.float32)
    ur_image = np.array([bbox[2], bbox[1]], dtype=np.float32)
    ll_image = np.array([bbox[0], bbox[3]], dtype=np.float32)
    lr_image = np.array([bbox[2], bbox[3]], dtype=np.float32)
    center_image = np.array([(ul_image[0] + lr_image[0])/2, (ul_image[1] + lr_image[1])/2], dtype=np.float32)
    

    ul_camera = np.array([(ul_image[0]-ox)/fx, (ul_image[1]-oy)/fy, 1.0], dtype=np.float32)
    ur_camera = np.array([(ur_image[0]-ox)/fx, (ur_image[1]-oy)/fy, 1.0], dtype=np.float32)
    ll_camera = np.array([(ll_image[0]-ox)/fx, (ll_image[1]-oy)/fy, 1.0], dtype=np.float32)
    lr_camera = np.array([(lr_image[0]-ox)/fx, (lr_image[1]-oy)/fy, 1.0], dtype=np.float32)
    center_camera = np.array([(center_image[0]-ox)/fx, (center_image[1]-oy)/fy, 1.0], dtype=np.float32)


    ul_camera_norm = ul_camera / np.linalg.norm(ul_camera)
    ur_camera_norm = ur_camera / np.linalg.norm(ur_camera)
    ll_camera_norm = ll_camera / np.linalg.norm(ll_camera)
    lr_camera_norm = lr_camera / np.linalg.norm(lr_camera)
    center_camera_norm = center_camera / np.linalg.norm(center_camera)
    
    center_camera = np.array([0., 0., 0.])
    
    return center_camera, ul_camera_norm, ur_camera_norm, ll_camera_norm, lr_camera_norm, center_camera_norm

def line_plane_intersection(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):
    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("No intersection or line is within plane")
    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray_direction + plane_point
    return Psi

def scale_tx_ty(sf, tx, ty, patch_size=80, image_size=(96, 160)):
    scaled_patch_size = patch_size * sf
    max_tx = image_size[1] - scaled_patch_size
    max_ty = image_size[0] - scaled_patch_size
    return tx * max_tx, ty * max_ty

def printd(*args, **kwargs):
    if DEBUG_GRAD:
        print(*args, **kwargs)

# Helper to interpolate exactly N points evenly along a path of corners
def interpolate_path(corners, n_points):
    # Calculate lengths of each segment
    dists = np.sqrt(np.sum(np.diff(corners, axis=0)**2, axis=1))
    # Cumulative distance (0, d1, d1+d2, ...)
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    
    # We want n_points distributed evenly from 0 to total_dist
    even_dists = np.linspace(0, total_dist, n_points)
    
    # Interpolate X and Y based on distance
    x = np.interp(even_dists, cumulative_dist, corners[:, 0])
    y = np.interp(even_dists, cumulative_dist, corners[:, 1])
    
    return np.column_stack([x, y])

def gen_target_trajectory(trajectory, monitor_center=[2., 0., 1.], num_steps=25):
    """
    Generates a trajectory with EXACTLY num_steps points.
    Bounds: x in [-0.6, 0.6], y in [-1.0, 1.0].
    """
    z_val = 1.0
    yaw_val = 0.0
    xy_points = None

    if trajectory == 'square':
        # Define the corners of the path
        corners = np.array([
            [0.6, 0.5],     # Top-Right
            [-0.6, 0.5],    # Top-Left
            [-0.6, -0.5],   # Bottom-Left
            [0.6, -0.5],    # Bottom-Right
            [0.6, 0.5],     # Top-Right
            
        ])
        xy_points = interpolate_path(corners, num_steps)

    elif trajectory == 'circle':
        # Parametric generation naturally supports exact counts
        t = np.linspace(0, 2 * np.pi, num_steps)
        x = 0.5 * np.cos(t) # Radius 0.5 fits in [-0.6, 0.6]
        y = 0.5 * np.sin(t)
        xy_points = np.column_stack([x, y])

    elif trajectory == 'line_x':
        # Center -> Right -> Left -> Center
        corners = np.array([
            [0., 0.],
            [0.6, 0.],
            [-0.6, 0.],
            [0., 0.]
        ])
        xy_points = interpolate_path(corners, num_steps)

    elif trajectory == 'line_y':
        # Center -> Up -> Down -> Center
        corners = np.array([
            [0., 0.],
            [0., 1.0],
            [0., -1.0],
            [0., 0.]
        ])
        xy_points = interpolate_path(corners, num_steps)

    elif trajectory == 'figure8':
        t = np.linspace(0, 2 * np.pi, num_steps)
        x = 0.5 * np.sin(2 * t) # Width 1.0 (fits -0.6 to 0.6)
        y = 0.9 * np.sin(t)     # Height 1.8 (fits -1.0 to 1.0)
        xy_points = np.column_stack([x, y])

    elif trajectory == 'diagonal_line':
        # Top-Right -> Bottom-Left -> Center
        corners = np.array([
            [0., 0.],
            [0.6, 0.8],
            [-0.6, -0.8],
            [0., 0.]
        ])
        xy_points = interpolate_path(corners, num_steps)

    elif trajectory == 'triangle':
        # Triangle shape adjusted to fit bounds
        corners = np.array([
            [0.6, 0.],    # Start Right
            [-0.2, 0.8],    # top
            [-0.2, -0.8],   # bottom
            [0.6, 0.0]     # Close loop
        ])
        
        # Add initial hover at center to match previous logic safely
        # Note: This adds distance from center to start point
        # full_path = np.vstack([
        #     np.array([0., 0.]), # Start at center (0,0)
        #     corners
        # ])
        xy_points = interpolate_path(corners, num_steps)

    elif trajectory == 's':
        # Scaling parameters
        rx = 0.6  # Horizontal radius
        ry = 0.5  # Vertical radius for each half

        # Top arc: Center (0, 0.5)
        t1 = np.linspace(0.1 * np.pi, 1.5 * np.pi, num_steps // 2)
        x1 = rx * np.cos(t1)
        y1 = ry * np.sin(t1) + 0.5

        # Bottom arc: Center (0, -0.5)
        t2 = np.linspace(0.5 * np.pi, -0.9 * np.pi, num_steps // 2 + num_steps % 2)
        x2 = rx * np.cos(t2)
        y2 = ry * np.sin(t2) - 0.5

        # Combine
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        xy_points = np.column_stack([x, y])


    elif trajectory == 'c':
        # Elliptical Arc opening to the Right
        # t goes from 45 degrees to 315 degrees
        t = np.linspace(np.pi/4, 7*np.pi/4, num_steps)
        x = 0.5 * np.cos(t) # x > 0 at start/end, x < 0 in middle (Back of C)
        y = 0.8 * np.sin(t) # Stretched vertically to fill y bounds
        
        # Current logic creates a C opening to the LEFT (x is neg in middle). 
        # Flip x to open RIGHT:
        x = x + 0.2 # Shift slightly right so it centers better
        
        xy_points = np.column_stack([x, y])
    
    elif trajectory == 'u':
        # U shape: Semi-ellipse opening upwards
        # Parameters
        rx = 0.6            # Width from center to side
        ry = 0.6            # Height of the curved part
        y_transition = -0.4 # y-coordinate where lines meet the curve (-1 + 0.6)

        # 1. Left Vertical Line: From top left (-0.6, 1) down to (-0.6, -0.4)
        y1 = np.linspace(1, y_transition, num_steps // 3)
        x1 = np.full_like(y1, -rx)

        # 2. Bottom Arc: Semi-ellipse centered at (0, -0.4)
        # Angle ranges from pi (left) to 2pi (right) to sweep the bottom
        t = np.linspace(np.pi, 2 * np.pi, num_steps // 3 + num_steps % 3)
        x2 = rx * np.cos(t)
        y2 = ry * np.sin(t) + y_transition

        # 3. Right Vertical Line: From (0.6, -0.4) up to top right (0.6, 1)
        y3 = np.linspace(y_transition, 1, num_steps // 3 )
        x3 = np.full_like(y3, rx)

        # Combine
        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])

        # print(x.shape, y.shape)
        xy_points = np.column_stack([x, y])

    elif trajectory == 'slingshot_left':
        start = np.array([0., 0.])
        point = np.array([0., 3.0])
        xy_points = point.reshape(1, 2).repeat(num_steps, axis=0)
        xy_points[0] = start
    elif trajectory == 'slingshot_right':
        start = np.array([0., 0.])
        point = np.array([0., -3.0])
        xy_points = point.reshape(1, 2).repeat(num_steps, axis=0)
        xy_points[0] = start
    elif trajectory == 'slingshot_forward':
        start = np.array([0., 0.])
        point = np.array([3.0, 0.])
        xy_points = point.reshape(1, 2).repeat(num_steps, axis=0)
        xy_points[0] = start
    elif trajectory == 'slingshot_backward':
        start = np.array([0., 0.])
        point = np.array([-3.0, 0.])
        xy_points = point.reshape(1, 2).repeat(num_steps, axis=0)
        xy_points[0] = start

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory}")

    # Combine with Z and Yaw
    z = np.ones((num_steps, 1)) * z_val
    # calculate yaw to always face monitor center
    yaw = np.arctan2(monitor_center[1] - xy_points[:,1], monitor_center[0] - xy_points[:,0]).reshape(-1, 1)
    # Result shape: (num_steps, 4)
    waypoints = np.hstack([xy_points, z, yaw])

    # --- SAFETY PADDING ---
    # Because your loop runs until `len - 1`, we append the final point 5 times.
    # This ensures the drone stays at the final goal for a few frames 
    # instead of cutting off early.
    last_point = waypoints[-1]
    padding = np.tile(last_point, (5, 1))
    waypoints = np.vstack((waypoints, padding))

    return torch.tensor(waypoints, dtype=torch.float32)        

def load_model(path, device, config):
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
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
    except RuntimeError:
        print("RuntimeError while trying to load the saved model!")
        print("Seems like the model config does not match the saved model architecture.")
        print("Please check if you're loading the right model for the chosen config!")

    return model

def load_quantized(path, device):
    model = FrontnetQuantizedModel(path, device)

    return model

class FrontnetQuantizedModel(torch.nn.Module):
    def __init__(self, path, device):
        super(FrontnetQuantizedModel, self).__init__()

        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        self.weights = {}
        for node in onnx_model.graph.initializer:
            data = torch.tensor(numpy_helper.to_array(node)).to(device)
            if 'kappa' in node.name:
                name = node.name.replace('kappa', 'gamma')
            elif 'lamda' in node.name:
                name = node.name.replace('lamda', 'beta')
            else:
                name = node.name
            self.weights.update({name: data})
        

        constants = []
        for node in onnx_model.graph.node:
            if node.op_type == "Constant":
                const_value = numpy_helper.to_array(node.attribute[0].t)
                constants.append(const_value.item())
        self.constants = torch.tensor(np.array(constants)).to(device)
        

    def forward(self, x):
        # layer 0
        conv0 = torch.nn.functional.conv2d(x, self.weights['conv.weight'], stride=2, padding=2, bias=None)
        bn0 = self.weights['bn.gamma'] * conv0 + self.weights['bn.beta']
        mul0 = bn0 * self.constants[0]
        div0 = mul0 / self.constants[1]
        rel0 = torch.nn.functional.relu(div0)
        max0 = torch.nn.functional.max_pool2d(rel0, kernel_size=2, stride=2, padding=0)


        # layer 1
        conv1_1 = torch.nn.functional.conv2d(max0, self.weights['layer1.conv1.weight'], stride=2, padding=1, bias=None)
        bn1_1 = self.weights['layer1.bn1.gamma'] * conv1_1 + self.weights['layer1.bn1.beta']
        mul1_1 = bn1_1 * self.constants[2]
        div1_1 = mul1_1 / self.constants[3]
        rel1_1 = torch.nn.functional.relu(div1_1)

        conv1_2 = torch.nn.functional.conv2d(rel1_1, self.weights['layer1.conv2.weight'], stride=1, padding=1, bias=None)
        bn1_2 = self.weights['layer1.bn2.gamma'] * conv1_2 + self.weights['layer1.bn2.beta']
        mul1_2 = bn1_2 * self.constants[4]
        div1_2 = mul1_2 / self.constants[5]
        rel1_2 = torch.nn.functional.relu(div1_2)

        # layer 2
        conv2_1 = torch.nn.functional.conv2d(rel1_2, self.weights['layer2.conv1.weight'], stride=2, padding=1, bias=None)
        bn2_1 = self.weights['layer2.bn1.gamma'] * conv2_1 + self.weights['layer2.bn1.beta']
        mul2_1 = bn2_1 * self.constants[6]
        div2_1 = mul2_1 / self.constants[7]
        rel2_1 = torch.nn.functional.relu(div2_1)

        conv2_2 = torch.nn.functional.conv2d(rel2_1, self.weights['layer2.conv2.weight'], stride=1, padding=1, bias=None)
        bn2_2 = self.weights['layer2.bn2.gamma'] * conv2_2 + self.weights['layer2.bn2.beta']
        mul2_2 = bn2_2 * self.constants[8]
        div2_2 = mul2_2 / self.constants[9]
        rel2_2 = torch.nn.functional.relu(div2_2)

        # layer 3
        conv3_1 = torch.nn.functional.conv2d(rel2_2, self.weights['layer3.conv1.weight'], stride=2, padding=1, bias=None)
        bn3_1 = self.weights['layer3.bn1.gamma'] * conv3_1 + self.weights['layer3.bn1.beta']
        mul3_1 = bn3_1 * self.constants[10]
        div3_1 = mul3_1 / self.constants[11]
        rel3_1 = torch.nn.functional.relu(div3_1)

        conv3_2 = torch.nn.functional.conv2d(rel3_1, self.weights['layer3.conv2.weight'], stride=1, padding=1, bias=None)
        bn3_2 = self.weights['layer3.bn2.gamma'] * conv3_2 + self.weights['layer3.bn2.beta']
        mul3_2 = bn3_2 * self.constants[12]
        div3_2 = mul3_2 / self.constants[13]
        rel3_2 = torch.nn.functional.relu(div3_2)

        flat = rel3_2.flatten(1)
        out = torch.nn.functional.linear(flat, self.weights['fc.weight'], self.weights['fc.bias'])      
        x_q = out[:, 0]
        y_q = out[:, 1]
        z_q = out[:, 2]
        phi_q = out[:, 3]

        # de-quantize results
        x = x_q * 2.46902e-05 + 1.02329e+00
        y = y_q * 2.46902e-05 + 7.05523e-04
        z = z_q * 2.46902e-05 + 2.68245e-01
        phi = phi_q * 2.46902e-05 + 5.60173e-04

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return [x, y, z, phi]

    # def forward(self, batch):
    #     batch_x = []
    #     batch_y = []
    #     batch_z = []
    #     batch_phi = []
    #     for x in batch:
    #         x, y, z, phi = self._forward_single(x)
    #         batch_x.append(x)
    #         batch_y.append(y)
    #         batch_z.append(z)
    #         batch_phi.append(phi)

    #     return [torch.cat(batch_x), torch.cat(batch_y), torch.cat(batch_z), torch.cat(batch_phi)]

def load_dataset(path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True):
    """
    Loads a dataset from the given path. 
    Parameters
        ----------
        path
            The path to the dataset
        batch_size
            The size of the batches the dataset will contain
        shuffle
            If set to True, the data will be shuffled randomly
        drop_last
            If set to True, the last batch of the dataset will be dropped. 
            This ensures that all returned batches are of the same size.
        num_workers
            Set the number of workers.
        train
            If set to True, the function will return the train set. If set to 
            False, the test set will be returned instead.
        train_set_size
            Set the ratio of total images included in the train set.
        IMRC
            If set to True, the datasets will include images from our flight 
            space acquired with our camera configuration. Further information
            on dataset acquisition is provided in the README. 
    """
    # load images and labels from the stored dataset
    [images, labels] = DataProcessor.ProcessTestData(path)

    # if training data should be extended by our custom dataset, set IMRC to True
    if IMRC:
        import pickle
        with open(f"{project_root}/misc/IMRC_images.pickle", "rb") as f:
            imrc_data = pickle.load(f)

        imrc_images = imrc_data['x']
        imrc_labels = imrc_data['y']

        images = np.concatenate([images, imrc_images])
        labels = np.concatenate([labels, imrc_labels])

    rng = np.random.default_rng(1749)

    indices = np.arange(len(images))
    rng.shuffle(indices)
    split_idx = int(len(images) * train_set_size)

    if train:
        # create a torch dataset from the loaded data
        dataset = Dataset(images[indices[:split_idx]], labels[indices[:split_idx]])
    else:
        dataset = Dataset(images[indices[split_idx:]], labels[split_idx:])

    # for quick and convinient access, create a torch DataLoader with the given parameters
    data_params = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last':drop_last, 'num_workers': 0, 'pin_memory': False}
    data_loader = data.DataLoader(dataset, **data_params)
    
    return data_loader


def calc_saliency(img, gt, model):
    input = img.requires_grad_(True)
    prediction = torch.stack(model(input.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

    loss_x = torch.nn.L1Loss()(prediction[0], gt[0])
    loss_y = torch.nn.L1Loss()(prediction[1], gt[1])
    loss_z = torch.nn.L1Loss()(prediction[2], gt[2])
    loss_phi = torch.nn.L1Loss()(prediction[3], gt[3])

    loss = loss_x + loss_y + loss_z + loss_phi

    loss.backward()

    saliency = input.grad.data.abs()

    return saliency

def plot_saliency(img, gt, model):
    saliency = calc_saliency(img, gt, model)
    img = img[0][0].detach().cpu().numpy()
    saliency = saliency[0][0].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(8, 2))
    ax[0].imshow(img, cmap='gray')
    ax[1].set_title('Saliency Map')
    ax[1].imshow(saliency, cmap='hot')
    ax[2].set_title('Superimposed')
    ax[2].imshow(img + (200000*saliency), cmap='gray')

    return fig

def inverse_norm(val, minimum, maximum):
    return np.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)

def get_transformation(sf, tx, ty):
    translation_vector = torch.stack([tx, ty]).unsqueeze(0) #torch.zeros([1], device=tx.device)]).unsqueeze(0)

    eye = torch.eye(2, 2).unsqueeze(0).to(sf.device)
    scale = eye * sf

    # print(scale.shape, translation_vector.shape)

    transformation_matrix = torch.cat([scale, translation_vector], dim=2)
    return transformation_matrix.float()

def norm_transformation(sf, tx, ty):
    tx_tanh = torch.tanh(tx)
    ty_tanh = torch.tanh(ty)
    scaling_norm = 0.1 * (torch.tanh(sf) + 1) + 0.3 # normalizes scaling factor to range [0.3, 0.5]

    return scaling_norm, tx_tanh, ty_tanh


def gen_noisy_transformations(batch_size, sf, tx, ty):
    noisy_transformation_matrix = []
    for i in range(batch_size):
        sf_n = sf + np.random.normal(0.0, 0.1)
        tx_n = tx + np.random.normal(0.0, 0.1)
        ty_n = ty + np.random.normal(0.0, 0.1)

        scale_norm, tx_norm, ty_norm = norm_transformation(sf_n, tx_n, ty_n)
        matrix = get_transformation(scale_norm, tx_norm, ty_norm)

        # random_yaw = np.deg2rad(np.random.normal(-10, 10))
        # random_pitch = np.deg2rad(np.random.normal(-5, 5))
        # random_roll = np.deg2rad(np.random.normal(-5, 5))
        #noisy_rotation = torch.tensor(get_rotation(random_yaw, random_pitch, random_roll)).float().to(matrix.device)

        #matrix[..., :3, :3] = noisy_rotation @ matrix[..., :3, :3]

        noisy_transformation_matrix.append(matrix)
    
    return torch.cat(noisy_transformation_matrix)

def load_patch(path, mode):
    patches = []
    for file_p in sorted(glob.glob(str(path) + '/' + str(mode)+ '*/' + 'patches.npy')):
        patches.append(np.load(file_p)[-1])

    return np.array(patches)

def load_position(path, mode):
    positions = []
    for file_p in sorted(glob.glob(str(path) + '/' + str(mode)+ '*/' + 'positions_norm.npy')):
        positions.append(np.load(file_p))

    positions = np.rollaxis(np.array(positions), 2, 0)[-1]
    positions = np.rollaxis(positions, 2, 1)
    positions = np.rollaxis(positions, 1, 0)

    return positions


# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q


# def plot_patch(patch, image, title='Plot', save=False, path='./'):

#     img_min, img_max = patch.batch_place(image)

#     f = plt.figure(constrained_layout=True, figsize=(10, 4))
#     subfigs = f.subfigures(1, 2, width_ratios=[1, 3])
#     fig_patch = subfigs[0].subplots(1,1)
#     fig_patch.imshow(patch.patch[0][0].detach().cpu().numpy(), cmap='gray')
#     subfigs[0].suptitle('Patch', fontsize='x-large')

#     subfigs[1].suptitle('placed', fontsize='x-large')
#     fig_placed = subfigs[1].subplots(1,2)
#     fig_placed[0].imshow(img_min[0][0].detach().cpu().numpy(), cmap='gray')
#     fig_placed[0].set_title('min direction')
#     fig_placed[1].imshow(img_max[0][0].detach().cpu().numpy(), cmap='gray')
#     fig_placed[1].set_title('max direction')

#     f.suptitle(title, fontsize='xx-large')
    
#     if save:
#         plt.savefig(path+title+'.jpg', transparent=False)
#         plt.close()
#     else: 
#         return f


