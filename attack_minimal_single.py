import torch
import torch.nn.functional as F
import numpy as np

import os

from tqdm import trange

import time

from util import load_dataset

from camera import Camera
from pathlib import Path
import matplotlib.pyplot as plt

import pickle
from scipy.optimize import linprog

from cv2 import findHomography

def normalize_yaw_t(yaw):
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))

def normalize_yaw(yaw):
    return np.atan2(np.sin(yaw), np.cos(yaw))

def dist(c1, c2):
        elementwise = torch.square(c1 - c2)
        # elementwise * torch.tensor([1, 1, 2, 2/.4, 2, 2])
        return torch.sqrt(torch.sum(elementwise, axis=-1))

def solve_quadratic(a, b, c):
    """
    Solves ax^2 + bx + c = 0, returning two real roots.
    Returns (None, None) if no real roots exist.
    """
    # Handle degenerate case (a=0 -> linear equation)
    if np.abs(a) < 1e-9:
        if np.abs(b) < 1e-9:
            return (None, None) # No solution
        root = -c / b
        return (root, root)
        
    discriminant = b**2 - 4*a*c
    
    if discriminant < -1e-9: # Allow for small float errors
        # No real roots
        return (None, None)
    
    # Ensure discriminant is non-negative
    sqrt_discriminant = np.sqrt(max(0.0, discriminant))
    
    # Calculate roots
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)
    
    return (root1, root2)

def bb_from_xyz(camera_intrinsic, camera_extrinsic, new_xyz, radius):
    """
    Calculates the 2D bounding box silhouette of a sphere
    centered at new_xyz (in drone frame) with a given RADIUS.
    
    This is the inverse of the xyz_from_bb function.
    """
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    ox = camera_intrinsic[0, 2]
    oy = camera_intrinsic[1, 2]

    # === Step 1: Transform from World Coords to Camera Coords ===
    # camera_extrinsic is T_c_w (World-to-Camera)
    new_xyz_h = np.array([*new_xyz, 1.0])
    xyz_h = camera_extrinsic @ new_xyz_h
    xyz = xyz_h[:3] # 3D point (sphere center) in camera coordinates

    # === Step 2: Get Center Ray (ac) and Distance ===
    distance = np.linalg.norm(xyz)
    
    # Object is behind the camera (z is negative or zero)
    if xyz[2] <= 1e-6:
        print("Error: Object is behind or inside the camera.")
        return None

    # Camera is inside the sphere, silhouette is not defined
    if distance < radius:
        print(f"Error: Camera is inside the object (distance {distance} < radius {radius}).")
        return None

    # 'ac' is the ray to the center, projected onto the z=1 plane
    xc_cam = xyz[0] / xyz[2]
    yc_cam = xyz[1] / xyz[2]
    ac = np.array([xc_cam, yc_cam, 1.0])
    norm_ac_sq = np.dot(ac, ac) # norm(ac)^2

    # === Step 3: Find Tangent Angle ===
    # From geometry: sin(theta/2) = RADIUS / distance
    # We need cos^2(theta/2) = 1 - sin^2(theta/2)
    # This is the squared cosine of the angle between the center ray (ac)
    # and any tangent ray (a_tangent).
    cos_half_theta_sq = 1.0 - (radius / distance)**2
    
    # This is the common 'A' term for our quadratic solvers
    A = norm_ac_sq * cos_half_theta_sq

    # === Step 4: Solve for Horizontal Tangents (x1_cam, x2_cam) ===
    # We solve a quadratic equation for x_cam, given yc_cam
    # (A - xc_cam^2) * x^2 - (2*K1*xc_cam) * x + (A*K1 - K1^2) = 0
    # where K1 = yc_cam^2 + 1.0
    
    K1_x = yc_cam**2 + 1.0
    a_x = A - xc_cam**2
    b_x = -2 * K1_x * xc_cam
    c_x = K1_x * (A - K1_x)
    
    x1_cam, x2_cam = solve_quadratic(a_x, b_x, c_x)
    
    if x1_cam is None:
        print("Error: Could not find horizontal tangent rays.")
        return None

    # === Step 5: Solve for Vertical Tangents (y1_cam, y2_cam) ===
    # We solve a symmetric quadratic equation for y_cam, given xc_cam
    # (A - yc_cam^2) * y^2 - (2*K2*yc_cam) * y + (A*K2 - K2^2) = 0
    # where K2 = xc_cam^2 + 1.0
    
    K1_y = xc_cam**2 + 1.0
    a_y = A - yc_cam**2
    b_y = -2 * K1_y * yc_cam
    c_y = K1_y * (A - K1_y)
    
    y1_cam, y2_cam = solve_quadratic(a_y, b_y, c_y)
    
    if y1_cam is None:
        print("Error: Could not find vertical tangent rays.")
        return None

    # === Step 6: Convert Rays back to Pixel Coordinates ===
    px_1 = x1_cam * fx + ox
    px_2 = x2_cam * fx + ox
    
    py_1 = y1_cam * fy + oy
    py_2 = y2_cam * fy + oy

    # === Step 7: Construct Bounding Box ===
    bb = np.array([
        min(px_1, px_2), # bb[0] = x_min
        min(py_1, py_2), # bb[1] = y_min
        max(px_1, px_2), # bb[2] = x_max
        max(py_1, py_2)  # bb[3] = y_max
    ])
    
    return bb

def _perspective_grid(
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
    
def project_patch(patches, T_matrices, images):
    """ Project the patches on the batch of camera images.
    Args:
        patches: Tensor of shape [B, C, H_p, W_p] (batch of patches).
        T_matrices: Tensor of shape [B, 3, 3] (batch of transformation matrices).
        images: Tensor of shape [B, C, H_i, W_i] (batch of images).
    Returns:
        Tensor of manipulated images of shape [B, C, H_i, W_i].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patches = patches.to(device)
    T_matrices = T_matrices.to(device)
    images = images.to(device)

    batch_size, _, p_height, p_width = patches.shape
    # _, _, i_height, i_width = images.shape

    i_height, i_width = images.shape[-2::]
    while len(images.shape) < 4:
        images = images.unsqueeze(0)

    # Create masks for patches
    masks = torch.ones_like(patches, dtype=torch.float32, device=device)

    # print(T_matrices[0])

    # Invert transformation matrices
    inv_T_matrices = torch.inverse(T_matrices)

    # Flatten transformation matrices for grid computation
    # print(inv_T_matrices.shape)
    coeffs = inv_T_matrices.reshape(batch_size, -1)
    # print(coeffs.shape)

    # Generate perspective grids for batch
    grids = _perspective_grid(
        coeffs, w=p_width, h=p_height, ow=i_width, oh=i_height, 
        dtype=torch.float32, device=device, center=[1., 1.]
    )

    # Apply grid sampling for patches and masks
    transformed_patches = torch.nn.functional.grid_sample(
        patches, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    )
    bit_masks = torch.nn.functional.grid_sample(
        masks, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    ).bool()

    # Combine transformed patches with original images
    manipulated_images = images * ~bit_masks
    manipulated_images += transformed_patches

    return manipulated_images

def norm_transformation(sf, tx, ty, scale_min, scale_max, tx_min, tx_max, ty_min, ty_max):
    # new patch placement implementation might need different tx, ty limits!:
    #sf_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min
    #tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    #ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min

    sf_norm = single_norm(sf, scale_min, scale_max)
    tx_norm = single_norm(tx, tx_min, tx_max)
    ty_norm = single_norm(ty, ty_min, ty_max)



    return sf_norm, tx_norm, ty_norm

def noisy_transformations(sf, tx, ty):
    sf_n = sf + np.random.normal(0.0, 0.1)
    tx_n = tx + np.random.normal(0.0, 0.1)
    ty_n = ty + np.random.normal(0.0, 0.1)
    return sf_n, tx_n, ty_n

def construct_T_matrix(sf, tx, ty, scale_min=0.2, scale_max=0.575, tx_min=48., tx_max=128., ty_min=20., ty_max=66., noise=True):
    # print("Inside construct T:")
    # print("sf:", sf)
    # print("tx:", tx)
    # print("ty:", ty)
    # print("tx_min, tx_max, ty_min, ty_max:", tx_min, tx_max, ty_min, ty_max)

    if noise:
        sf, tx, ty = noisy_transformations(sf, tx, ty)
    # noisy_sf, noisy_tx, noisy_ty = noisy_transformations(sf, tx, ty)
    # print("ty_min, ty_max:", ty_min, ty_max)
    # print("tx_min, tx_max:", tx_min, tx_max)
    norm_scale, norm_tx, norm_ty = norm_transformation(sf=sf, tx=tx, ty=ty, scale_min=scale_min, scale_max=scale_max, tx_min=tx_min, tx_max=tx_max, ty_min=ty_min, ty_max=ty_max)
    # print(norm_tx, norm_ty)

    T_matrix = torch.zeros((3, 3), device=sf.device)
    scale_T = torch.eye(2, device=sf.device) * norm_scale
    T_matrix[:2, :2] = scale_T
    T_matrix[0, 2] = norm_tx
    T_matrix[1, 2] = norm_ty
    T_matrix[2, 2] = 1.0
    # print(T_matrix)
    return T_matrix

def single_norm(value, min_value, max_value):
    norm_value = (max_value - min_value) * (torch.tanh(value) + 1) * 0.5 + min_value
    return norm_value

def inverse_norm(norm_value, min_value, max_value):
    inverse_norm = torch.atanh(2 * (norm_value - min_value) / (max_value - min_value) - 1)
    return inverse_norm


def calc_heading_vec(radius, angle):
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    zero = torch.zeros_like(x, device=angle.device, dtype=torch.float32)
    return torch.stack((x, y, zero), dim=-1)


def T_matrix(pose):
    T = torch.eye(4, dtype=torch.float32).to(pose.device)
    normalized_yaw = normalize_yaw_t(pose[3])

    sin_yaw = torch.sin(normalized_yaw).to(torch.float32)
    cos_yaw = torch.cos(normalized_yaw).to(torch.float32)
    zero = torch.zeros_like(sin_yaw, device=pose.device, dtype=torch.float32)

    rotation_matrix_row1 = torch.stack([cos_yaw, -sin_yaw, zero], dim=-1)
    rotation_matrix_row2 = torch.stack([sin_yaw, cos_yaw, zero], dim=-1)
    rotation_matrix_row3 = torch.tensor([0., 0., 1.], device=pose.device, dtype=torch.float32)
    R = torch.stack((rotation_matrix_row1, rotation_matrix_row2, rotation_matrix_row3), dim=0)

    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T


def calc_monitor_corners(drone_pose, projector_world, camera_extrinsic, camera_intrinsic):
    """Calculate the corners of the monitor in the image space."""
    one = torch.tensor(1.0, device=drone_pose.device, dtype=torch.float32)
    projector_in_drone = torch.stack([torch.inverse(drone_pose) @ torch.stack([*projector_coords, one], dim=-1) for projector_coords in projector_world])
    projector_camera = torch.stack([(camera_extrinsic @ patch_coords)[:3] for patch_coords in projector_in_drone])
    projector_image = torch.stack([(camera_intrinsic @ patch_coords) for patch_coords in projector_camera])  

    projector_image_ul, projector_image_ur, projector_image_ll, projector_image_lr = projector_image
    projector_image_ul = torch.stack([projector_image_ul[0] / projector_image_ul[2], projector_image_ul[1] / projector_image_ul[2]])
    projector_image_ur = torch.stack([projector_image_ur[0] / projector_image_ur[2], projector_image_ur[1] / projector_image_ur[2]])
    projector_image_ll = torch.stack([projector_image_ll[0] / projector_image_ll[2], projector_image_ll[1] / projector_image_ll[2]])
    projector_image_lr = torch.stack([projector_image_lr[0] / projector_image_lr[2], projector_image_lr[1] / projector_image_lr[2]])
    
    return torch.stack([projector_image_ul, projector_image_ur, 
                        projector_image_ll, projector_image_lr], dim=0)


def gen_target_trajectory(trajectory):
    if trajectory == 'square':

    # # Square corners (x, y) without z and yaw for now
        corners = np.array([
        [0., 1.],
        [-1., 1.],
        [-1., -1.],
        [0., -1.],
        [0., 1.] # End at the starting point
        ])

        edges = list(zip(corners[:-1], corners[1:]))
        n_edges = len(edges)
    
        # Reserve 1 point per corner, distribute the rest
        extra_points = 20 - n_edges  
        base = extra_points // n_edges
        remainder = extra_points % n_edges
        
        points = []
        for i, (start, end) in enumerate(edges):
            # Number of points on this edge (including the corner at 'end')
            num_on_edge = base + (1 if i < remainder else 0) + 1
            
            # Interpolate along the edge
            xs = np.linspace(start[0], end[0], num_on_edge, endpoint=False)
            ys = np.linspace(start[1], end[1], num_on_edge, endpoint=False)
            edge_points = np.column_stack([xs, ys])
            
            points.extend(edge_points)

        points = np.array(points)

        z = np.ones((points.shape[0],))  # Create z with shape (20,)
        yaw = np.zeros((points.shape[0],))  # Create yaw with shape (20,)
        waypoints = np.hstack((points, z[:, None], yaw[:, None]))  # Stack points, z, and yaw to get shape (20, 4)
        # print(waypoints.shape)

        return torch.tensor(waypoints, dtype=torch.float32)


    if trajectory == 'circle':
        t = np.linspace(0, 2 * np.pi, 20)
        x = 0.5 * np.cos(t)
        y = 0.5 * np.sin(t)
        z = np.ones_like(t)  # Constant height at 1
        yaw = np.zeros_like(t)  # Constant yaw
        target_trajectory = np.column_stack((x, y, z, yaw))
        return torch.tensor(target_trajectory, dtype=torch.float32)

    if trajectory == 'line_x':
        points = np.array([0., 0.5, 0., -1., 0.])

        # Compute cumulative distances along the path
        distances = np.cumsum(np.abs(np.diff(points)))
        distances = np.insert(distances, 0, 0)  # start at 0

        # Generate 20 evenly spaced distances
        even_distances = np.linspace(0, distances[-1], 20)

        # Interpolate to get evenly spaced points
        x = np.interp(even_distances, distances, points)
        y = np.zeros_like(x)
        z = np.ones_like(x)  # Constant height at 1
        yaw = np.zeros_like(x)  # Constant yaw
        target_trajectory = np.column_stack((x, y, z, yaw))
        return torch.tensor(target_trajectory, dtype=torch.float32)

    if trajectory == 'line_y':
        points = np.array([0., 1.0, 0., -1.0, 0.])

        # Compute cumulative distances along the path
        distances = np.cumsum(np.abs(np.diff(points)))
        distances = np.insert(distances, 0, 0)  # start at 0

        # Generate 20 evenly spaced distances
        even_distances = np.linspace(0, distances[-1], 20)

        # Interpolate to get evenly spaced points
        y = np.interp(even_distances, distances, points)
        x = np.zeros_like(y)
        z = np.ones_like(y)  # Constant height at 1
        yaw = np.zeros_like(y)  # Constant yaw
        target_trajectory = np.column_stack((x, y, z, yaw))
        return torch.tensor(target_trajectory, dtype=torch.float32)


    if trajectory == 'figure8':
        t = np.linspace(0, 2 * np.pi, 20)
        x = 0.2 * np.sin(2 * t)  # Horizontal figure 8
        y = 0.6 * np.sin(t)  # Vertical figure 8
        z = np.ones_like(t)  # Constant height at 1
        yaw = np.zeros_like(t)  # Constant yaw
        target_trajectory = np.column_stack((x, y, z, yaw))
        return torch.tensor(target_trajectory, dtype=torch.float32)

    if trajectory == 'diagonal_line':
        points_x = np.array([0., 0.6, 0., -0.6, 0.])
        points_y = np.array([0., 0.6, 0., -0.6, 0.])

        # Compute cumulative distances along the path
        distances = np.cumsum(np.sqrt(np.diff(points_x)**2 + np.diff(points_y)**2))
        distances = np.insert(distances, 0, 0)  # start at 0

        # Generate 20 evenly spaced distances
        even_distances = np.linspace(0, distances[-1], 20)

        # Interpolate to get evenly spaced points
        x = np.interp(even_distances, distances, points_x)
        y = np.interp(even_distances, distances, points_y)
        z = np.ones_like(x)  # Constant height at 1
        yaw = np.zeros_like(x)  # Constant yaw
        target_trajectory = np.column_stack((x, y, z, yaw))
        return torch.tensor(target_trajectory, dtype=torch.float32)
    if trajectory == 'triangle':
        corners = np.array([
            [0.6, 0.],
            [0., 0.5],
            [0., -0.5],
            [0.6, 0.] 
        ])

        edges = list(zip(corners[:-1], corners[1:]))
        n_edges = len(edges)

        # Reserve 1 point per corner, distribute the rest
        extra_points = 18 - n_edges  
        base = extra_points // n_edges
        remainder = extra_points % n_edges

        points = []
        for i, (start, end) in enumerate(edges):
            # Number of points on this edge (including the corner at 'end')
            num_on_edge = base + (1 if i < remainder else 0) + 1

            # Interpolate along the edge
            xs = np.linspace(start[0], end[0], num_on_edge, endpoint=False)
            ys = np.linspace(start[1], end[1], num_on_edge, endpoint=False)
            edge_points = np.column_stack([xs, ys])

            points.extend(edge_points)

        points = np.array(points)

        z = np.ones((points.shape[0],))  # Create z with shape (20,)
        yaw = np.zeros((points.shape[0],))  # Create yaw with shape (20,)
        waypoints = np.hstack((points, z[:, None], yaw[:, None]))  # Stack points, z, and yaw to get shape (20, 4)
        # add initial and final point to complete the triangle
        init_pose = np.array([[0., 0., 1., 0.]])
        waypoints = np.vstack((init_pose, waypoints, init_pose))

        return torch.tensor(waypoints, dtype=torch.float32)
    else:
        raise ValueError("Unknown trajectory type")
    

def get_patch_T(monitor_corners_world):
    device = monitor_corners_world.device
    monitor_width_up = monitor_corners[1, 0] - monitor_corners[0, 0]
    monitor_width_down = monitor_corners[3, 0] - monitor_corners[2, 0]
    monitor_width = min(monitor_width_up, monitor_width_down)


    monitor_height_left = monitor_corners[2, 1] - monitor_corners[0, 1]
    monitor_height_right = monitor_corners[3, 1] - monitor_corners[1, 1]
    monitor_height = min(monitor_height_left, monitor_height_right)

    tx_min, ty_min = monitor_corners[0, 0], monitor_corners[0, 1]

    scale_width = monitor_width / 80.
    scale_height = monitor_height / 45.

    max_scale = min(scale_width, scale_height)
    
    T = torch.eye(3, device=device, dtype=torch.float32)  # Identity transformation matrix
    T[:2, :2] *= max_scale  # Scale down to half the size of the patch
    T[0, 2] = tx_min
    T[1, 2] = ty_min

    return T

class PController:
    def __init__(self, max_vel=1.0, max_yaw_rate=0.5, kp_pos=1.0, kp_yaw=2.0):
        """
        Args:
            max_vel: Maximum velocity in m/s
            max_yaw_rate: Maximum yaw rate in rad/s
            kp_pos: Proportional gain for position (higher = snappier)
            kp_yaw: Proportional gain for yaw
        """
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw

    def normalize_angle(self, angle):
        """Wraps angle to [-pi, pi] to find shortest rotation path."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, current_state: torch.tensor, target_state: torch.tensor, dt: torch.float32):
        """
        Moves the drone towards target for duration dt.
        
        Args:
            current_state: dict or list [x, y, z, yaw]
            target_state: dict or list [x, y, z, yaw]
            dt: time step in seconds
            
        Returns:
            new_state: [x, y, z, yaw]
            cmd_vel: The velocity command used [vx, vy, vz, yaw_rate] (for logging)
        """
        # Unpack states (assuming list format [x, y, z, yaw])
        curr_pos = current_state[:3]
        curr_yaw = current_state[3]
        
        targ_pos = target_state[:3]
        targ_yaw = target_state[3]

        # --- 1. Position Control ---
        # Calculate error vector
        pos_error = targ_pos - curr_pos
        
        # Calculate desired velocity (P-controller)
        vel_cmd = pos_error * self.kp_pos
        
        # Clip velocity to max speed (maintain direction)
        speed = torch.norm(vel_cmd)
        if speed > self.max_vel:
            vel_cmd = (vel_cmd / speed) * self.max_vel

        # --- 2. Yaw Control ---
        # Calculate yaw error (shortest path)
        yaw_error = normalize_yaw_t(targ_yaw - curr_yaw)
        
        # Calculate desired yaw rate
        yaw_rate_cmd = yaw_error * self.kp_yaw
        
        # Clip yaw rate
        yaw_rate_cmd = torch.clamp(yaw_rate_cmd, -self.max_yaw_rate, self.max_yaw_rate)

        # --- 3. Integration (Simulation Step) ---
        # Update position: x_new = x_old + v * dt
        new_pos = curr_pos + vel_cmd * dt
        
        # Update yaw: yaw_new = yaw_old + rate * dt
        new_yaw = normalize_yaw_t(curr_yaw + yaw_rate_cmd * dt)
        
        # Pack result
        new_state = torch.cat((new_pos, new_yaw.unsqueeze(0)))
        cmd_vel = torch.cat((vel_cmd, yaw_rate_cmd.unsqueeze(0)))
        
        return new_state, cmd_vel

def compute_dtw_distance(target_traj: torch.tensor, actual_traj: torch.tensor):
    """
    Computes the Dynamic Time Warping (DTW) distance between two 3D trajectories.
    
    Args:
        target_traj: np.array or tensor of shape (N, 3)
        actual_traj: np.array or tensor of shape (M, 3)
        
    Returns:
        float: The normalized DTW distance (average distance per point alignment)
    """
    # # Ensure inputs are numpy arrays
    # if hasattr(target_traj, 'cpu'): target_traj = target_traj.cpu().numpy()
    # if hasattr(actual_traj, 'cpu'): actual_traj = actual_traj.cpu().numpy()
    
    # 1. Compute the pairwise distance matrix (Euclidean)
    # Shape: (N, M)
    dists = torch.cdist(target_traj, actual_traj, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    
    # 2. Initialize the cumulative cost matrix
    N, M = dists.shape
    dtw_matrix = torch.zeros((N, M), device=dists.device, dtype=dists.dtype)
    dtw_matrix[0, 0] = dists[0, 0]
    
    # 3. Fill first row and first column
    for i in range(1, N):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + dists[i, 0]
    for j in range(1, M):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + dists[0, j]
        
    # 4. Fill the rest using the recurrence relation
    # Cost = current_dist + min(left, up, diagonal)
    for i in range(1, N):
        for j in range(1, M):
            min_prev = torch.min(torch.stack([
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            ]))
            dtw_matrix[i, j] = dists[i, j] + min_prev
            
    # 5. Return normalized distance (optional but recommended for comparison)
    # The value at dtw_matrix[-1, -1] is the total accumulated cost.
    # Normalizing by path length (N+M) makes it interpretable as "avg error in meters"
    return dtw_matrix[-1, -1] / (N + M)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet', help='Model to use for prediction')
    parser.add_argument('-t', '--trajectory', type=str, choices=['figure8', 'square', 'circle', 'line_x', 'line_y', 'diagonal_line', 'triangle'], default='figure8', help='Target Trajectory')
    parser.add_argument('--display_size', type=int, default=60, help='Size of the display in pixels (default: 60")')
    parser.add_argument('--patch_mode', type=str, choices=['optimal', 'velo', 'timeout', 'black', 'white', 'random', 'fap', 'diffusion', 'interpolation', 'corpus'], default='optimal', help='Mode to initialize the patch: optimal, timeout, black, white, random')
    parser.add_argument('--temperature', type=str, choices=['warm', 'cold', 'none'], default='cold', help='Either restart from random patch (cold) or from the last patch (warm)')
    parser.add_argument('--pic_mode', type=str, choices=['random', 'idx'], default='idx', help='Mode to select image: random or specific index')
    parser.add_argument('--img_idx', type=int, default=0, help='Index of the image to use from the dataset')
    parser.add_argument('--corpus_size', type=int, choices=[1000, 2000, 3000], default=1000, help='Number of patches in the corpus (only for corpus/interpolation/diffusion patch mode)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout for optimization step in Hz (default: None for no timeout, otherwise int Hz)')
    parser.add_argument('--ghost_mode', action='store_true', default=False, help='Enable ghost mode (yolov5 only)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.time()

    # img_idx = args.img_idx

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

    model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'frontnet':
        print('Loading Frontnet model...')
        from util import load_model
        model = load_model(f"{project_root}/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
        model.eval()
    elif model_name == 'yolov5':
        print('Loading YOLOv5 model...')
        from yolo_bounding import YOLOBox
        model = YOLOBox()


    dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = 1, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)

    print(len(dataset))

    patch_mode = args.patch_mode
    directory = f'{model_name}/{patch_mode}'

    projector_size = args.display_size

    if patch_mode =='timeout' and (isinstance(args.timeout, int) or isinstance(args.timeout, float)):
        timeout = 1 / args.timeout  # seconds
        directory = Path(f'{model_name}/timeout_{args.timeout}Hz')
    else:
        timeout = None

    if model_name == 'yolov5' and args.ghost_mode:
        directory = f'{directory}_ghost'

    if args.patch_mode == 'interpolation' or args.patch_mode == 'corpus' or args.patch_mode == 'diffusion':
        directory = f'{directory}/{args.corpus_size}'


    if args.pic_mode == 'random':
        if patch_mode == 'optimal' or patch_mode == 'timeout' or patch_mode == 'velo':
            directory = f'{directory}/{args.temperature}'
        output_dir = Path(f'{directory}') / args.trajectory / f'{args.display_size}z' / f'random' / f'{args.seed}'
    else:
        if patch_mode == 'optimal' or patch_mode == 'timeout' or patch_mode == 'velo':
            directory = f'{directory}/{args.temperature}'
        output_dir = Path(f'{directory}') / args.trajectory / f'{args.display_size}z' / f'image_{args.img_idx}' / f'{args.seed}'
    print(output_dir)

    os.makedirs(output_dir, exist_ok=True)


    if args.model == 'frontnet':
        initial_patch_size = (45, 80)  # Height, Width
    elif args.model == 'yolov5':
        initial_patch_size = (150, 320)  # Height, Width

    if patch_mode == 'velo' or patch_mode == 'timeout':
        controller = PController(max_vel=1.0, max_yaw_rate=0.5, kp_pos=1.0, kp_yaw=2.0)
        dt = 1/30
        if patch_mode == 'timeout':
            dt = timeout

    if args.patch_mode == 'fap':
        import yaml
        fap_patches = torch.tensor(np.load(f'{args.model}/fap/last_patch.npy')).to(device).unsqueeze(1)
        probabilities_per_patch = np.load(f'{args.model}/fap/stats_p.npy')[-1]

        # print("FAP probabilities per patch:", probabilities_per_patch)

        assignment = {'forward': None, 'backward': None, 'stay': None, 'left': None, 'right': None}

         # SETTINGS
        with open(f'{args.model}/fap/settings.yaml') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

        optim_targets = [values for _, values in settings['targets'].items()]
        optim_targets = np.array(optim_targets, dtype=float).T

        for i, target in enumerate(optim_targets):
            if np.array_equal(target, np.array([1., 0., 0.])):
                assignment['stay'] = np.argmax(probabilities_per_patch[:, i])
            elif np.array_equal(target, np.array([1.5, 0., 0.])):
                assignment['forward'] = np.argmax(probabilities_per_patch[:, i])
            elif np.array_equal(target, np.array([0.5, 0., 0.])):
                assignment['backward'] = np.argmax(probabilities_per_patch[:, i])
            elif np.array_equal(target, np.array([1., 1., 0.])):
                assignment['left'] = np.argmax(probabilities_per_patch[:, i])
            elif np.array_equal(target, np.array([1., -1., 0.])):
                assignment['right'] = np.argmax(probabilities_per_patch[:, i])
            else:
                print("Unknown target:", target)
        # print("FAP assignment:", assignment)

    if args.patch_mode == 'diffusion':
        from diffusion.diffusion_model import DiffusionModel
        diffusion_model = DiffusionModel(device=device)

        diffusion_model.load(f'results/diffusion_training/{model_name}/{100}/frontnet100test.pth')


    # # tests to improve yolo
    # with open(f"diffusion/yolo100.pickle", "rb") as f:
    #     patch_dataset = pickle.load(f)

    #     corpus_patches = []
    #     for idx_corpus in range(len(patch_dataset)):
    #         corpus_patches.append(patch_dataset[idx_corpus][0])

    # corpus_patches = np.array(corpus_patches) # shape (N, 45, 80)
    # random_start_patch = corpus_patches[np.random.randint(0, len(corpus_patches))]

    if args.patch_mode == 'interpolation' or args.patch_mode == 'corpus':
        with open(f"diffusion/{model_name}{args.corpus_size//1000}k.pickle", "rb") as f:
            patch_dataset = pickle.load(f)

        corpus_patches = []
        corpus_targets = []
        corpus_positions = []
        for i in range(args.corpus_size):
            corpus_patches.append(patch_dataset[i][0])
            corpus_targets.append(patch_dataset[i][1])
            corpus_positions.append(patch_dataset[i][2])


        corpus_patches = np.array(corpus_patches) # shape (N, 45, 80)
        corpus_targets = np.array(corpus_targets) # shape (N, 1, 4) -> x, y, z, yaw
        corpus_positions = np.array(corpus_positions) # shape (N, 1, 3), sf in range [0.4, 0.8], tx, ty in range [0, 1]

        corpus_patches = np.array([(patch - np.min(patch)) / (np.max(patch) - np.min(patch)) for patch in corpus_patches]) # normalize
        # keep patches as (N, H, W) to match sim_diffusion_attack InterpolatedPatchThread
        corpus_patches = torch.tensor(corpus_patches, device=device, dtype=torch.float32)
        corpus_targets = torch.tensor(corpus_targets, device=device, dtype=torch.float32).squeeze(1)
        corpus_positions = torch.tensor(corpus_positions, device=device, dtype=torch.float32).squeeze(1)

        conditioning_gt = torch.cat((corpus_positions, corpus_targets), dim=1)


    # img = torch.ones((1, 1, 96, 160), device=device, dtype=torch.float32) * 0.5  # gray image
    # img_idx = np.random.randint(0, len(dataset))
    # img = dataset.dataset[img_idx][0].to(device).unsqueeze(0) / 255.0
    # print(img.shape)

    cam = Camera('camera_calibration.yaml', device=device)
    camera_intrinsic = cam.camera_intrinsic_tens
    camera_extrinsic = cam.camera_extrinsic_tens
    radius = cam.radius

    all_drone_poses = []

    drone_pose = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)  # Initial pose
    current_velocity = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)  # Initial velocity


    T_drone_in_world = torch.eye(4, device=device, dtype=torch.float32)
    T_drone_in_world[:3, 3] = drone_pose[:3]

    all_drone_poses.append([*T_drone_in_world[:3, 3].clone().detach().cpu().numpy(), 0.0])
    
    
    # t = np.linspace(0, 2 * np.pi, 20)
    # x = 0.5 * np.sin(2 * t)  # Horizontal figure 8
    # y = 1.5 * np.sin(t)  # Vertical figure 8
    # z = np.ones_like(t)  # Constant height at 1
    # yaw = np.zeros_like(t)  # Constant yaw
    # target_trajectory = np.column_stack((x, y, z, yaw))
    # target_trajectory = torch.tensor(target_trajectory, dtype=torch.float32, device=device)

    target_trajectory = gen_target_trajectory(args.trajectory).to(device)

    # projector_world = torch.tensor([[2, 1, 2.2],   # ul
    #                             [2, -1.5, 2.2], #ur
    #                             [2, 1, 0.8], # ll
    #                             [2, -1.5, 0.8]], #lr
    #                             dtype=torch.float32, device=device) 

    projector_center = torch.tensor([2., 0.0, 1.], dtype=torch.float32, device=device)  # Center of the projector (x,y,z)
    projector_diagonal = projector_size * 0.0254 # 60" in m, projector diagonal

    projector_height = np.round(projector_diagonal / np.sqrt((16./9.)**2 + 1), 2)  # Height of the projector in m
    projector_width = np.round(projector_height * (16. / 9.), 2)  # Width of the projector in m


    projector_world = torch.tensor([[projector_center[0], projector_center[1] + projector_width / 2, projector_center[2] + projector_height / 2], # upper left corner
                                    [projector_center[0], projector_center[1] - projector_width / 2, projector_center[2] + projector_height / 2], # upper right corner
                                    [projector_center[0], projector_center[1] + projector_width / 2, projector_center[2] - projector_height / 2], # lower left corner
                                    [projector_center[0], projector_center[1] - projector_width / 2, projector_center[2] - projector_height / 2]], # lower right corner
                                    dtype=torch.float32, device=device)  # Projector corners in world coordinates

    print("Projector corners in world coordinates:")
    print(projector_world)                    



    image_bb = torch.tensor([0.0, 0.0, 160., 96.], dtype=torch.float32, device=device)


    # monitor_corners = np.array([[48., 20.],     # upper left corner (x, y, 1)
    #                             [128., 20.],    # upper right corner
    #                             [48., 66.],     # lower left corner
    #                             [128., 66.]])   # lower right corner

    loss = torch.inf
    distance = torch.inf

    img_idx = args.img_idx

    optim_steps = 0

    target_idx = 1
    retries = 0
    # max_retries = 10
    # given dt and target_trajectory, compute max_retries
    if (patch_mode == 'timeout' or patch_mode == 'velo') and dt is not None:
        avg_dist_per_step = 0.1  # assume average distance of 0.1m per step
        total_time = 0.0
        total_distance = 0.0
        for i in range(1, len(target_trajectory)):
            dist = torch.dist(target_trajectory[i-1][:3], target_trajectory[i][:3], p=2).item()
            total_distance += dist
        estimated_total_time = total_distance / avg_dist_per_step * dt
        max_retries = int(estimated_total_time / dt / len(target_trajectory)) * 2  # allow double the estimated time per target
        print(f"Estimated total time: {estimated_total_time:.2f}s, setting max_retries to {max_retries}")
    else:
        max_retries = 10  # default value

    while target_idx < len(target_trajectory) - 1:
        target = target_trajectory[target_idx]
        if patch_mode == 'velo' or patch_mode == 'timeout':
            cur_pos = torch.tensor(all_drone_poses[-1][:3]).to(device)
            # tgt_pos = torch.from_numpy(target[:3]).to(device)
            if torch.dist(cur_pos, target[:3], p=2) > 0.05 and retries <= max_retries:
                print("Current position:", cur_pos)
                print("Target position:", target[:3])
                print("Distance to target:", torch.dist(cur_pos, target[:3], p=2).item())
                # Move drone towards target using P-controller
                print("Target not reached, repeating target.")
        #         # do not increment target_idx -> repeat same target
        #         continue
                retries += 1
            else:
                print("Target reached, moving to next target.")
                target_idx += 1
                retries = 0
        elif distance < 0.05:
            target_idx += 1
            distance = torch.inf

        # retries = 0
        print(f"--- Target idx: {target_idx} ---")
        print(f"--- Optim steps so far: {optim_steps} ---")
        print("Current pose: ", T_drone_in_world[:3, 3].detach().cpu().numpy(), torch.atan2(T_drone_in_world[1,0], T_drone_in_world[0,0]).item())
        # process/advance to next target
        # target_idx += 1

        if args.pic_mode == 'random':
            img_idx = np.random.randint(0, len(dataset))
            

        img = dataset.dataset[img_idx][0].to(device).unsqueeze(0) / 255.0

        target = target_trajectory[target_idx]

        # if model_name == 'yolov5':
        #     T_setpoint_world = T_matrix(target)
        #     recovered_target_yaw = torch.atan2(T_setpoint_world[1,0], T_setpoint_world[0,0])
        #     T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
        #     T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(recovered_target_yaw - torch.pi)).to(device)
        #     # print("T_direction_world:")
        #     T_pred_in_world_recovered = torch.linalg.inv(T_direction_world) @ T_setpoint_world
        #     T_pred_in_drone_recovered = torch.linalg.inv(T_drone_in_world) @ T_pred_in_world_recovered

        #     print("Predicted relative position (Yolo):", T_pred_in_drone_recovered[:3, 3].detach().cpu().numpy())

        #     # recovered_yaw = torch.atan2(T_pred_in_drone_recovered[1,0], T_pred_in_drone_recovered[0,0])
        #     try:
        #         bb = bb_from_xyz(camera_intrinsic.detach().cpu().numpy(), camera_extrinsic.detach().cpu().numpy(), T_pred_in_drone_recovered[:3, 3].detach().cpu().numpy(), RADIUS)
        #         bb = torch.tensor(bb, dtype=torch.float32).to(device).unsqueeze(0)  # add batch dimension
        #         # scale to image of size 320 x 640
        #         bb[:, [0, 2]] *= (640.0 / 160.0)  # x coords
        #         bb[:, [1, 3]] *= (320.0 / 96.0)   # y coords
            
        #     except TypeError:
        #         print("Bounding box could not be computed, setting to bb detected by Yolo")
        #         with torch.no_grad():
        #             img_v = torch.nn.functional.interpolate(img, size=(320, 640), mode='bilinear', align_corners=False)
        #             img_v = torch.repeat_interleave(img_v, repeats=3, dim=1)  # to 3 channels
        #             bb = model(img_v).squeeze(1)
            


        monitor_corners = calc_monitor_corners(T_drone_in_world, projector_world, camera_extrinsic, camera_intrinsic)
        print("Monitor corners:")
        print(monitor_corners)

        if model_name == 'yolov5':
            # scale from 96x160 to 320x640
            monitor_corners[:, 0] *= (640.0 / 160.0)  # x coords
            monitor_corners[:, 1] *= (320.0 / 96.0)   # y coords


        # add homogeneous coordinate for homography
        monitor_corners_homogeneous = torch.tensor([[monitor_corners[0,0], monitor_corners[0,1], 1.],
                                        [monitor_corners[1,0], monitor_corners[1,1], 1.],
                                        [monitor_corners[2,0], monitor_corners[2,1], 1.],
                                        [monitor_corners[3,0], monitor_corners[3,1], 1.]], dtype=torch.float32, device=device)  
        
        # patch_coordinates = torch.stack([monitor_corners[0], monitor_corners[3]])
        # current_patch_size = (int((monitor_corners[3,1] - monitor_corners[0,1]).item()), int((monitor_corners[3,0] - monitor_corners[0,0]).item()))

        # patch_coordinates = torch.tensor([[0., 0., 1.],
        #                                 [patch_size[1], 0., 1.],
        #                                 [0., patch_size[0], 1.],
        #                                 [patch_size[1], patch_size[0], 1.]], dtype=torch.float32, device=device)


        patch_coordinates = torch.tensor([[0., 0., 1.],
                                          [initial_patch_size[1], 0., 1.],
                                          [0., initial_patch_size[0], 1.],
                                          [initial_patch_size[1], initial_patch_size[0], 1.]], dtype=torch.float32, device=device)
        
        # T = get_patch_T(monitor_corners)
        # print("Patch transformation T:")

        # print(T)

        T = findHomography(patch_coordinates.detach().cpu().numpy(), monitor_corners_homogeneous.detach().cpu().numpy(), method=1, maxIters=5000)[0]
        print("Patch transformation T:")
        print(T)
        T = torch.tensor(T, dtype=torch.float32, device=device)


        if patch_mode == 'black':
            patch = torch.zeros((1, 1, initial_patch_size[0], initial_patch_size[1]), device=device, dtype=torch.float32)
        elif patch_mode == 'white':
            patch = torch.ones((1, 1, initial_patch_size[0], initial_patch_size[1]), device=device, dtype=torch.float32)
        elif patch_mode == 'fap':
            # check if positive/negative change in x or y in target position is needed relative to current drone pose
            
            T_setpoint_world = T_matrix(target)
            setpoint_yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
            # Current drone pose in world
            T_drone_in_world = T_matrix(torch.tensor(all_drone_poses[-1], device=device, dtype=torch.float32))
            
            T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
            T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(setpoint_yaw - torch.pi)).to(device)


            T_pred_in_world = torch.inverse(T_direction_world) @ T_setpoint_world
            T_pred_in_drone = torch.inverse(T_drone_in_world) @ T_pred_in_world
            target_yaw = torch.atan2(T_pred_in_drone[1, 0], T_pred_in_drone[0, 0])
            
            relative_movement = T_pred_in_drone[:2, 3]
            # print("Debugging FAP selection:")
            # print("Relative movement:", relative_movement)

            max_idx = torch.argmax(torch.abs(relative_movement))
            # print("Max index:", max_idx)
            # print("Max value:", relative_movement[max_idx])

            if max_idx == 0:
                if relative_movement[max_idx] > 0:
                    # print("Loading forward patch")
                    patch = fap_patches[assignment['forward']]
                elif torch.abs(relative_movement[max_idx]) < 0.2:
                    # print("Loading stay patch")
                    patch = fap_patches[assignment['stay']]
                else:
                    # print("Loading backward patch")
                    patch = fap_patches[assignment['backward']]
            if max_idx == 1:
                if relative_movement[max_idx] > 0:
                    # print("Loading left patch")
                    patch = fap_patches[assignment['left']]
                else:
                    # print("Loading right patch")
                    patch = fap_patches[assignment['right']]
        elif patch_mode == 'diffusion' or patch_mode == 'interpolation' or patch_mode == 'corpus':
            # target_yaw = normalize_yaw_t(target[3])
            # Desired setpoint in world coordinates (the target pose)
            T_setpoint_world = T_matrix(target)
            setpoint_yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
            # Current drone pose in world
            T_drone_in_world = T_matrix(torch.tensor(all_drone_poses[-1], device=device, dtype=torch.float32))
            
            T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
            T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(setpoint_yaw - torch.pi)).to(device)


            T_pred_in_world = torch.inverse(T_direction_world) @ T_setpoint_world
            T_pred_in_drone = torch.inverse(T_drone_in_world) @ T_pred_in_world
            target_yaw = torch.atan2(T_pred_in_drone[1, 0], T_pred_in_drone[0, 0])

            sf = T[0, 0]
            tx = T[0, 2]
            ty = T[1, 2]

            if tx < 5e-2:
                tx = 5e-2
            if tx > (1 - 5e-2):
                tx = 1 - 5e-2

            conditioning = torch.tensor([sf, tx, ty, *T_pred_in_drone[:3, 3], target_yaw], dtype=torch.float32, device=device)
            
            if patch_mode == 'interpolation':
                # print("Debugging Interpolation Mode")
                distance = dist(conditioning[:6], conditioning_gt[:, :6])
                order = torch.argsort(distance)
                ordered_combined = conditioning_gt[order].detach().cpu().clone().numpy()
                # start from the closest single exemplar
                patch = corpus_patches[order[0]].clone()  # shape (H, W)
                # print("patch shape before loop: ", patch.shape)
                # print("Patch min/max before loop: ", patch.min(), patch.max())
                for n in range(1, len(order)):
                    result = linprog(
                        bounds=[(0,1)]*n,
                        c=np.ones(n),
                        A_eq=ordered_combined[:n].T,
                        b_eq=conditioning.detach().cpu().numpy(),
                    )
                    # print(n, result.success, result.fun)
                    if result.success and result.fun <= 1.:
                        # coeffs on device and float32
                        coeffs = torch.as_tensor(result.x, device=device, dtype=torch.float32)
                        # print(coeffs.sum())
                        # take the top-n exemplar patches (shape: n, H, W), permute to (H, W, n)
                        exemplars = corpus_patches[order][:n]  # (n, H, W)
                        patch = (exemplars.permute(1, 2, 0) * coeffs[None, None, :]).sum(dim=-1)  # (H, W)
                        break
                # print("Patch shape after loop: ", patch.shape, patch.min(), patch.max())
                # make patch compatible with project_patch API: (1, 1, H, W)
                patch = patch.unsqueeze(0).unsqueeze(0)
            
            elif patch_mode == 'corpus':
                # print("Conditioning:", conditioning[:6])
                distances = dist(conditioning[:6], conditioning_gt[:, :6])
                closest_idx = torch.argmin(distances)
                # print("Closest idx:", closest_idx.item())
                # print("Closest conditioning:", conditioning_gt[closest_idx][:6])
                patch = corpus_patches[closest_idx].unsqueeze(0).unsqueeze(0)  # make patch compatible with project_patch API: (1, 1, H, W)
                # print("Closest corpus idx:", closest_idx.item(), "Distance:", distances[closest_idx].item())
                # print("Patch min/max:", patch.min().item(), patch.max().item())
                # print("Patch shape:", patch.shape)
            
            elif patch_mode == 'diffusion':
                patch = diffusion_model.sample(1, conditioning, device, patch_size=initial_patch_size, n_steps=10)

        else:
            patch = torch.rand((1, 1, initial_patch_size[0], initial_patch_size[1]), device=device, dtype=torch.float32)  # random patch
            # patch = torch.tensor(random_start_patch, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # random patch from corpus
            # patch = torch.nn.functional.interpolate(patch.clone(), size=patch_size, mode='bilinear', align_corners=False)

        if patch_mode == 'optimal' or patch_mode == 'timeout' or patch_mode == 'velo':
            patch = patch.requires_grad_(True)


       

        # check = torch.stack([T @ row for row in patch_coordinates])
        # print("Patch coordinates after transformation:")
        # print(check)

        if patch_mode == 'optimal' or patch_mode == 'timeout' or patch_mode == 'velo':

            if args.temperature == 'warm' and target_idx > 2:
                patch = best_patch.clone().detach().requires_grad_(True)
                # patch = torch.nn.functional.interpolate(patch.clone(), size=patch_size, mode='bilinear', align_corners=False).requires_grad_(True)

            opt = torch.optim.Adam([patch], lr=3e-2)  # was at 3e-2
            scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, end_factor=1., total_iters=1000)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=1000)

            
            distance = torch.inf
            i = 0

            best_loss = torch.inf
            best_distance = torch.inf
            best_patch = patch.clone()
            
            best_setpoint = None

            time_start_optim_step = time.time()
            losses = []
            skip = False

            if model_name == 'yolov5' and args.ghost_mode:
                T_setpoint_world = T_matrix(target)
                setpoint_yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
                # Current drone pose in world
                T_drone_in_world = T_matrix(torch.tensor(all_drone_poses[-1], device=device, dtype=torch.float32))
                
                T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
                T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(setpoint_yaw - torch.pi)).to(device)


                T_pred_in_world = torch.inverse(T_direction_world) @ T_setpoint_world
                T_pred_in_drone = torch.inverse(T_drone_in_world) @ T_pred_in_world
                target_yaw = torch.atan2(T_pred_in_drone[1, 0], T_pred_in_drone[0, 0])

            while distance > 0.05 and i < 5000:
                if timeout is not None and (time.time() - time_start_optim_step > timeout):  # 30 Hz
                    break
                opt.zero_grad()

                if model_name == 'yolov5':
                    img = torch.nn.functional.interpolate(img, size=(320, 640), mode='bilinear', align_corners=False)
                    if args.ghost_mode:
                        try:
                            target_box = torch.tensor([bb_from_xyz(model.cam.camera_intrinsic, model.cam.camera_extrinsic, T_pred_in_drone[:3, 3].detach().cpu().numpy(), model.cam.radius.detach().cpu().item())], device=device, dtype=torch.float32)
                            # scale to image of size 320 x 640
                            scaled_target_box = target_box.clone()
                            scaled_target_box[0, 0] *= (640.0 / 160.0)  # x coords
                            scaled_target_box[0, 1] *= (320.0 / 96.0)   # y coords
                            scaled_target_box[0, 2] *= (640.0 / 160.0)  # x coords
                            scaled_target_box[0, 3] *= (320.0 / 96.0)   # y coords
                        except TypeError:
                            print("Bounding box could not be computed, skipping...")
                            target_idx += 1
                            retries = 0
                            # best_setpoint = torch.tensor(all_drone_poses[-1], device=device, dtype=torch.float32) # set current pose as setpoint
                            best_patch = patch.detach().clone()
                            target_box = None
                            scaled_target_box = None
                            skip = True
                        
                    # scaled_target_box = torch.tensor([target_box], device=device, dtype=torch.float32)  # (1, 4)

                    # sanity check if box can be computed back to T_pred_in_drone
                    # recovered_box = scaled_target_box.clone()
                    # recovered_box[:, [0, 2]] *= (160.0 / 640.)  # x coords
                    # recovered_box[:, [1, 3]] *= (96.0 / 320.)   # y coords
                    # recovered_xyzyaw = model.cam.batch_xyz_from_boxes(recovered_box)
                    # print("Recovered xyzyaw:", recovered_xyzyaw)
                    # print("Original relative position:", T_pred_in_drone[:3, 3].detach().cpu().numpy(), "Original yaw:", target_yaw.item())
                    # print("Distance:", torch.dist(recovered_xyzyaw[0, :3], T_pred_in_drone[:3, 3], p=2).item())

                    # T_rec_pred_in_drone = T_matrix(recovered_xyzyaw[0])
                    # T_rec_pred_in_world = T_drone_in_world @ T_rec_pred_in_drone
                    # # print("yaw recovered from box:", recovered_xyzyaw[0, 3].item())
                    # target_yaw = torch.tensor(0, device=device, dtype=torch.float32)#normalize_yaw_t(recovered_xyzyaw[0, 3])
                    # T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
                    # T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
                    # # print("Direction in world within loop:")
                    # # print(T_direction_world)

                    # T_rec_setpoint_world = T_direction_world @ T_rec_pred_in_world
                    # yaw = torch.atan2(T_rec_setpoint_world[1, 0], T_rec_setpoint_world[0, 0])
                    # rec_setpoint_yaw = normalize_yaw_t(yaw)

                    # print("Recovered setpoint in world: ", T_rec_setpoint_world[:3, 3], rec_setpoint_yaw.item())
                    # print("Original target: ", target[:3], target[3].item())


                if T[0, 0] > 0.1:  # avoid too small patches
                    manipulated_image = project_patch(
                        patches=patch, 
                        T_matrices=T.unsqueeze(0),  # add batch dimension
                        images=img
                    )
                    

                else:
                    manipulated_image = img.clone()
                    if not skip:
                        print("Patch too small, skipping to next target.")
                        target_idx += 1
                        retries = 0
                        best_patch = patch.detach().clone()
                        skip = True
                        # best_setpoint = torch.tensor(all_drone_poses[-1], device=device, dtype=torch.float32) # set current pose as setpoint
                        # break
                    # target_idx += 1
                    # retries = 0
                    # print("Patch too small, skipping to next target.")
                    # best_setpoint = torch.tensor(all_drone_poses[-1], device=device, dtype=torch.float32) # set current pose as setpoint
                    # break


                # # add noise to manipulated image
                # noise = torch.randn_like(manipulated_image) * 0.1
                # manipulated_image = manipulated_image + noise

                manipulated_image.clamp_(0., 1.)

                if model_name == 'frontnet':
                    x, y, z, yaw = model(manipulated_image*255.)
                    # print("x, y, z, yaw:", x, y, z, yaw)
                    prediction = torch.stack([x, y, z, yaw])
                    prediction = prediction.squeeze(2).mT

                    T_pred_in_drone = T_matrix(prediction[0])
                    T_pred_in_world = T_drone_in_world @ T_pred_in_drone
                    # print("T_pred_in_world within loop:")
                    # print(T_pred_in_world)

                    target_yaw = normalize_yaw_t(prediction[0, 3])
                    T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
                    T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
                    # print("Direction in world within loop:")
                    # print(T_direction_world)

                    T_setpoint_world = T_direction_world @ T_pred_in_world

                    # yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
                    # setpoint_yaw = normalize_yaw_t(yaw)
                    setpoint_yaw = target_yaw  # keep yaw predicted by network




                    # print("Target: ", target)
                    # print("Predicted setpoint in world: ", T_setpoint_world[:3, 3], target_yaw)

                    prediction = torch.stack([*T_setpoint_world[:3, 3], setpoint_yaw]).to(device) # prediction values


                    if T[0, 0] < 0.1:
                        best_patch = patch.detach().clone()
                        best_setpoint = prediction.detach().clone()
                        break  # avoid too small patches
                    # distance = torch.norm(prediction[0, :3] - target[:3], p=2)
                    #distance = F.mse_loss(prediction[0, :3], target[:3])
                    # distance = torch.mean((prediction[0, :3] - target[:3]).pow(2))
                    # distance = torch.sqrt((prediction[0, :3] - target[:3]).pow(2)).mean()
                    # print("Prediction:", prediction.shape)
                    # print("Target:", target.shape)
                    distance = torch.dist(prediction[:3], target[:3], p=2)
                    angular_loss = 1 - torch.cos(normalize_yaw_t(prediction[3]) - normalize_yaw_t(target[3]))

                    # print("Distance old:", distance_old)
                    # print("Distance new:", distance)

                    # x,y,z,yaw loss
                    loss = distance + angular_loss #+ (error_boxes * 0.001)  # try adding small weight to yolo box loss

                    
                    if patch_mode == 'velo' or patch_mode == 'timeout':
                        current_state = torch.stack([*T_drone_in_world[:3, 3], setpoint_yaw]).to(device)
                        prediction, vel_cmd = controller.step(current_state=current_state, target_state=prediction, dt=dt)
                        T_setpoint_world = T_matrix(prediction)

                elif model_name == 'yolov5':

                    manipulated_image_resized = manipulated_image.repeat_interleave(3, dim=1)  # to 3 channels
                    if args.ghost_mode and scaled_target_box is not None:
                        #center of scaled target box
                        center_target_box = torch.tensor([(scaled_target_box[0,0] + scaled_target_box[0,2]) / 2,
                                                        (scaled_target_box[0,1] + scaled_target_box[0,3]) / 2], device=device)

                        predicted_boxes, predicted_scores = model(manipulated_image_resized, target_anchor=center_target_box)  # yolo expects images in range [0, 1], out (B, 4)
                        # # print("Predicted boxes:", predicted_boxes, predicted_boxes.shape)
                        # # print("Predicted scores:", predicted_scores, predicted_scores.shape)
                        # # print("Scaled target box:", scaled_target_box)

                        loss_confidence = F.mse_loss(predicted_scores, torch.ones_like(predicted_scores).to(device))

                        if predicted_boxes.shape[0] > 1:
                            # choose box with highest confidence
                            scores = F.softmax(predicted_scores, dim=0)
                            predicted_boxes = (predicted_boxes * scores.unsqueeze(-1)).sum(dim=0, keepdim=True)
                            # predicted_scores = scores.sum(dim=0, keepdim=True)

                        
                        loss_boxes = F.mse_loss(predicted_boxes, scaled_target_box)

                    # # average_box = predicted_boxes.mean(dim=0)
                    # # scaled_box = average_box.clone()
                    # predicted_boxes_old = model.detection_single(manipulated_image_resized).squeeze(1)  # (1, 4)
                    predicted_boxes, _ = model(manipulated_image_resized, target_anchor=None)
                    # if predicted_boxes.grad_fn is None:
                    #     print("No box detected, skipping iteration")
                    #     continue
                    # print(predicted_boxes, predicted_boxes.shape)
                    scaled_box = predicted_boxes.clone()
                    # scale back to 160x96
                    scaled_box[:, [0, 2]] *= (160.0 / 640.)  # x coords
                    scaled_box[:, [1, 3]] *= (96.0 / 320.)   # y coords
                    # print("Averaged predicted box:", average_box)
                    # print("Scaled box for 160x96 image: ", scaled_box)
                    # print("Target box for 160x96 image: ", target_box)

                    
                    
                    recovered_xyzyaw = model.cam.batch_xyz_from_boxes(scaled_box)
                    T_rec_pred_in_drone = T_matrix(recovered_xyzyaw[0])
                    T_rec_pred_in_world = T_drone_in_world @ T_rec_pred_in_drone
                    target_yaw = normalize_yaw_t(recovered_xyzyaw[0, 3])
                    T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
                    T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
                    # print("Direction in world within loop:")
                    # print(T_direction_world)

                    T_setpoint_world = T_direction_world @ T_rec_pred_in_world
                    yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
                    setpoint_yaw = normalize_yaw_t(yaw)

                    prediction = torch.stack([*T_setpoint_world[:3, 3], setpoint_yaw]).to(device) # prediction values

                    distance = torch.dist(prediction[:3], target[:3], p=2)
                    angular_loss = 1 - torch.cos(normalize_yaw_t(prediction[3]) - normalize_yaw_t(target[3]))

                    loss = distance + angular_loss

                    if args.ghost_mode:
                        loss = loss + loss_confidence + loss_boxes

                    #loss = loss_confidence + loss_boxes + distance + angular_loss

                    with torch.no_grad():
                        if patch_mode == 'velo' or patch_mode == 'timeout':
                            current_state = torch.stack([*T_drone_in_world[:3, 3], setpoint_yaw]).to(device)
                            prediction, vel_cmd = controller.step(current_state=current_state, target_state=prediction, dt=dt)
                            T_setpoint_world = T_matrix(prediction)
                            prediction = torch.stack([*T_setpoint_world[:3, 3], normalize_yaw_t(torch.atan2(T_setpoint_world[1,0], T_setpoint_world[0,0]))]).to(device)

                    # print("Recovered xyzyaw:", recovered_xyzyaw)
                    # print("Original relative position:", T_pred_in_drone[:3, 3].detach().cpu().numpy(), "Original yaw:", target_yaw.item())
                    # print("Distance:", torch.dist(recovered_xyzyaw[0, :3], T_pred_in_drone[:3, 3], p=2).item())

                    # prediction = model.detection_single(manipulated_image_resized).squeeze(1)  # (1, 4)
                    # print("Old prediction:", prediction)
                    # # scale back to 160x96
                    # prediction[:, [0, 2]] *= (160.0 / 640.0)  # x coords
                    # prediction[:, [1, 3]] *= (96.0 / 320.0)   # y coords

                    # prediction = model.cam.batch_xyz_from_boxes(prediction) #  xyzyaw from bounding box
                    # # print("Prediction:", prediction, prediction.shape)
                    # # print("Target:", target, target.shape)
                    
                    # loss = loss_confidence + loss_boxes + distance + angular_loss
                   
                
                
                if distance < best_distance:
                    best_distance = distance.detach().clone()
                    best_patch = patch.detach().clone()
                    best_setpoint = prediction.detach().clone()

                if not skip:
                    loss.backward()
                    patch.grad = patch.grad.sign()
                    opt.step()
                    scheduler.step()
                else:
                    print("No gradients, stopping optimization.")
                    break
                losses.append(loss.detach().cpu().item())
                if i % 100 == 0:
                    print(f"Iter {i}, loss: {loss}, distance: {distance}, angle: {angular_loss}, mean loss last 50 iters: {np.mean(losses[-50:])}")
                    # print("Std last 50 losses:", np.std(losses[-50:]))

                patch.data.clamp_(0., 1.)
                i += 1

            print("Output dir: ", output_dir)

            np.save(output_dir / f'patch_{optim_steps}.npy', best_patch.detach().cpu().numpy())

            np.save(output_dir / f'T_{optim_steps}.npy', T.detach().cpu().numpy())


            with torch.no_grad():
                manipulated_image = project_patch(
                    patches=best_patch, 
                    T_matrices=T.unsqueeze(0),  # add batch dimension
                    images=img
                )


            print("Iterations needed: ", i)

            T_drone_in_world = T_matrix(best_setpoint)
            all_drone_poses.append(best_setpoint.detach().cpu().numpy())

            optim_steps += 1

        else: # random, black, white
            if patch_mode == 'random':
                patch = torch.rand((1, 1, initial_patch_size[0], initial_patch_size[1]), device=device, dtype=torch.float32)


            if model_name == 'yolov5':
                img = torch.nn.functional.interpolate(img, size=(320, 640), mode='bilinear', align_corners=False)

            manipulated_image = project_patch(
                    patches=patch, 
                    T_matrices=T.unsqueeze(0),  # add batch dimension
                    images=img
                )
            
            manipulated_image.clamp_(0., 1.)

            if model_name == 'frontnet':
                x, y, z, yaw = model(manipulated_image*255.)
                # print("x, y, z, yaw:", x, y, z, yaw)
                prediction = torch.stack([x, y, z, yaw])
                prediction = prediction.squeeze(2).mT

            elif model_name == 'yolov5':
                    # resize to 640x320
                    # manipulated_image_inter = torch.nn.functional.interpolate(manipulated_image, size=(320, 640), mode='bilinear', align_corners=False)
                    # gray to rgb
                    manipulated_image_resized = manipulated_image_inter.repeat_interleave(3, dim=1)

                    prediction = model(manipulated_image_resized).squeeze(1)  # yolo expects images in range [0, 1], out (B, 1, 4)
                    
                    #scale back to 160x96
                    prediction[:, [0, 2]] *= (160.0 / 640.0)  # x coords
                    prediction[:, [1, 3]] *= (96.0 / 320.0)   # y coords

                    # print("YOLOv5 prediction before scaling to 160x96:", prediction)
                    # print("Scaled bounding box for 160x96 image: ", prediction)

                    prediction = model.cam.batch_xyz_from_boxes(prediction) #  xyzyaw from bounding box

            # prediction values
            T_pred_in_drone = T_matrix(prediction[0])
            T_pred_in_world = T_drone_in_world @ T_pred_in_drone
            # print("T_pred_in_world within loop:")
            # print(T_pred_in_world)

            target_yaw = normalize_yaw_t(prediction[0, 3])
            T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
            T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
            # print("Direction in world within loop:")
            # print(T_direction_world)

            T_setpoint_world = T_direction_world @ T_pred_in_world

            yaw = torch.atan2(T_setpoint_world[1, 0], T_setpoint_world[0, 0])
            setpoint_yaw = normalize_yaw_t(yaw)

            prediction = torch.stack([*T_setpoint_world[:3, 3], setpoint_yaw]).to(device) # prediction values
            if patch_mode == 'velo' or patch_mode == 'timeout':
                current_state = torch.stack([*T_drone_in_world[:3, 3], setpoint_yaw]).to(device)
                prediction, vel_cmd = controller.step(current_state=current_state, target_state=prediction, dt=dt)
                T_setpoint_world = T_matrix(prediction)
            
            
            best_setpoint = prediction[0].detach().clone()
            np.save(output_dir / f'T_{target_idx}.npy', T.detach().cpu().numpy())

            T_drone_in_world = T_setpoint_world.clone()
            all_drone_poses.append(best_setpoint.detach().cpu().numpy())

        
        
        # print(all_drone_poses)

        print("Current drone pose: ", best_setpoint)
        print("Target pose that was to be reached: ", target)
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(manipulated_image[0, 0].detach().cpu().numpy(), cmap='gray')
        # plt.plot(monitor_corners[:, 0], monitor_corners[:, 1], 'r--', label='Monitor corners')
        
        # monitor_corners are in format (ul_x, ul_y), (ur_x, ur_y), (ll_x, ll_y), (lr_x, lr_y)
        monitor_corners = monitor_corners.detach().cpu().numpy()
        axs[0].plot([monitor_corners[0, 0], monitor_corners[1, 0]], [monitor_corners[0, 1], monitor_corners[1, 1]], 'r--')  # top edge
        axs[0].plot([monitor_corners[0, 0], monitor_corners[2, 0]], [monitor_corners[0, 1], monitor_corners[2, 1]], 'r--')  # left edge
        axs[0].plot([monitor_corners[1, 0], monitor_corners[3, 0]], [monitor_corners[1, 1], monitor_corners[3, 1]], 'r--')  # right edge
        axs[0].plot([monitor_corners[2, 0], monitor_corners[3, 0]], [monitor_corners[2, 1], monitor_corners[3, 1]], 'r--')  # bottom edge    

        axs[1].plot(target_trajectory[:, 0].detach().cpu().numpy(), target_trajectory[:, 1].detach().cpu().numpy(), 'r--')
        axs[1].plot(np.array(all_drone_poses)[:, 0], np.array(all_drone_poses)[:, 1])
        axs[1].scatter(projector_world[:, 0].detach().cpu().numpy(), projector_world[:, 1].detach().cpu().numpy(), c='b', label='Projector corners')
        axs[1].scatter(best_setpoint.detach().cpu().numpy()[0], best_setpoint.detach().cpu().numpy()[1], color='black')
        axs[1].arrow(best_setpoint.detach().cpu().numpy()[0], best_setpoint.detach().cpu().numpy()[1],
                        0.3 * np.cos(best_setpoint.detach().cpu().numpy()[3]), 0.3 * np.sin(best_setpoint.detach().cpu().numpy()[3]),
                        head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        
        axs[1].set_xlim(-2, 2)
        axs[1].set_ylim(-2, 2)
        plt.tight_layout()

        plt.savefig(output_dir / f'optim_step{optim_steps}.png')
        plt.close()

    all_drone_poses = np.array(all_drone_poses)
    np.save(output_dir / 'all_drone_poses.npy', all_drone_poses)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(target_trajectory[:, 0].detach().cpu().numpy(), target_trajectory[:, 1].detach().cpu().numpy(), 'r--')
    # ax.plot(all_drone_poses[:, 0], all_drone_poses[:, 1])
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-1.5, 1.5)
    # plt.tight_layout()
    # plt.savefig('test_single/trajectory.png')
    # plt.close()


    # # euclidean distance between all_drone_poses and target_trajectory
    # distance = np.linalg.norm(all_drone_poses[:, :3] - target_trajectory[1:, :3].detach().cpu().numpy())
    # print("Euclidean distances between drone poses and target trajectory:", distance)

    # if all_drone_poses longer than target_trajectory, calc distance with dynamic time warping
    all_drone_poses_tensor = torch.tensor(all_drone_poses[:, :3], device=device, dtype=torch.float32)
    target_trajectory_tensor = target_trajectory[:, :3]
    dtw_distance = compute_dtw_distance(all_drone_poses_tensor, target_trajectory_tensor)
    print("DTW distance between drone poses and target trajectory:", dtw_distance.item())

    end_time = time.time()
    print("Total optimization time (s): ", end_time - start_time)
    # save elapsed time to a text file
    with open(output_dir / 'elapsed_time.txt', 'w') as f:
        f.write("Elapsed time (s): \n")
        f.write(f"{end_time - start_time}\n")
        f.write(f"DTW distance: \n")
        f.write(f"{dtw_distance.item()}\n")
        f.write(f"Start time: \n")
        f.write(f"{start_time}\n")
        f.write(f"End time: \n")
        f.write(f"{end_time}\n")
