import torch
import numpy as np
from cv2 import findHomography
from .camera import Camera
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Math Helpers ---
def normalize_yaw(yaw):
    """Wraps angle to [-pi, pi]."""
    if isinstance(yaw, torch.Tensor):
        return torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return np.arctan2(np.sin(yaw), np.cos(yaw))

def calc_heading_vec(radius, angle, device):
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    zero = torch.zeros_like(x, device=device)
    return torch.stack((x, y, zero), dim=-1)

def T_matrix(pose):
    """Converts [x, y, z, yaw] to 4x4 Transformation Matrix."""
    T = torch.eye(4, dtype=torch.float32).to(pose.device)
    normalized_yaw = normalize_yaw(pose[3])

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

def solve_quadratic(a, b, c):
    """Solves ax^2 + bx + c = 0."""
    if np.abs(a) < 1e-9:
        if np.abs(b) < 1e-9: return (None, None)
        return (-c / b, -c / b)
    discriminant = b**2 - 4*a*c
    if discriminant < -1e-9: return (None, None)
    sqrt_d = np.sqrt(max(0.0, discriminant))
    return ((-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a))

def bb_from_xyz(camera_intrinsic, camera_extrinsic, new_xyz, radius):
    """Calculates 2D bbox of a sphere. Crucial for YOLO Ghost Mode."""
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    ox, oy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

    # Transform World -> Camera
    new_xyz_h = np.array([*new_xyz, 1.0])
    xyz = (camera_extrinsic @ new_xyz_h)[:3]
    
    distance = np.linalg.norm(xyz)
    if xyz[2] <= 1e-6 or distance < radius: return None

    xc, yc = xyz[0] / xyz[2], xyz[1] / xyz[2]
    ac_sq = xc**2 + yc**2 + 1.0
    cos_sq = 1.0 - (radius / distance)**2
    A = ac_sq * cos_sq
    
    x1, x2 = solve_quadratic(A - xc**2, -2 * (yc**2 + 1) * xc, (yc**2 + 1) * (A - (yc**2 + 1)))
    y1, y2 = solve_quadratic(A - yc**2, -2 * (xc**2 + 1) * yc, (xc**2 + 1) * (A - (xc**2 + 1)))
    
    if x1 is None or y1 is None: return None
    
    px = np.array([x1, x2]) * fx + ox
    py = np.array([y1, y2]) * fy + oy
    return np.array([min(px), min(py), max(px), max(py)])

# --- Controller ---
class PController:
    def __init__(self, max_vel=1.0, max_yaw_rate=0.2, kp_pos=1.0, kp_yaw=2.0):
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw

    def step(self, current_state, target_state, dt):
        """Returns (predicted_next_state, cmd_vel)"""
        # Position
        pos_error = target_state[:3] - current_state[:3]
        vel_cmd = pos_error * self.kp_pos
        speed = torch.norm(vel_cmd)
        if speed > self.max_vel:
            vel_cmd = (vel_cmd / speed) * self.max_vel

        # Yaw
        yaw_error = normalize_yaw(target_state[3] - current_state[3])
        yaw_rate_cmd = torch.clamp(yaw_error * self.kp_yaw, -self.max_yaw_rate, self.max_yaw_rate)

        # Integration
        new_pos = current_state[:3] + vel_cmd * dt
        new_yaw = normalize_yaw(current_state[3] + yaw_rate_cmd * dt)
        
        return torch.cat((new_pos, new_yaw.unsqueeze(0))), torch.cat((vel_cmd, yaw_rate_cmd.unsqueeze(0)))

# --- Simulation Environment ---
class DroneSimulation:
    def __init__(self, device, display_size_inch=60, start_pose=[0., 0., 1., 0.]):
        self.device = device
        self.cam = Camera(f'{project_root}/configs/camera_calibration.yaml', device=device)
        self.controller = PController(max_vel=15.0, max_yaw_rate=3.0, kp_pos=1.5, kp_yaw=3.0)
        self.pose = torch.tensor(start_pose, device=device, dtype=torch.float32)
        self._setup_monitor(display_size_inch)
        
    def _setup_monitor(self, size_inch):
        # we assume a 16:9 monitor for simplicity
        center = torch.tensor([2., 0.0, 1.], dtype=torch.float32, device=self.device)
        diag_m = size_inch * 0.0254
        h_m = np.round(diag_m / np.sqrt((16./9.)**2 + 1), 2)
        w_m = np.round(h_m * (16. / 9.), 2)
        
        # UL, UR, LL, LR (World Frame)
        self.monitor_world = torch.tensor([
            [center[0], center[1] + w_m/2, center[2] + h_m/2],
            [center[0], center[1] - w_m/2, center[2] + h_m/2],
            [center[0], center[1] + w_m/2, center[2] - h_m/2],
            [center[0], center[1] - w_m/2, center[2] - h_m/2]
        ], dtype=torch.float32, device=self.device)

    def get_view_geometry(self, patch_size_px, img_size_px):
        """Calculates Homography T and monitor corners in image."""
        
        T_drone = T_matrix(self.pose)
        inv_pose = torch.inverse(T_drone)
        
        ones = torch.ones(4, 1, device=self.device)
        pts_w_h = torch.cat([self.monitor_world, ones], dim=1).T
        pts_drone = inv_pose @ pts_w_h
        pts_cam = self.cam.camera_extrinsic_tens @ pts_drone[:4, :]
        pts_img_h = self.cam.camera_intrinsic_tens @ pts_cam[:3, :]
        
        # Check Visibility
        # pts_img_h[2] is camera-space depth; all points behind the camera -> not visible
        if not torch.any(pts_img_h[2, :] > 0): return None, None, False

        pts_img = pts_img_h[:2, :] / pts_img_h[2, :]
        monitor_corners = pts_img.T # (4, 2)

        # print("monitor_corners: ", monitor_corners)

        if img_size_px[0] > 96. or img_size_px[1] > 160:
            monitor_corners[:, 0] *= (640.0 / 160.0)  # x coords
            monitor_corners[:, 1] *= (320.0 / 96.0)   # y coords

        # print("Adjusted monitor_corners: ", monitor_corners)

        # As long as one point is within view, we consider it visible
        inside = ((monitor_corners[:, 0] >= 0) & (monitor_corners[:, 0] <= img_size_px[1]) &
                  (monitor_corners[:, 1] >= 0) & (monitor_corners[:, 1] <= img_size_px[0]))
        if not torch.any(inside): return None, None, False

        # Clamp corners to the image frame so the homography targets the visible region
        monitor_corners[:, 0] = torch.clamp(monitor_corners[:, 0], 0.0, float(img_size_px[1]))
        monitor_corners[:, 1] = torch.clamp(monitor_corners[:, 1], 0.0, float(img_size_px[0]))

        # Homography
        src_pts = np.array([
            [0., 0.], [patch_size_px[1], 0.], 
            [0., patch_size_px[0]], [patch_size_px[1], patch_size_px[0]]
        ], dtype=np.float32)
        dst_pts = monitor_corners.detach().cpu().numpy()
        
        try:
            H, _ = findHomography(src_pts, dst_pts, method=0)
            T = torch.tensor(H, dtype=torch.float32, device=self.device)
        except: return None, None, False
            
        # Validity Checks
        is_valid = True
        if T[0,0] < 0.1: is_valid = False 
        # if T[0,2] < -10 or T[1,2] < -10: is_valid = False
        # if T[0,2] > img_size_px[1] + 10 or T[1,2] > img_size_px[0] + 10: is_valid = False
        
        return T, monitor_corners, is_valid

    def update_physics(self, vel_cmd, dt):
        # print(vel_cmd, dt)
        self.pose[:3] += vel_cmd[:3].clone().detach() * dt
        self.pose[3] += vel_cmd[3].clone().detach() * dt
        self.pose[3] = normalize_yaw(self.pose[3])