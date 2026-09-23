import numpy as np
import torch
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.util import gen_target_trajectory
from src.simulation import T_matrix, normalize_yaw, calc_heading_vec


target_trajectory = gen_target_trajectory('triangle')


drone_pose = target_trajectory[0]
setpoint_pose = target_trajectory[1]

print("Drone Pose: ", drone_pose.detach().cpu().numpy())
print("Setpoint Pose: ", setpoint_pose.detach().cpu().numpy())

T_drone_world = T_matrix(drone_pose)
T_setpoint_world = T_matrix(setpoint_pose)

target_yaw = torch.atan2((setpoint_pose[1] - drone_pose[1]), (setpoint_pose[0] - drone_pose[0]))
target_yaw = normalize_yaw(target_yaw)
print("Target Yaw in Drone Frame: ", target_yaw.item())

T_dir = torch.eye(4)
T_dir[:3, 3] = calc_heading_vec(1.0, normalize_yaw(target_yaw-torch.pi), device=drone_pose.device).squeeze()

T_pred_world = torch.inverse(T_dir) @ T_setpoint_world
print("Predicted Pose in World Frame: ", T_pred_world.detach().cpu().numpy())

T_pred_drone = torch.inverse(T_drone_world) @ T_pred_world
print("Predicted Pose in Drone Frame: ", T_pred_drone.detach().cpu().numpy())



T_pred_world = T_drone_world @ T_pred_drone
print("Recovered T pred in world frame: ", T_pred_world.detach().cpu().numpy())
T_dir = torch.eye(4)
T_dir[:3, 3] = calc_heading_vec(1.0, normalize_yaw(target_yaw-torch.pi), device=drone_pose.device).squeeze()

T_setpoint = T_dir @ T_pred_world
print("Recovered T setpoint in world frame: ", T_setpoint.detach().cpu().numpy())
