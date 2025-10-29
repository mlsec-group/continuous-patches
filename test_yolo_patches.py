import yaml
import numpy as np
import cv2
from util import load_dataset
from plots import img_placed_patch
from camera import Camera
import argparse

import torch
from yolo_bounding import YOLOBox
from attack_minimal_single import T_matrix, normalize_yaw_t, calc_heading_vec

if __name__ == "__main__":

    # model = YOLOBox()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_target_x = np.random.uniform(0.,1.5, 1)
    random_target_y = np.random.uniform(-1,1, 1)
    random_target_z = np.random.uniform(-0.5,0.5, 1)
    
    random_target_yaw = np.random.uniform(-0.3, 0.3, 1)

    drone_pose_in_world = torch.tensor([1.5, -.2, 1., 0.])  # x, y, z, yaw
    T_drone_in_world = T_matrix(drone_pose_in_world)
    print("T_drone_in_world:")
    print(T_drone_in_world)

    target = np.hstack((random_target_x, random_target_y, random_target_z, random_target_yaw))
    target = torch.tensor(target, dtype=torch.float32)
    print("Target: ", target)

    T_pred_in_drone = T_matrix(target)
    print("T_pred_in_drone:")
    print(T_pred_in_drone)

    T_pred_in_world = T_drone_in_world @ T_pred_in_drone
    print("T_pred_in_world:")
    print(T_pred_in_world)
    
    target_yaw = normalize_yaw_t(target[3])
    T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
    T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
    print("T_direction_world:")
    print(T_direction_world)

    T_setpoint_world = T_direction_world @ T_pred_in_world
    print("T_setpoint_world:")
    print(T_setpoint_world)

    recovered_target_yaw = torch.atan2(T_setpoint_world[1,0], T_setpoint_world[0,0])
    print("Recovered target yaw: ", recovered_target_yaw)

    T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
    T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(recovered_target_yaw - torch.pi)).to(device)
    print("T_direction_world:")
    print(T_direction_world)

    T_pred_in_world_recovered = torch.linalg.inv(T_direction_world) @ T_setpoint_world
    print("T_pred_in_world_recovered:")
    print(T_pred_in_world_recovered)

    # reverse process as sanity check to recover target in drone frame
    T_world_in_drone = torch.linalg.inv(T_drone_in_world)
    T_pred_in_drone_recovered = T_world_in_drone @ T_pred_in_world_recovered
    print("T_pred_in_drone_recovered:")
    print(T_pred_in_drone_recovered)

    # recover yaw
    recovered_yaw = torch.atan2(T_pred_in_drone_recovered[1,0], T_pred_in_drone_recovered[0,0])
    print("Recovered yaw: ", recovered_yaw)
    print("Original target yaw: ", target[3])