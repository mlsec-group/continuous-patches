import numpy as np
import rowan

import matplotlib.pyplot as plt


drone_pose = np.array([0.234, 0.86, 1.0, 0.0])


target_trajectory = np.array([#[0.0, 0.25, 1., 0.0],
                                  [0.0, 0.50, 1., 0.0],
                                  #[0.0, 0.75, 1., 0.0],
                                  [0.0, 1.00, 1., 0.0],
                                  #[0.0, 0.75, 1., 0.0],
                                  [0.0, 0.50, 1., 0.0],
                                  #[0.0, 0.25, 1., 0.0],
                                  [0.0, 0.00, 1., 0.0],
                                  #[0.0, -0.25, 1., 0.0],
                                  [0.0, -0.50, 1., 0.0],
                                  #[0.0, -0.75, 1., 0.0],
                                  [0.0, -1.00, 1., 0.0],
                                  #[0.0, -0.75, 1., 0.0],
                                  [0.0, -0.50, 1., 0.0],
                                  #[0.0, -0.25, 1., 0.0],
                                  [0.0, 0.00, 1., 0.0]])


current_checkpoint = target_trajectory[0]


print("Drone pose: ", drone_pose)
print("Checkpoint position world: ", current_checkpoint)


direction_vector = [1. * np.cos(np.pi), 1. * np.sin(np.pi), 0.]

p_frontnet_in_world = current_checkpoint[:3] - direction_vector

print("Checkpoint position frontnet in world: ", p_frontnet_in_world)


quats_checkpoint = rowan.from_euler(0., 0., current_checkpoint[3], convention='xyz') # returns qw, qx, qy, qz
rot_matrix_checkpoint = rowan.to_matrix(quats_checkpoint)
T_checkpoint = np.eye(4)
T_checkpoint[:3, :3] = rot_matrix_checkpoint
T_checkpoint[:3, 3] = current_checkpoint[:3]

direction_vector = [1. * np.cos(np.pi), 1. * np.sin(np.pi), 0.]
T_direction = np.eye(4)
T_direction[:3, 3] = direction_vector

T_fn_world = np.linalg.inv(T_direction) @ T_checkpoint

print("T fn world: ", T_fn_world)


# transformations matrix statt einzelne funktionen 
quats = rowan.from_euler(0., 0., drone_pose[3], convention='xyz') # returns qw, qx, qy, qz
rot_matrix = rowan.to_matrix(quats)
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rot_matrix
transformation_matrix[:3, 3] = drone_pose[:3]

# p_frontnet_in_body = rowan.rotate(quats, p_frontnet_in_world-drone_pose[:3])

p_frontnet_in_body = (np.linalg.inv(transformation_matrix) @ np.array([*p_frontnet_in_world, 1.]))[:3]

print("Checkpoint position frontnet in drone frame: ", p_frontnet_in_body)


T_fn_drone = np.linalg.inv(transformation_matrix) @ T_fn_world
print("T fn drone: ", T_fn_drone)

quats_fn_drone = rowan.from_matrix(T_fn_drone[:3, :3])
euler_angles = rowan.to_euler(quats_fn_drone, convention='xyz')
print("euler angles from matrix: ", euler_angles)


target_x, target_y, target_z = p_frontnet_in_body.tolist()


predicted_pose = np.array([target_x, target_y, target_z, -np.pi])
T_fn_p_drone = np.eye(4)
T_fn_p_drone[:3, :3] = rowan.to_matrix(rowan.from_euler(0., 0., predicted_pose[3], convention='xyz'))
T_fn_p_drone[:3, 3] = predicted_pose[:3]
print("T fn p drone: ", np.round(T_fn_p_drone, 2))

quats = rowan.from_euler(0., 0., drone_pose[3], convention='xyz') # returns qw, qx, qy, qz


rot_matrix = rowan.to_matrix(quats)
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rot_matrix
transformation_matrix[:3, 3] = drone_pose[:3]

T_fn_p_world = transformation_matrix @ T_fn_p_drone

print("T fn p world: ", np.round(T_fn_p_world, 2))


rotated_desired = rowan.rotate(quats, predicted_pose[:3])
# print("Predicted pose in world: ", rotated_desired)
target_pos = drone_pose[:3] + rotated_desired
print("Target pose in world: ", target_pos)

# predicted yaw angles are discarded since they are very faulty
# global_pos = target_pos - drone_pose[:3]
# target_yaw = np.arctan2(global_pos[1], global_pos[0]) #- np.pi

# frontnet_yaw = -np.pi
direction_vector = [1. * np.cos(predicted_pose[3]), 1. * np.sin(predicted_pose[3]), 0.]

T_direction_world = np.eye(4)
# T_direction_world[:3, :3] = rowan.to_matrix(rowan.from_euler(0., 0., 0., convention='xyz'))
T_direction_world[:3, 3] = direction_vector

T_setpoint_world = T_direction_world @ T_fn_world
print("T setpoint world: ", np.round(T_setpoint_world, 2))
setpoint_yaw = rowan.to_euler(rowan.from_matrix(T_setpoint_world[:3, :3]), convention='xyz')[2]

new_setpoint = target_pos + direction_vector


print("new setpoint: ", new_setpoint)


# plot in 2d (ommit z axis), add arrows for direction
# fig, ax = plt.subplots()
