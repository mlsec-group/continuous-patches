import numpy as np
import rowan

import matplotlib.pyplot as plt
import cv2

import yaml

import torch

def gen_horizontal_line(start, end, num_points):
    x = np.zeros(num_points)
    y = np.linspace(start, end, num_points)
    return np.column_stack([x, y])


def gen_circle(r, n):
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.array([x, y]).T

def gen_transformation_matrix(sf, tx, ty):
    matrix = np.zeros((3,3))
    matrix[0,0] = sf
    matrix[1,1] = sf
    matrix[0, 2] = tx
    matrix[1, 2] = ty
    matrix[2, 2] = 1.
    return matrix

def get_bb_patch(transformation):
    transformation_t = torch.tensor(transformation).unsqueeze(0)
    inv_transform = torch.inverse(transformation_t)[:, :2]
    affine_grid = torch.nn.functional.affine_grid(inv_transform, size=(1, 1, 96, 160), align_corners=True)

    transformed_patch = torch.nn.functional.grid_sample(torch.ones(1, 1, 96, 160).double(), affine_grid, align_corners=True, padding_mode='zeros')

    patch_coords = torch.nonzero(transformed_patch)[..., 2:]

    xmin = patch_coords[0][1].item()
    ymin = patch_coords[0][0].item()
    xmax = patch_coords[-1][1].item()
    ymax = patch_coords[-1][0].item()

    # fig, ax = plt.subplots()
    # ax.imshow(transformed_patch[0][0], cmap='gray')
    # plt.show()

    return [xmin, ymin, xmax, ymax]


# printed patch width = 28.2 cm
RADIUS = 0.137 # in m

# compute relative position of center of patch in camera frame
def xyz_from_bb(bb,mtrx, dist_coeffs):
    # bb - xmin,ymin,xmax,ymax
    # mtrx, dist_vec = get_camera_parameters()
    fx = np.array(mtrx)[0][0]
    fy = np.array(mtrx)[1][1]
    ox = np.array(mtrx)[0][2]
    oy = np.array(mtrx)[1][2]
    # get pixels for bb side center
    P1 = np.array([bb[0],(bb[1] + bb[3])/2])
    P2 = np.array([bb[2],(bb[1] + bb[3])/2])
    # rectify pixels
    P1_rec = cv2.undistortPoints(P1, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix
    P2_rec = cv2.undistortPoints(P2, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix

    # get rays for pixels
    a1 = np.array([(P1_rec[0]-ox)/fx, (P1_rec[1]-oy)/fy, 1.0])
    a2 = np.array([(P2_rec[0]-ox)/fx, (P2_rec[1]-oy)/fy, 1.0])
    # normalize rays
    a1_norm = np.linalg.norm(a1)
    a2_norm = np.linalg.norm(a2)
    # get the distance    
    distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))
    # get central ray
    ac = (a1+a2)/2
    # get the position
    xyz = distance*ac/np.linalg.norm(ac)
    return xyz


def calc_max_possible_victim_change(current_victim_pose, target_position):
    max_new_pos = current_victim_pose[:2, 3] + target_position[:2]   # v_x + target_x, v_y + target_y
    max_new_pos[0] -= 1. # substract safety radius

    victim_quat = rowan.from_matrix(current_victim_pose[:3, :3])

    new_pose_world = np.zeros((4, 4))
    new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
    new_pose_world[:2, 3] = max_new_pos
    new_pose_world[-1, -1] = 1.

    return new_pose_world

def update_attacker_pose(optimized_patch_positions, T_victim_world_c, T_victim_world_d, camera_intrinsics, camera_extrinsics, ):

    target_positions = np.array([[1, 1, 0], [1, -1, 0]])
    
    A = np.array([[True, False], [False, True]])
    # possible_attacker_position = np.array([[0.28316405,  0.20245424, -0.2117372 ],
                                        #    [0.30532456, -0.20091306, -0.22638479]])



    T_attacker_in_world_d = {'matrices': [], 'distances': []}
    for k in range(A.shape[1]):

        m = np.argwhere(A[..., k]==True)[0, 0]
        transformation_matrix = gen_transformation_matrix(*optimized_patch_positions[m,k])

        bounding_box_placed_patch = get_bb_patch(transformation_matrix)

        patch_in_camera = xyz_from_bb(bounding_box_placed_patch, camera_intrinsics)
        # print(patch_in_camera)
        patch_in_victim = (np.linalg.inv(camera_extrinsics) @ [*patch_in_camera, 1])[:3]
        print(patch_in_victim)
        # print(patch_rel_position)T_victim_world_c

        # T_patch_in_victim = np.zeros((4,4))
        # T_patch_in_victim[:3, :3] = np.eye(3,3)
        # T_patch_in_victim[:3, 3] = patch_in_victim
        # T_patch_in_victim[-1, -1] = 1.

        # print(T_patch_in_victim)

        #T_patch_in_world = T_victim_world @ T_patch_in_victim   <--- seems faulty to me
        T_patch_in_world = T_victim_world_c.copy()
        T_patch_in_world[:3, 3] += patch_in_victim
        # print(T_patch_in_world)

        T_attacker_in_world = T_patch_in_world.copy()
        T_attacker_in_world[2, 3] += 0.096  # center of CF is 9.6 cm above center of patch

        

        T_attacker_in_world_d['matrices'].append(T_attacker_in_world)

        T_possible_new_victim_pose = calc_max_possible_victim_change(T_victim_world_c, target_positions[k])

        # print(T_victim_world_c.shape, T_victim_world_d.shape, T_possible_new_victim_pose.shape)

        # Why base this decision on the victim pose?
        # "The objective is to minimize the tracking error of the victim
        # between its current pose pv and desired pose  ̄pv , i.e.∫t ∥ ̄pv (t) − ˆpv (t)∥dt."
        T_attacker_in_world_d['distances'].append(np.linalg.norm((T_victim_world_d-T_possible_new_victim_pose), ord=2))

    return T_attacker_in_world_d['matrices'][np.argmin(T_attacker_in_world_d['distances'])]


if __name__ == "__main__":

    # current victim pose, will be read from Mocap
    T_victim_world_c = np.array([[-1., 0., 0., 0.],        # initial pose in world frame 
                            [0., -1., 0., 0.],           # roll = 0, pitch = 0, yaw=0 
                            [0., 0., 1., 1.],            # no rotation, z = 1
                            [0., 0., 0., 1.] ])       

    # current attacker pose, will be read from Mocap
    T_attacker_world_c = np.array([[-1., 0., 0., 0.5],        # initial pose in world frame 
                            [0., -1., 0., -0.5],              # roll = 0, pitch = 0, yaw=0 
                            [0., 0., 1., 1.],                 # no rotation, z = 1
                            [0., 0., 0., 1.] ])           


    # assignment
    A = np.array([[True, False], [False, True]])

    # example opimized position norm
    # need to be read from eval/exp?/mode/positions_norm.npy[-1]
    # (or we store them in a new yaml file)
    positions_p0 = np.array([[0.5, -0.6, 0.4], [0.3, -0.2, -1.0]])
    positions_p1 = np.array([[0.5, -0.8, 0.3], [0.5, 0.6, 0.4]])
    patch_positions_image = np.stack([positions_p0, positions_p1])

    # all target poses
    # can be read from settings file
    target_positions = np.array([[1, 1, 0], [1, -1, 0]])#, [1, 0, 0], [2, 0, 0]])
    #num_targets = len(target_positions)


    camera_intrinsics = np.load("/home/pia/Documents/Coding/adversarial_frontnet/misc/aideck7/intrinsic.npy")
    camera_extrinsics = np.load("/home/pia/Documents/Coding/adversarial_frontnet/misc/aideck7/extrinsic.npy")

    print(camera_intrinsics, camera_extrinsics)

    # generate desired victim trajectory
    # e.g. only change y direction
    # first move to the right(y=1 in our flight space) and then to the left(y=-1 in our flight space)
    desired_trajectory = gen_horizontal_line(1, -1, 20)   #gen_circle(2., 20)
    # set first desired victim pose 
    desired_idx = 0   # victim is at 0,0 at the start which is the mid point of the desired trajectory
    T_victim_world_d = T_victim_world_c.copy()
    # keep rotation, only change tx, ty to new desired position
    T_victim_world_d[:2, 3] = desired_trajectory[desired_idx]
    # print(T_victim_world_d)


    T_attacker_world_c = update_attacker_pose(patch_positions_image, T_victim_world_c, T_victim_world_d, camera_intrinsics, camera_extrinsics)

    print(T_attacker_world_c)

    # TODO: add loop and fake victim update



    

# print(T_patch_in_world)



# desired next victim pose in world frame
#desired_trajectory = gen_circle(2., 20)
# desired_pose = np.array([[-1., 0., 0., 0.],          # desired pose in world frame 
#                             [0., -1., 0., 0.],         # 
#                             [0., 0., 1., 1.]])         # no rotation, z = 1

# #desired_idx = np.random.randint(low= 0, high=len(desired_trajectory)-1)
# #desired_pose[:2, 3] = desired_trajectory[desired_idx]

# # assignment
# #A = np.array([[True, True, False, False], [False, False, True, True]])
# A = np.array([[True, False], [False, True]])

# all_victim_positions = []
# all_desired_positions = []

# all_victim_positions.append(victim_pose[:2, 3].T)
# all_desired_positions.append(desired_pose[:2, 3].T)

# for i in range(5):
#     for target_idx in range(num_targets):



#     print(f"------{i}------")              
#     victim_pos= victim_pose[:2, 3]

#     victim_quat = rowan.from_matrix(victim_pose[:3, :3])


#     # desired_yaw should be 0
#     #r, p, desired_yaw = rowan.to_euler(rowan.from_matrix(victim_pose[:3,:3] @ desired_pose[:3, :3]))
#     desired_yaw = 0.

#     # frontnet_out = np.array([1, -1, 0]) + np.random.rand(3,)
#     # print("Frontnet prediction: ", frontnet_out)


#     possible_new_positions = np.zeros((num_targets, 3, 4))
#     # print(possible_new_positions)

#     for target_idx in range(num_targets):
#         # since we're keeping the yaw angle of the victim fixed, 
#         # we can simply compute the max. achievable new position in world coordinates:
#         max_new_pos = victim_pos + target_positions[target_idx][:2]
#         max_new_pos[0] -= 1. # safety radius
#         max_new_pos[0] -= np.random.uniform(0., 0.2) # debug

#         new_pose_world = np.zeros((3, 4))
#         new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
#         new_pose_world[:2, 3] = max_new_pos

#         possible_new_positions[target_idx, :, :] = new_pose_world


#     # maybe not even needed
#     #     print("target: ", target_positions[target_idx])
#     #     print("victim_pos: ", victim_pos)
#     #     new_pos = rowan.rotate(victim_quat, target_positions[target_idx])
#     #     print("new_pos:", new_pos)
#     #     heading_vec = [1*np.cos(desired_yaw), 1*np.sin(desired_yaw), 0.]
#     #     print("heading_vec:", heading_vec)

#     #     new_pos_world = new_pos + victim_pos + heading_vec
#     #     print("new_pos in world: ", new_pos_world)
#     #     new_pose_world = np.zeros((3, 4))
#     #     new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
#     #     new_pose_world[..., 3][:3] = new_pos_world 
#     #     # print("new pose in world: ", new_pose_world)
#     #     possible_new_positions[target_idx, :, :] = new_pose_world

#     # # print(possible_new_positions)

#     distances = [np.linalg.norm(possible_new_positions[target_idx]-desired_pose, ord=2) for target_idx in range(num_targets)]
#     print("l2 dist to desired: ", distances)
#     k = np.argmin(distances)


#     print("k: ", k)
#     # print("Assignment: ", A[..., k])
#     m = np.argmax(A[..., k])
#     print("m: ", m)
#     # print(f"Presenting patch {m} at position {positions[m, k]} (in image frame)")

#     # pretend that pose got updated by frontnet output
#     # the pose will later be read from the mocap system
#     position_change = possible_new_positions[k] - victim_pose
#     victim_pose[:2, 3] += (0.3*(position_change[:2, 3]) + np.random.uniform(-0.3, 0.3, (2,))) # add a bit of uncertainty
#     print("updated victim pose: ", victim_pose)
#     dist_after_update = np.linalg.norm((victim_pose-desired_pose), ord=2)
#     print("error to desired: ", dist_after_update)
    

#     # distances = [np.linalg.norm(victim_pos-desired_trajectory[d_idx]) for d_idx in range(len(desired_trajectory))]
    
#     # # desired_idx = np.argmin(distances)
#     # #desired_pose[:2, 3] = desired_trajectory[desired_idx]

#     if dist_after_update <= 0.2:
#         if desired_idx < len(desired_trajectory)-1:
#             desired_idx += 1
#         else:
#             desired_idx = 0
#         desired_pose[:2, 3] = desired_trajectory[desired_idx]
#         print("updated desired pose! ", desired_pose)

#     all_victim_positions.append(victim_pose[:2, 3].copy())
#     all_desired_positions.append(desired_pose[:2, 3].copy())


# all_victim_positions = np.array(all_victim_positions)
# all_desired_positions = np.array(all_desired_positions)



# fig, ax = plt.subplots()
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5., 5)
# ax.plot(desired_trajectory.T[1], desired_trajectory.T[0])
# ax.plot(all_victim_positions.T[1],  all_victim_positions.T[0], label='victim position')#, marker=(3, 0, desired_yaw), markersize=20, linestyle='None')
# # for i, (a,b) in enumerate(zip(all_victim_positions.T[1], all_victim_positions.T[0])): 
# #     plt.text(a, b-0.007, str(i))

# ax.plot(all_desired_positions.T[1], all_desired_positions.T[0], marker=(3, 0, desired_yaw), markersize=20, linestyle='None', label='desired checkpoints')
# ax.legend()
# plt.show()