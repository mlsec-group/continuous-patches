import yaml
import numpy as np
import cv2
from util import load_dataset
from plots import img_placed_patch
from camera import Camera
import argparse

import torch
from yolo_bounding import YOLOBox
from attack_minimal_single import T_matrix, normalize_yaw_t, calc_heading_vec, project_patch
import torch.nn.functional as F


RADIUS = 0.3
IM_HEIGHT = 96
IM_WIDTH = 160

#### sanity check for setpoint calculation
    # drone_pose_in_world = torch.tensor([1.5, -.2, 1., 0.])  # x, y, z, yaw
    # T_drone_in_world = T_matrix(drone_pose_in_world)
    # print("T_drone_in_world:")
    # print(T_drone_in_world)

    # target = np.hstack((random_target_x, random_target_y, random_target_z, random_target_yaw))
    # target = torch.tensor(target, dtype=torch.float32)
    # print("Target: ", target)

    # T_pred_in_drone = T_matrix(target)
    # print("T_pred_in_drone:")
    # print(T_pred_in_drone)

    # T_pred_in_world = T_drone_in_world @ T_pred_in_drone
    # print("T_pred_in_world:")
    # print(T_pred_in_world)
    
    # target_yaw = normalize_yaw_t(target[3])
    # T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
    # T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
    # print("T_direction_world:")
    # print(T_direction_world)

    # T_setpoint_world = T_direction_world @ T_pred_in_world
    # print("T_setpoint_world:")
    # print(T_setpoint_world)

    # recovered_target_yaw = torch.atan2(T_setpoint_world[1,0], T_setpoint_world[0,0])
    # print("Recovered target yaw: ", recovered_target_yaw)

    # T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
    # T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(recovered_target_yaw - torch.pi)).to(device)
    # print("T_direction_world:")
    # print(T_direction_world)

    # T_pred_in_world_recovered = torch.linalg.inv(T_direction_world) @ T_setpoint_world
    # print("T_pred_in_world_recovered:")
    # print(T_pred_in_world_recovered)

    # # reverse process as sanity check to recover target in drone frame
    # T_world_in_drone = torch.linalg.inv(T_drone_in_world)
    # T_pred_in_drone_recovered = T_world_in_drone @ T_pred_in_world_recovered
    # print("T_pred_in_drone_recovered:")
    # print(T_pred_in_drone_recovered)

    # # recover yaw
    # recovered_yaw = torch.atan2(T_pred_in_drone_recovered[1,0], T_pred_in_drone_recovered[0,0])
    # print("Recovered yaw: ", recovered_yaw)
    # print("Original target yaw: ", target[3])

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

def bb_from_xyz(camera_intrinsic, camera_extrinsic, new_xyz, RADIUS):
    """
    Calculates the 2D bounding box silhouette of a sphere
    centered at new_xyz (world coords) with a given RADIUS.
    
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
    if distance < RADIUS:
        print(f"Error: Camera is inside the object (distance {distance} < RADIUS {RADIUS}).")
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
    cos_half_theta_sq = 1.0 - (RADIUS / distance)**2
    
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

def gen_random_patch_coords():
    random_sf = np.random.uniform(0.2, 1.1)
    random_upper_left_x = np.random.uniform(0, 160 - 80 * random_sf, 1)
    random_upper_left_y = np.random.uniform(0, 96 - 45 * random_sf, 1)
    random_lower_right_x = random_upper_left_x + 80 * random_sf
    random_lower_right_y = random_upper_left_y + 45 * random_sf

    upper_left = [random_upper_left_x[0], random_upper_left_y[0]]
    lower_right = [random_lower_right_x[0], random_lower_right_y[0]]

    tx_min = upper_left[0]
    ty_min = lower_right[1]

    T = torch.eye(3, device=device, dtype=torch.float32)  # Identity transformation matrix
    T[:2, :2] *= random_sf  # Scale down to half the size of the patch
    T[0, 2] = tx_min
    T[1, 2] = ty_min

    return T

def check_bb_overlap(bb1, bb2, percentage=0.2):
    """
    Check if two bounding boxes overlap by a certain percentage.
    bb1, bb2: [x_min, y_min, x_max, y_max]
    percentage: minimum overlap percentage required to return True
    """
    x_min_overlap = max(bb1[0], bb2[0])
    y_min_overlap = max(bb1[1], bb2[1])
    x_max_overlap = min(bb1[2], bb2[2])
    y_max_overlap = min(bb1[3], bb2[3])

    if x_min_overlap < x_max_overlap and y_min_overlap < y_max_overlap:
        # Calculate areas
        area_bb1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        area_bb2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        area_overlap = (x_max_overlap - x_min_overlap) * (y_max_overlap - y_min_overlap)

        # Calculate overlap percentages
        overlap_percentage_bb1 = area_overlap / area_bb1
        overlap_percentage_bb2 = area_overlap / area_bb2

        if overlap_percentage_bb1 >= percentage or overlap_percentage_bb2 >= percentage:
            return True

    return False

if __name__ == "__main__":

    # 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cam_image_corner = np.array([0., 0., 160., 96.], dtype=np.float32)

    cam = Camera(path='camera_calibration.yaml')
    camera_intrinsic = cam.camera_intrinsic
    camera_extrinsic = cam.camera_extrinsic

    print("Camera Intrinsic:")
    print(camera_intrinsic)
    print("Camera Extrinsic:")
    print(camera_extrinsic)

    bb = None

    while bb is None:
        random_target_x = np.random.uniform(0.,1.5, 1)
        random_target_y = np.random.uniform(-1,1, 1)
        random_target_z = np.random.uniform(-0.5,0.5, 1)
        
        random_target_yaw = np.random.uniform(-0.3, 0.3, 1)


        target = np.hstack((random_target_x, random_target_y, random_target_z, random_target_yaw))
        
        bb = bb_from_xyz(camera_intrinsic, camera_extrinsic, target[:3], RADIUS)

        # check if bb has overlap with image
        if bb is not None and not check_bb_overlap(bb, cam_image_corner, percentage=0.2):
            bb = None
    
    print("Target xyz: ", target[:3])
    print("Calculated bounding box: ", bb)

     # sanity check
    xyz = cam.xyz_from_bb(bb)
    print("Recovered target xyz: ", xyz)

    bb = torch.tensor(bb, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)  # add batch dimension


    model = YOLOBox()
    model.model.eval()


    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_dataloader = load_dataset(path=dataset_path, batch_size=1, shuffle=True, drop_last=False, num_workers=0)

    patch = torch.rand(1, 1, 45, 80).to(device) * 255.
    patch.requires_grad_(True)

    T = gen_random_patch_coords().unsqueeze(0).to(device)

    opt = torch.optim.Adam([patch], lr=1e-3)

    loss = torch.tensor(0.).to(device)
    # loss.requires_grad_(True)

    epochs = 10

    for i in range(epochs):
        epoch_loss = 0.
        for step, (batch, _) in enumerate(train_dataloader):



            opt.zero_grad()

            manipulated_image = project_patch(
                    patches=patch, 
                    T_matrices=T,
                    images=batch.to(device)
                )

            manipulated_image.clamp_(0., 1.)

            # print("Manipulated image shape: ", manipulated_image.shape, manipulated_image.min().item(), manipulated_image.max().item())

            manipulated_image = torch.nn.functional.interpolate(manipulated_image, size=(320, 640), mode='bilinear', align_corners=False)
            # gray to rgb
            manipulated_image = manipulated_image.repeat_interleave(3, dim=1)

            # print("Manipulated image shape after repeat: ", manipulated_image.shape)

            prediction = model(manipulated_image) # of shape (B, 1, 4)

            # print("Prediction: ", prediction, prediction.shape)
            # print("BB: ", bb, bb.shape)

            loss = F.pairwise_distance(prediction, bb, p=2).mean()

            loss.backward()
            opt.step()

            epoch_loss += loss.item()


            patch.data.clamp_(0., 1.)

        print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader)}")
