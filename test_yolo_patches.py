import yaml
import numpy as np
import matplotlib.pyplot as plt
from util import load_dataset
from camera import Camera
import argparse

import torch
from yolo_bounding import YOLOBox
from attack_minimal_single import T_matrix, normalize_yaw_t, calc_heading_vec, project_patch
import torch.nn.functional as F

import os

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
    check = False
    # i = 0
    while not check:
        random_sf = np.random.uniform(0.2, 1.1)
        random_upper_left_x = np.random.uniform(0, 160 - (80 * random_sf), 1)
        random_upper_left_y = np.random.uniform(0, 96 - (45 * random_sf), 1)
        random_lower_right_x = random_upper_left_x + (80 * random_sf)
        random_lower_right_y = random_upper_left_y + (45 * random_sf)

        upper_left = [random_upper_left_x[0], random_upper_left_y[0]]
        lower_right = [random_lower_right_x[0], random_lower_right_y[0]]
        check = check_bb_overlap((*upper_left, *lower_right), [0., 0., 160., 96.], threshold=0.5)
        # i += 1
        # print("BB coords: ", (*upper_left, *lower_right))
        # print(f"Trying to generate random patch coords, attempt {i}, success: {check}")

    tx_min = upper_left[0]
    ty_min = upper_left[1]

    T = torch.eye(3, device=device, dtype=torch.float32)  # Identity transformation matrix
    T[:2, :2] *= random_sf  # Scale down to half the size of the patch
    T[0, 2] = tx_min
    T[1, 2] = ty_min

    return T

def check_bb_overlap(rect1, rect2, threshold=0.5):
    """
    Return True if at least `threshold` fraction of rect1's area is inside rect2.

    rect1, rect2: (x1, y1, x2, y2) (corners can be in any order)
    threshold: fraction in [0,1]
    """
    # Normalize coordinates (ensure x1<=x2, y1<=y2)
    ax1, ay1, ax2, ay2 = rect1
    bx1, by1, bx2, by2 = rect2

    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    # Intersection coordinates
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # Intersection area
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    # Area of rect1
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

    if area_a <= 0.0:
        # Degenerate rect1: treat as not meeting any positive threshold
        return threshold == 0.0

    fraction_inside = inter_area / area_a
    return fraction_inside >= threshold

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

    os.makedirs('debugging/output_patches', exist_ok=True)

    bb = None

    while bb is None:
        random_target_x = np.random.uniform(0.,1.5, 1)
        random_target_y = np.random.uniform(-1,1, 1)
        random_target_z = np.random.uniform(-0.5,0.5, 1)
        
        random_target_yaw = np.random.uniform(-0.3, 0.3, 1)


        target = np.hstack((random_target_x, random_target_y, random_target_z, random_target_yaw))
        
        bb = bb_from_xyz(camera_intrinsic, camera_extrinsic, target[:3], RADIUS)

        # check if bb has overlap with image
        if bb is not None and not check_bb_overlap(bb, cam_image_corner, threshold=0.4):
            bb = None
    
    print("Target xyz: ", target[:3])
    print("Calculated bounding box: ", bb)

     # sanity check
    xyz = cam.xyz_from_bb(bb)
    print("Recovered target xyz: ", xyz)

    bb = torch.tensor(bb, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)  # add batch dimension
    # scale to image of size 320 x 640
    bb[:, :, [0, 2]] *= (640.0 / 160.0)
    bb[:, :, [1, 3]] *= (320.0 / 96.0)
    print("Scaled bounding box for 640x320 image: ", bb)

    model = YOLOBox()
    model.model.eval()

    from util import load_model
    frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
    frontnet.eval()


    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    patch = torch.rand(1, 1, 45, 80).to(device)
    patch.requires_grad_(True)

    T = gen_random_patch_coords().unsqueeze(0).to(device)

    opt = torch.optim.Adam([patch], lr=5e-2)

    loss = torch.tensor(0.).to(device)
    # loss.requires_grad_(True)

    epochs = 1

    for i in range(epochs):
        epoch_loss = 0.
        for step, (batch, _) in enumerate(train_dataloader):



            opt.zero_grad()

            batch = batch / 255.

            # print("Batch shape: ", batch.shape, batch.min().item(), batch.max().item())

            manipulated_image = project_patch(
                    patches=patch, 
                    T_matrices=T,
                    images=batch.to(device)
                )

            manipulated_image.clamp_(0., 1.)

            # print("Manipulated image shape: ", manipulated_image.shape, manipulated_image.min().item(), manipulated_image.max().item())

            manipulated_image_y = torch.nn.functional.interpolate(manipulated_image, size=(320, 640), mode='bilinear', align_corners=False)
            # gray to rgb
            manipulated_image_y = manipulated_image_y.repeat_interleave(3, dim=1)

            # print("Manipulated image shape after repeat: ", manipulated_image.shape)

            prediction = model(manipulated_image_y) # of shape (B, 1, 4) xyxy format

            # with torch.no_grad():
            #     print("Prediction shape: ", prediction.shape)
            #     pred_c = prediction.clone().detach().squeeze(1)  # (B, 4)
            #     pred_c[:, [0, 2]] *= (160.0 / 640.0)  # x coords
            #     pred_c[:, [1, 3]] *= (96.0 / 320.0)   # y coords

            #     pred_c = model.cam.batch_xyz_from_boxes(pred_c, RADIUS) #  xyzyaw from bounding box

            #     x, y, z, yaw = frontnet(manipulated_image*255.)
            #     prediction_frontnet = torch.stack([x, y, z, yaw])
            #     prediction_frontnet = prediction_frontnet.squeeze(2).mT

            #     print("Example frontnet prediction: ", prediction_frontnet[0])
            #     print("Example YOLO prediction: ", pred_c[0])

            #     distance = torch.dist(prediction_frontnet[:, :3], pred_c[:, :3], p=2)
            #     print("Distance between frontnet and yolo predictions: ", distance.mean().item())

            # print("Prediction shape: ", prediction.shape)


            # print("Prediction: ", prediction[0], prediction.shape)
            # print("BB: ", bb, bb.shape)

            #loss = F.smooth_l1_loss(prediction, bb.repeat(manipulated_image.shape[0],1,1))
            # loss = F.mse_loss(prediction, bb.repeat(manipulated_image.shape[0],1,1))

            loss.backward()
            opt.step()

            epoch_loss += loss.item()


            patch.data.clamp_(0., 1.)


        if (i+1) % 1 == 0:
            print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader)}")
            # sanity check loss
            # with torch.no_grad():
            #     print(F.mse_loss(prediction, bb))
            #     print(F.l1_loss(prediction, bb))
            #     print(F.smooth_l1_loss(prediction, bb))
            #     print(F.pairwise_distance(prediction, bb).mean())
            # save intermediate patch
            np_patch = patch.detach().cpu().squeeze().numpy() # 45 x 80
            # print(np_patch.shape, np_patch.min(), np_patch.max())


            # save patch (np_patch in [0,1])
            plt.imsave(f'debugging/output_patches/patch_epoch_{i+1}.png', np_patch, cmap='gray', vmin=0, vmax=1)
            plt.close()

            # visualize on a sample image
            manipulated_sample = project_patch(
                patches=patch,
                T_matrices=T,
                images=batch[0:1].to(device)
                )

            # print("Manipulated sample shape: ", manipulated_sample.shape, manipulated_sample.min().item(), manipulated_sample.max().item())

            manipulated_sample = torch.nn.functional.interpolate(manipulated_sample, size=(320, 640), mode='bilinear', align_corners=False)
            # gray to rgb
            manipulated_sample = manipulated_sample.repeat_interleave(3, dim=1)

            # --- get model prediction for the sample before converting to numpy ---
            with torch.no_grad():
                pred_bb_tensor = model(manipulated_sample)  # expect shape (1,1,4)
            pred_bb_np = pred_bb_tensor.detach().cpu().squeeze().numpy()  # [x_min, y_min, x_max, y_max]

            # convert to numpy image for saving/visualization
            manipulated_sample = manipulated_sample.detach().cpu().numpy().squeeze() # 3 x 320 x 640
            img = manipulated_sample.transpose(1, 2, 0).copy()  # H x W x 3, values in [0,1]

            # retrieve ground-truth bb (was stored on device as (1,1,4)) and scale to current image size
            bb_np = bb.detach().cpu().squeeze().numpy()  # [x_min, y_min, x_max, y_max]

            # draw ground-truth (green) and prediction (red) boxes on img (H=320, W=640)
            gt = np.nan_to_num(bb_np).astype(float)
            pred = np.nan_to_num(pred_bb_np).astype(float)

            def to_int_clip(coords, H, W):
                x1, y1, x2, y2 = coords
                x1, x2 = int(np.clip(round(min(x1, x2)), 0, W - 1)), int(np.clip(round(max(x1, x2)), 0, W - 1))
                y1, y2 = int(np.clip(round(min(y1, y2)), 0, H - 1)), int(np.clip(round(max(y1, y2)), 0, H - 1))
                return x1, y1, x2, y2

            def draw_box(img, x1, y1, x2, y2, color, width=2):
                H, W = img.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
                for w in range(width):
                    xl, xr = x1 + w, x2 - w
                    yt, yb = y1 + w, y2 - w
                    if xl <= xr:
                        img[yt, xl:xr+1] = color
                        img[yb, xl:xr+1] = color
                    if yt <= yb:
                        img[yt:yb+1, xl] = color
                        img[yt:yb+1, xr] = color

            H, W = img.shape[:2]
            gt_x1, gt_y1, gt_x2, gt_y2 = to_int_clip(gt, H, W)
            pr_x1, pr_y1, pr_x2, pr_y2 = to_int_clip(pred, H, W)

            draw_box(img, gt_x1, gt_y1, gt_x2, gt_y2, color=np.array([0.0, 1.0, 0.0]), width=2)
            draw_box(img, pr_x1, pr_y1, pr_x2, pr_y2, color=np.array([1.0, 0.0, 0.0]), width=2)

            plt.imsave(f'debugging/output_patches/manipulated_sample_epoch_{i+1}.png', img, vmin=0, vmax=1)
            plt.close()
