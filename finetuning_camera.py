import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from camera import Camera
from yolo_bounding import YOLOBox

from util import load_model, load_dataset

import matplotlib.pyplot as plt

from tqdm import trange
    
def normalize_yaw_t(yaw):
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))

def _as_tensor(v, device=None, dtype=None):
        return v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device, dtype=dtype)

def create_camera_intrinsic(fx, fy, ox, oy):
    # ensure device and dtype match inputs (prefer fx if it's a tensor)
    if isinstance(fx, torch.Tensor):
        device = fx.device
        dtype = fx.dtype
    else:
        device = torch.device('cpu')
        dtype = torch.get_default_dtype()

    fx = _as_tensor(fx, device=device, dtype=dtype).reshape(1)
    fy = _as_tensor(fy, device=device, dtype=dtype).reshape(1)
    ox = _as_tensor(ox, device=device, dtype=dtype).reshape(1)
    oy = _as_tensor(oy, device=device, dtype=dtype).reshape(1)

    z0 = torch.zeros(1, device=device, dtype=dtype)
    z1 = torch.ones(1, device=device, dtype=dtype)

    row0 = torch.stack([fx, z0, ox], dim=0)
    row1 = torch.stack([z0, fy, oy], dim=0)
    row2 = torch.stack([z0, z0, z1], dim=0)

    intrinsic = torch.stack([row0, row1, row2], dim=0)
    return intrinsic

def create_camera_extrinsic(roll, pitch, yaw, tx, ty, tz):
    # ensure device and dtype match inputs (prefer roll if it's a tensor)
    if isinstance(roll, torch.Tensor):
        device = roll.device
        dtype = roll.dtype
    else:
        device = torch.device('cpu')
        dtype = torch.get_default_dtype()

    roll = _as_tensor(roll, device=device, dtype=dtype)
    pitch = _as_tensor(pitch, device=device, dtype=dtype)
    yaw = _as_tensor(yaw, device=device, dtype=dtype)
    tx = _as_tensor(tx, device=device, dtype=dtype)
    ty = _as_tensor(ty, device=device, dtype=dtype)
    tz = _as_tensor(tz, device=device, dtype=dtype)

    cr = torch.cos(roll).unsqueeze(0); sr = torch.sin(roll).unsqueeze(0)
    cp = torch.cos(pitch).unsqueeze(0); sp = torch.sin(pitch).unsqueeze(0)
    cy = torch.cos(yaw).unsqueeze(0); sy = torch.sin(yaw).unsqueeze(0)

    z0 = torch.zeros(1, device=device, dtype=dtype)
    z1 = torch.ones(1, device=device, dtype=dtype)



    # print("Shapes: ", cr.shape, sr.shape, cp.shape, sp.shape, cy.shape, sy.shape)
    # print(z0.shape, z1.shape)

    # build rotation matrices using stack/operations so gradients are preserved
    R_row1 = torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=0).squeeze()
    R_row2 = torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=0).squeeze()
    R_row3 = torch.stack([-sp, cp*sr, cp*cr], dim=0).squeeze()

    R = torch.stack([R_row1, R_row2, R_row3], dim=0)

    # t = torch.stack([tx, ty, tz], dim=0).reshape(3, 1)

    row0 = torch.stack([R[0, 0], R[0, 1], R[0, 2], tx], dim=0)
    row1 = torch.stack([R[1, 0], R[1, 1], R[1, 2], ty], dim=0)
    row2 = torch.stack([R[2, 0], R[2, 1], R[2, 2], tz], dim=0)
    row3 = torch.stack([z0, z0, z0, z1], dim=0).squeeze(1)

    extrinsic = torch.stack([row0, row1, row2, row3], dim=0)
    # print("Extrinsic shape: ", extrinsic.shape)
    # print(extrinsic.grad_fn)
    return extrinsic

def tensor_xyz_from_bb(bb, fx, fy, ox, oy, extrinsic, radius):
        
    center = (bb[1] + bb[3]) / 2

    # build rays without in-place ops to keep autograd graph
    a1x = (bb[0] - ox) / fx
    a1y = (center - oy) / fy
    a2x = (bb[2] - ox) / fx
    a2y = (center - oy) / fy

    one = torch.ones_like(a1x)
    a1 = torch.stack((a1x, a1y, one))
    a2 = torch.stack((a2x, a2y, one))

    # normalize rays
    a1_norm = torch.linalg.norm(a1)
    a2_norm = torch.linalg.norm(a2)

    # distance on the circle of radius (differentiable)
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=bb.device, dtype=bb.dtype))
    # radius can be float; lift to tensor to keep device/dtype
    rad = radius if isinstance(radius, torch.Tensor) else torch.tensor(radius, device=bb.device, dtype=torch.float32)
    cosang = torch.dot(a1, a2) / (a1_norm * a2_norm + 1e-12)
    distance = sqrt2 * rad / torch.sqrt(1.0 - cosang + 1e-12)

    # central ray and xyz
    ac = (a1 + a2) * 0.5
    xyz = distance * ac / (torch.linalg.norm(ac) + 1e-12)


    new_xyz = (torch.linalg.inv(extrinsic) @ torch.cat((xyz, torch.ones(1, device=xyz.device, dtype=xyz.dtype))))[:3]
    return new_xyz

def batch_xyz_from_boxes(boxes, fx, fy, ox, oy, extrinsic, radius):
    # build per-sample outputs and stack to preserve graph
    outs = []
    for i in range(boxes.shape[0]):
        coords = tensor_xyz_from_bb(boxes[i], fx, fy, ox, oy, extrinsic, radius)
        yaw = torch.atan2(coords[1], coords[0])
        outs.append(torch.cat((coords, yaw.unsqueeze(0))))
    return torch.stack(outs, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cam_image_corner = np.array([0., 0., 160., 96.], dtype=np.float32)

cam = Camera(path='camera_calibration.yaml', device=device)
camera_intrinsic = torch.tensor(cam.camera_intrinsic, device=device)
camera_extrinsic = torch.tensor(cam.camera_extrinsic, device=device)

print("Camera Intrinsic: ", camera_intrinsic)
print("Camera Extrinsic: ", camera_extrinsic)

fx = camera_intrinsic[0][0].to(device=device).clone()
fy = camera_intrinsic[1][1].to(device=device).clone()
ox = camera_intrinsic[0][2].to(device=device).clone()
oy = camera_intrinsic[1][2].to(device=device).clone()

fx = fx.requires_grad_(True)
fy = fy.requires_grad_(True)
ox = ox.requires_grad_(True)
oy = oy.requires_grad_(True)

# calc roll, pitch, yaw from rotation matrix
R = camera_extrinsic[:3, :3]
sy = torch.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
singular = sy < 1e-6
if not singular:
    original_roll = torch.atan2(R[2,1], R[2,2])
    original_pitch = torch.atan2(-R[2,0], sy)
    original_yaw = torch.atan2(R[1,0], R[0,0])
else:
    original_roll = torch.atan2(-R[1,2], R[1,1])
    original_pitch = torch.atan2(-R[2,0], sy)
    original_yaw = 0
original_tx = camera_extrinsic[0, 3]
original_ty = camera_extrinsic[1, 3]
original_tz = camera_extrinsic[2, 3]

print("Original roll, pitch, yaw: ", original_roll.item(), original_pitch.item(), original_yaw.item())
norm_original_roll = normalize_yaw_t(original_roll)
norm_original_pitch = normalize_yaw_t(original_pitch)
norm_original_yaw = normalize_yaw_t(original_yaw)
print("Normalized Original roll, pitch, yaw: ", norm_original_roll.item(), norm_original_pitch.item(), norm_original_yaw.item())

# sanity check:
reconstructed_extrinsic = create_camera_extrinsic(original_roll, original_pitch, original_yaw, original_tx, original_ty, original_tz)
print("Reconstructed Extrinsic: ", reconstructed_extrinsic)
l1_distance_extrinsic = torch.abs(reconstructed_extrinsic - camera_extrinsic).sum()
print("L1 distance extrinsic (should be close to 0): ", l1_distance_extrinsic.item())

roll = norm_original_roll.to(device=device).clone()
pitch = norm_original_pitch.to(device=device).clone()
yaw = norm_original_yaw.to(device=device).clone()
tx = original_tx.to(device=device).clone()
ty = original_ty.to(device=device).clone()
tz = original_tz.to(device=device).clone()

roll = roll.requires_grad_(True)
pitch = pitch.requires_grad_(True)
yaw = yaw.requires_grad_(True)
tx = tx.requires_grad_(True)
ty = ty.requires_grad_(True)
tz = tz.requires_grad_(True)

print("Initial roll, pitch, yaw: ", roll.item(), pitch.item(), yaw.item())
print("Initial tx, ty, tz: ", tx.item(), ty.item(), tz.item())

yolo = YOLOBox()
yolo.model.eval()

frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
frontnet.eval()


dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
train_dataloader = load_dataset(path=dataset_path, batch_size=64, shuffle=True, drop_last=False, num_workers=0, IMRC=False)

radius = torch.tensor(0.25, device=device, requires_grad=True)
softmax_mult = torch.tensor(20.0, device=device, requires_grad=True)

opt = torch.optim.Adam([radius, fx, fy, ox, oy, roll, pitch, yaw, tx, ty, tz, softmax_mult], lr=1e-3)
# opt = torch.optim.Adam([radius, roll, pitch, yaw, tx, ty, tz, softmax_mult], lr=1e-2)

scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, end_factor=1., total_iters=1000)

best_loss = torch.inf

for i in trange(500):

    epoch_loss_yolo = 0.0

    # print("Camera intrinsic old: ", camera_intrinsic)
    # print("Camera extrinsic old: ", camera_extrinsic)

    # print("Current fx, fy, ox, oy: ", fx.item(), fy.item(), ox.item(), oy.item())
    # print("Current roll, pitch, yaw, tx, ty, tz: ", roll.item(), pitch.item(), yaw.item(), tx.item(), ty.item(), tz.item())

    epoch_loss_frontnet = 0.0

    angular_errors_yolo = []

    for step, (batch, gt) in enumerate(train_dataloader):
        opt.zero_grad()
        batch = batch.to(device)
        # print("batch min, max: ", batch.min().item(), batch.max().item())
        gt = gt.to(device)

        # print(gt)

        scaled_images = F.interpolate(batch/255., size=(320, 640), mode='bilinear', align_corners=False)
        scaled_images = scaled_images.repeat_interleave(3, dim=1)

        scaled_images.clamp_(0.0, 1.0)

        bounding_box = yolo(scaled_images, softmax_mult=softmax_mult).squeeze(1)
        bounding_box[:, [0, 2]] *= (160.0 / 640.0)  # x coords
        bounding_box[:, [1, 3]] *= (96.0 / 320.0)   # y coords

        # current_intrinsic = create_camera_intrinsic(fx, fy, ox, oy)
        current_extrinsic = create_camera_extrinsic(roll, pitch, yaw, tx, ty, tz)

        prediction_yolo = batch_xyz_from_boxes(bounding_box, fx, fy, ox, oy, current_extrinsic, radius) #  xyzyaw from bounding box
        # print("yolo shape: ", prediction_yolo.shape)

        fr_x, fr_y, fr_z, fr_yaw = frontnet(batch)
        prediction_frontnet = torch.stack([fr_x, fr_y, fr_z, fr_yaw], dim=1)
        prediction_frontnet = prediction_frontnet.squeeze(2)
        # print("frontnet shape: ", prediction_frontnet.shape)


        # print("GT: ", gt)
        # print("YOLO: ", prediction_yolo)
        # print("Frontnet: ", prediction_frontnet)
        # if torch.all_ze
        # print("Shapes GT: ", gt.shape, " YOLO: ", prediction_yolo.shape, " Frontnet: ", prediction_frontnet.shape)
        distance_yolo = torch.norm(gt[:, :3] - prediction_yolo[:, :3], p=2, dim=1)
        distance_frontnet = torch.norm(gt[:, :3] - prediction_frontnet[:, :3], p=2, dim=1)
        # else:
        #     distance_yolo = torch.norm(prediction_frontnet[:, :3] - prediction_yolo[:, :3], p=2, dim=1)
        # gt = torch.where(torch.nonzero(gt, ) gt, prediction_frontnet)
        # print(gt)
        # print("Distance YOLO mean: ", distance_yolo.mean().item())

        # print("Original roll, pitch, yaw: ", original_roll.item(), original_pitch.item(), original_yaw.item())
        # print("Current roll, pitch, yaw: ", roll.item(), pitch.item(), yaw.item())

        # if original_roll > roll:
        #     angular_error_roll = normalize_yaw_t(original_roll - roll)
        # else:
        #     angular_error_roll = normalize_yaw_t(roll - original_roll)
        # if original_pitch > pitch:
        #     angular_error_pitch = normalize_yaw_t(original_pitch - pitch)
        # else:
        #     angular_error_pitch = normalize_yaw_t(pitch - original_pitch)
        # if original_yaw > yaw:
        #     angular_error_yaw = normalize_yaw_t(original_yaw - yaw)
        # else:
        #     angular_error_yaw = normalize_yaw_t(yaw - original_yaw)
        # translation_current = torch.stack([tx, ty, tz], dim=0)
        # translation_original = torch.stack([original_tx, original_ty, original_tz], dim=0)
        # # print("Shapes translation: ", translation_current.shape, translation_original.shape)
        # distance_extrinsic = torch.norm(translation_current - translation_original, p=2)

        # # print("Angular Errors (roll, pitch, yaw): ", angular_error_roll.item(), angular_error_pitch.item(), angular_error_yaw.item())
        # # print("Distance Extrinsic: ", distance_extrinsic.item())

        # extrinsic_error = angular_error_roll + angular_error_pitch + angular_error_yaw + distance_extrinsic
        # # print("Extrinsic Error: ", extrinsic_error.item())


        # current_intrinsic = create_camera_intrinsic(fx, fy, ox, oy).squeeze(2)
        # # print("Current Intrinsic: ", current_intrinsic.shape, camera_intrinsic.shape)
        # l2_intrinsic = torch.norm(current_intrinsic - camera_intrinsic, p=2)
        # # l2_extrinsic = torch.norm(current_extrinsic - camera_extrinsic, p=2)
        # # print("L2 Intrinsic: ", l2_intrinsic.item())

        # with torch.no_grad():
        #     for gt_yaw, pred_yaw in zip(gt[:, 3], prediction_yolo[:, 3]):
        #         norm_gt_yaw = normalize_yaw_t(gt_yaw)
        #         norm_pred_yaw = normalize_yaw_t(pred_yaw)
        #         angular_error = normalize_yaw_t(norm_gt_yaw - norm_pred_yaw)
        #         angular_errors_yolo.append(angular_error.item())

        # # loss = l2_intrinsic + l2_extrinsic + distance_yolo.mean()
        # # minimize the top 5 distances
        # # sorted_distances, _ = torch.sort(distance_yolo)
        # #loss = l2_intrinsic + l2_extrinsic + sorted_distances[:5].mean()
        # #loss = extrinsic_error + sorted_distances[:5].mean()
        # loss = l2_intrinsic + extrinsic_error + distance_yolo.mean()
        # print("Loss: ", loss.item())

        loss = distance_yolo.mean()

        # l1_distance_intrinsic = torch.abs(fx - camera_intrinsic[0][0]) + torch.abs(fy - camera_intrinsic[1][1]) + torch.abs(ox - camera_intrinsic[0][2]) + torch.abs(oy - camera_intrinsic[1][2])
        # l1_distance_extrinsic = torch.abs(camera_extrinsic[0][3] - tx) + torch.abs(camera_extrinsic[1][3] - ty) + torch.abs(camera_extrinsic[2][3] - tz)

        loss.backward()
        opt.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

            best_fx = fx.clone().detach().item()
            best_fy = fy.clone().detach().item()
            best_ox = ox.clone().detach().item()
            best_oy = oy.clone().detach().item()
            best_roll = roll.clone().detach().item()
            best_pitch = pitch.clone().detach().item()
            best_yaw = yaw.clone().detach().item()
            best_tx = tx.clone().detach().item()
            best_ty = ty.clone().detach().item()
            best_tz = tz.clone().detach().item()
            best_radius = radius.clone().detach().item()
            best_softmax_mult = softmax_mult.clone().detach().item()

        epoch_loss_yolo += loss.item()
        epoch_loss_frontnet += distance_frontnet.mean().item()

    if (i+1) % 10 == 0:
        print(f"Epoch {i+1}, Loss YOLO: {epoch_loss_yolo / (len(train_dataloader))}, Radius: {radius.item()}, Softmax Mult: {softmax_mult.item()}")
        print("Distance YOLO: ", distance_yolo.mean().item())
        print("Distance Frontnet: ", distance_frontnet.mean().item())
        print("Frontnet Epoch Loss: ", epoch_loss_frontnet / (len(train_dataloader)))
        current_intrinsic = create_camera_intrinsic(fx.clone().detach(), fy.clone().detach(), ox.clone().detach(), oy.clone().detach())
        print(f"Intrinsic: {current_intrinsic}")

        current_extrinsic = create_camera_extrinsic(roll.clone().detach(), pitch.clone().detach(), yaw.clone().detach(), tx.clone().detach(), ty.clone().detach(), tz.clone().detach())
        print(f"Extrinsic: {current_extrinsic}")
        # print(f"Example prediction YOLO: {prediction_yolo[0]}, GT: {gt[0]}, Example prediction Frontnet: {prediction_frontnet[0]}")
        # print(f"Mean Angular Error YOLO: {np.mean(angular_errors_yolo)}")
        angular_errors_yolo = []
        # plot a figure with 2 subfigures, to the left, take one example image and draw the bounding box from yolo
        # to the right, draw a 3D representation of the position and orientation from yolo and the ground truth
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        img = batch[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        axs[0].imshow(img)
        box = bounding_box[0].detach().cpu().numpy()
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)
        axs[0].set_title('YOLO Bounding Box')

        axs[1].set_title('2D Position and Orientation')
        axs[1].set_xlim([-2., 2.])
        axs[1].set_ylim([-1., 3.5])
        axs[1].set_xlabel('Y')
        axs[1].set_ylabel('X')
        gt_pos = gt[0, :3].detach().cpu().numpy()
        pred_pos = prediction_yolo[0, :3].detach().cpu().numpy()
        frontnet_pos = prediction_frontnet[0, :3].detach().cpu().numpy()
        # add the yaw as an arrow
        gt_yaw = gt[0, 3].item()
        pred_yaw = prediction_yolo[0, 3].item()
        frontnet_yaw = prediction_frontnet[0, 3].item()
        axs[1].arrow(gt_pos[1], gt_pos[0], 0.2*np.sin(gt_yaw), 0.2*np.cos(gt_yaw), head_width=0.05, head_length=0.1, fc='g', ec='g')
        axs[1].arrow(pred_pos[1], pred_pos[0], 0.2*np.sin(pred_yaw), 0.2*np.cos(pred_yaw), head_width=0.05, head_length=0.1, fc='r', ec='r')
        axs[1].arrow(frontnet_pos[1], frontnet_pos[0], 0.2*np.sin(frontnet_yaw), 0.2*np.cos(frontnet_yaw), head_width=0.05, head_length=0.1, fc='b', ec='b')
        axs[1].plot(gt_pos[1], gt_pos[0], 'go', label='Ground Truth')
        axs[1].plot(pred_pos[1], pred_pos[0], 'ro', label='YOLO Prediction')
        axs[1].plot(frontnet_pos[1], frontnet_pos[0], 'bo', label='Frontnet Prediction')
        axs[1].legend()
        # plt.show()
        plt.savefig(f'comparison_yolo_frontnet_step_{i}.png')
        plt.close(fig)
    # distance_frontnet = torch.norm(gt[:, :3] - prediction_frontnet[:, :3], dim=1)
    # print("Distance YOLO: ", distance_yolo)
    # print("Distance Frontnet: ", distance_frontnet)

    # distance_yolo_to_frontnet = torch.norm(prediction_yolo[:, :3] - prediction_frontnet[:, :3], dim=1)
    # print("Distance YOLO to Frontnet: ", distance_yolo_to_frontnet)

    # gt_yaw = normalize_yaw_t(gt[:, 3])
    # pred_yaw_yolo = normalize_yaw_t(prediction_yolo[:, 3])
    # yaw_error_yolo = normalize_yaw_t(torch.abs(gt_yaw - pred_yaw_yolo))

    # print("Yaw Error YOLO: ", yaw_error_yolo)

    # prediction_frontnet = frontnet(batch*255.)

# save best parameters
torch.save({
    'fx': best_fx,
    'fy': best_fy,
    'ox': best_ox,
    'oy': best_oy,
    'roll': best_roll,
    'pitch': best_pitch,
    'yaw': best_yaw,
    'tx': best_tx,
    'ty': best_ty,
    'tz': best_tz,
    'radius': best_radius,
    'softmax_mult': best_softmax_mult,
}, 'best_finetuned_camera_yolo_frontnet.pth')