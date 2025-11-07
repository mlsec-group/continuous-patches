import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from camera import Camera
from yolo_bounding import YOLOBox

from util import load_model, load_dataset
    
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

def tensor_xyz_from_bb(bb, ox, oy, fx, fy, coeffs_extrinsic, radius):
        
    # printd('bounding box', bb.grad_fn)
    center = (bb[1] + bb[3])/2

    # get rays for pixels
    a1 = torch.ones(3, device=bb.device)
    a1[0] = (bb[0]-ox)/fx
    a1[1] = (center-oy)/fy

    a2 = torch.ones(3, device=bb.device)
    a2[0] = (bb[2]-ox)/fx
    a2[1] = (center-oy)/fy

    # printd('a1', a1.grad_fn)

    # normalize rays
    a1_norm = torch.linalg.norm(a1)
    a2_norm = torch.linalg.norm(a2)

    printd('a1 nnorm', a1_norm.grad_fn)

    # get the distance    
    distance = (np.sqrt(2)*radius)/(torch.sqrt(1-torch.dot(a1,a2)/(a1_norm*a2_norm)))

    # printd('distance', distance.grad_fn)

    # get central ray
    ac = (a1+a2)/2

    # get the position
    xyz = distance*ac/torch.linalg.norm(ac)

    camera_extrinsic = create_camera_extrinsic(*coeffs_extrinsic)

    new_xyz = (torch.linalg.inv(camera_extrinsic) @ torch.cat((xyz, torch.ones(1, device=xyz.device))))[:3]
    return new_xyz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cam_image_corner = np.array([0., 0., 160., 96.], dtype=np.float32)

cam = Camera(path='camera_calibration.yaml', device=device)
camera_intrinsic = torch.tensor(cam.camera_intrinsic, device=device)
camera_extrinsic = torch.tensor(cam.camera_extrinsic, device=device)

print("Camera Intrinsic: ", camera_intrinsic)
print("Camera Extrinsic: ", camera_extrinsic)

fx = camera_intrinsic[0][0].to(device=device).requires_grad_(True)
fy = camera_intrinsic[1][1].to(device=device).requires_grad_(True)
ox = camera_intrinsic[0][2].to(device=device).requires_grad_(True)
oy = camera_intrinsic[1][2].to(device=device).requires_grad_(True)

# calc roll, pitch, yaw from rotation matrix
R = camera_extrinsic[:3, :3]
sy = torch.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
singular = sy < 1e-6
if not singular:
    roll = torch.atan2(R[2,1], R[2,2])
    pitch = torch.atan2(-R[2,0], sy)
    yaw = torch.atan2(R[1,0], R[0,0])
else:
    roll = torch.atan2(-R[1,2], R[1,1])
    pitch = torch.atan2(-R[2,0], sy)
    yaw = 0
tx = camera_extrinsic[0, 3]
ty = camera_extrinsic[1, 3]
tz = camera_extrinsic[2, 3]

# sanity check:
reconstructed_extrinsic = create_camera_extrinsic(roll, pitch, yaw, tx, ty, tz)
print("Reconstructed Extrinsic: ", reconstructed_extrinsic)
l1_distance_extrinsic = torch.abs(reconstructed_extrinsic - camera_extrinsic).sum()
print("L1 distance extrinsic (should be close to 0): ", l1_distance_extrinsic.item())

roll = roll.to(device=device).requires_grad_(True)
pitch = pitch.to(device=device).requires_grad_(True)
yaw = yaw.to(device=device).requires_grad_(True)
tx = torch.tensor(tx, device=device).requires_grad_(True)
ty = torch.tensor(ty, device=device).requires_grad_(True)
tz = torch.tensor(tz, device=device).requires_grad_(True)

yolo = YOLOBox()
yolo.model.eval()

frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
frontnet.eval()


dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
train_dataloader = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0, IMRC=False)

radius = torch.tensor(0.3, device=device, requires_grad=True)

opt = torch.optim.Adam([radius, fx, fy, ox, oy, roll, pitch, yaw, tx, ty, tz], lr=1e-3)

best_loss = torch.inf

for i in range(1000):

    epoch_loss_yolo = 0.0

    for step, (batch, gt) in enumerate(train_dataloader):
        opt.zero_grad()
        batch = batch.to(device)
        # print("batch min, max: ", batch.min().item(), batch.max().item())
        gt = gt.to(device)

        scaled_images = F.interpolate(batch/255., size=(320, 640), mode='bilinear', align_corners=False)
        scaled_images = scaled_images.repeat_interleave(3, dim=1)

        scaled_images.clamp_(0.0, 1.0)

        prediction_yolo = yolo(scaled_images).squeeze(1)
        prediction_yolo[:, [0, 2]] *= (160.0 / 640.0)  # x coords
        prediction_yolo[:, [1, 3]] *= (96.0 / 320.0)   # y coords

        prediction_yolo = cam.batch_xyz_from_boxes(prediction_yolo, radius) #  xyzyaw from bounding box
        # print("yolo shape: ", prediction_yolo.shape)

        # x, y, z, yaw = frontnet(batch)
        # prediction_frontnet = torch.stack([x, y, z, yaw], dim=1)
        # prediction_frontnet = prediction_frontnet.squeeze(2)
        # print("frontnet shape: ", prediction_frontnet.shape)


        # print("GT: ", gt)
        # print("YOLO: ", prediction_yolo)
        # print("Frontnet: ", prediction_frontnet)

        distance_yolo = torch.norm(gt[:, :3] - prediction_yolo[:, :3], dim=1)

        current_intrinsic = create_camera_intrinsic(fx, fy, ox, oy)
        current_extrinsic = create_camera_extrinsic(roll, pitch, yaw, tx, ty, tz)

        l1_intrinsic = torch.abs(current_intrinsic - camera_intrinsic).sum()
        l1_extrinsic = torch.abs(current_extrinsic - camera_extrinsic).sum()

        loss = 0.1*(l1_intrinsic + l1_extrinsic) + distance_yolo.mean()

        # l1_distance_intrinsic = torch.abs(fx - camera_intrinsic[0][0]) + torch.abs(fy - camera_intrinsic[1][1]) + torch.abs(ox - camera_intrinsic[0][2]) + torch.abs(oy - camera_intrinsic[1][2])
        # l1_distance_extrinsic = torch.abs(camera_extrinsic[0][3] - tx) + torch.abs(camera_extrinsic[1][3] - ty) + torch.abs(camera_extrinsic[2][3] - tz)

        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'fx': fx.clone().detach().item(),
                'fy': fy.clone().detach().item(),
                'ox': ox.clone().detach().item(),
                'oy': oy.clone().detach().item(),
                'roll': roll.clone().detach().item(),
                'pitch': pitch.clone().detach().item(),
                'yaw': yaw.clone().detach().item(),
                'tx': tx.clone().detach().item(),
                'ty': ty.clone().detach().item(),
                'tz': tz.clone().detach().item(),
                'radius': radius.clone().detach().item()
            }, 'best_finetuned_camera_yolo_frontnet.pth')

        epoch_loss_yolo += loss.item()

    if (i+1) % 10 == 0:
        print(f"Epoch {i+1}, Loss YOLO: {epoch_loss_yolo / (len(train_dataloader))}, Radius: {radius.item()}")
        print(f"Intrinsic: {current_intrinsic}")
        print(f"Extrinsic: {current_extrinsic}")
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

    