import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from camera import Camera
from yolo_bounding import YOLOBox

from util import load_model, load_dataset
    
def normalize_yaw_t(yaw):
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cam_image_corner = np.array([0., 0., 160., 96.], dtype=np.float32)

cam = Camera(path='camera_calibration.yaml')
camera_intrinsic = cam.camera_intrinsic
camera_extrinsic = cam.camera_extrinsic

yolo = YOLOBox()
yolo.model.eval()

frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
frontnet.eval()


dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
train_dataloader = load_dataset(path=dataset_path, batch_size=1, shuffle=True, drop_last=False, num_workers=0, IMRC=False)

radius = 0.3

for step, (batch, gt) in enumerate(train_dataloader):
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
    print("yolo shape: ", prediction_yolo.shape)

    x, y, z, yaw = frontnet(batch)
    prediction_frontnet = torch.stack([x, y, z, yaw], dim=1)
    prediction_frontnet = prediction_frontnet.squeeze(2)
    print("frontnet shape: ", prediction_frontnet.shape)


    print("GT: ", gt)
    print("YOLO: ", prediction_yolo)
    print("Frontnet: ", prediction_frontnet)

    distance_yolo = torch.norm(gt[:, :3] - prediction_yolo[:, :3], dim=1)
    distance_frontnet = torch.norm(gt[:, :3] - prediction_frontnet[:, :3], dim=1)
    print("Distance YOLO: ", distance_yolo)
    print("Distance Frontnet: ", distance_frontnet)

    distance_yolo_to_frontnet = torch.norm(prediction_yolo[:, :3] - prediction_frontnet[:, :3], dim=1)
    print("Distance YOLO to Frontnet: ", distance_yolo_to_frontnet)

    # gt_yaw = normalize_yaw_t(gt[:, 3])
    # pred_yaw_yolo = normalize_yaw_t(prediction_yolo[:, 3])
    # yaw_error_yolo = normalize_yaw_t(torch.abs(gt_yaw - pred_yaw_yolo))

    # print("Yaw Error YOLO: ", yaw_error_yolo)

    # prediction_frontnet = frontnet(batch*255.)

