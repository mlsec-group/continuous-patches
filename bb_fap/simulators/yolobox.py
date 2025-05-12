# Code adapted from Natalie Huang (@natalieh235)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

import rowan

from torchvision.ops import generalized_box_iou_loss

import yaml

RADIUS = 0.3
IM_HEIGHT = 96
IM_WIDTH = 160
person_width = 10

TENSOR_DEFAULT_WIDTH = 640
BATCH_SIZE = 1
IMSIZE = (96, 160)
SOFTMAX_MULT = 15.

# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q

class Camera:
    def __init__(self, path):
        self.parse(path)
        
    def parse(self, path):
        with open(path) as f:
            camera_config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.camera_intrinsic = np.array(camera_config['camera_matrix'])
        self.distortion_coeffs = np.array(camera_config['dist_coeff'])
        
        rvec = np.array(camera_config['rvec'])
        tvec = camera_config['tvec']
        self.make_extrinsic(rvec, tvec)

        self.fx = np.array(self.camera_intrinsic)[0][0]
        self.fy = np.array(self.camera_intrinsic)[1][1]
        self.ox = np.array(self.camera_intrinsic)[0][2]
        self.oy = np.array(self.camera_intrinsic)[1][2]
    
    def make_extrinsic(self, rvec, tvec):
        self.camera_extrinsic = np.zeros((4,4))
        self.camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
        self.camera_extrinsic[:3, 3] = tvec
        self.camera_extrinsic[-1, -1] = 1.

        self.camera_extrinsic_tens = torch.tensor(self.camera_extrinsic, dtype=torch.float32)


    # compute relative position of center of patch in camera frame
    def xyz_from_bb(self, bb):
        
        # get pixels for bb side center
        P1 = np.array([bb[0],(bb[1] + bb[3])/2])
        P2 = np.array([bb[2],(bb[1] + bb[3])/2])

        # print(P1, P2)

        # get rays for pixels
        a1 = np.array([(P1[0]-self.ox)/self.fx, (P1[1]-self.oy)/self.fy, 1.0])
        a2 = np.array([(P2[0]-self.ox)/self.fx, (P2[1]-self.oy)/self.fy, 1.0])

        # normalize rays
        a1_norm = np.linalg.norm(a1)
        a2_norm = np.linalg.norm(a2)

        # get the distance    
        distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))

        ac = (a1+a2)/2

        # get the position
        xyz = distance*ac/np.linalg.norm(ac)
        new_xyz = (np.linalg.inv(self.camera_extrinsic) @ [*xyz, 1])[:3]
        return new_xyz

    # compute relative position of center of patch in camera frame
    def tensor_xyz_from_bb(self, bb):
        
        # printd('bounding box', bb.grad_fn)
        center = (bb[1] + bb[3])/2

        # get rays for pixels
        a1 = torch.ones(3)
        a1[0] = (bb[0]-self.ox)/self.fx
        a1[1] = (center-self.oy)/self.fy

        a2 = torch.ones(3)
        a2[0] = (bb[2]-self.ox)/self.fx
        a2[1] = (center-self.oy)/self.fy

        # printd('a1', a1.grad_fn)

        # normalize rays
        a1_norm = torch.linalg.norm(a1)
        a2_norm = torch.linalg.norm(a2)

        # printd('a1 nnorm', a1_norm.grad_fn)

        # get the distance    
        distance = (np.sqrt(2)*RADIUS)/(torch.sqrt(1-torch.dot(a1,a2)/(a1_norm*a2_norm)))

        # printd('distance', distance.grad_fn)

        # get central ray
        ac = (a1+a2)/2

        # get the position
        xyz = distance*ac/torch.linalg.norm(ac)

        new_xyz = (torch.linalg.inv(self.camera_extrinsic_tens) @ torch.cat((xyz, torch.ones(1))))[:3]
        return new_xyz
    
    def batch_xyz_from_boxes(self, boxes):
        batch_size = boxes.shape[0]
        xyzs = torch.zeros((batch_size, 3), device=boxes.device)
        # boxes is a tensor of size (B, 4)
        for i in range(batch_size):
            # print('boxes[i]', boxes[i], boxes[i].shape)
            coords = self.tensor_xyz_from_bb(boxes[i])
            # printd('coords', coords.grad_fn)
            xyzs[i] = coords

        return xyzs

    def point_from_xyz(self, coords):
        camera_frame = self.camera_extrinsic @ coords
        image_frame = self.camera_intrinsic @ camera_frame[:3]
        u, v, w = image_frame
        img_x = int(np.round(u/w, decimals=0))
        img_y = int(np.round(v/w, decimals=0))
        return (img_x, img_y)

class YOLOBox(nn.Module):
    conf = 0.4  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    softmax_mult = 15.

    def __init__(self, cam_config='simulators/camera_calibration.yaml'):
        super().__init__()

        # load model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", autoshape=False)
        self.model.eval()

        # camera
        self.cam = Camera(cam_config)

    def forward(self, og_imgs, show_imgs=False):
        imgs = og_imgs / 255.0

        imgs = torch.repeat_interleave(imgs, 3, dim=1)

        # yolo wants size (320, 640)
        resized_inputs = torch.nn.functional.interpolate(imgs, size=(TENSOR_DEFAULT_WIDTH//2, TENSOR_DEFAULT_WIDTH), mode="bilinear")
        output = self.model(resized_inputs)

        scale_factor = imgs.size()[3] / TENSOR_DEFAULT_WIDTH
        boxes, scores = self.extract_boxes_and_scores(output[0])

        # take a weighted average of the boxes
        soft_scores = F.softmax(scores * SOFTMAX_MULT, dim=1)
        soft_scores = soft_scores.unsqueeze(1)
        selected_boxes = torch.bmm(soft_scores, boxes) * scale_factor

        # # printd('selected ', selected_boxes.shape, selected_boxes.grad_fn)

        # debugging
        if show_imgs:
            # true best boxes
            highest_score_idxs = torch.argmax(scores, 1)

            for i in range(min(len(og_imgs), 10)):
                # print(og_imgs.shape)
                og_img = og_imgs[i].clone().detach().cpu().numpy()
                og_img = np.moveaxis(og_img, 0, -1)
                og_img = cv2.cvtColor(og_img,cv2.COLOR_GRAY2RGB)

                true_best_box = boxes[i, highest_score_idxs[i]] * scale_factor

                xmin, ymin, xmax, ymax = true_best_box.detach().cpu().numpy().astype(int)
                cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 0.), 1)

                selected_box = selected_boxes[i][0]
 
                xmin, ymin, xmax, ymax = int(selected_box[0]), int(selected_box[1]), int(selected_box[2]), int(selected_box[3])
                cv2.rectangle(og_img, (xmin, ymin), (xmax, ymax), (255, 0, 255.), 1)

                cv2.imwrite(f'person_new_{i}.png', og_img)

        xyzs = self.cam.batch_xyz_from_boxes(selected_boxes.squeeze(1))  # only using squeeze() here will cause all dimensions to be deleted if there's only one input image
        return xyzs
    
    def extract_boxes_and_scores(self, yolo_output):
        # Extract bounding boxes and scores from YOLO output
        # This function will be specific to the YOLO model's output format

        boxes = self.xywh2xyxy(yolo_output[:, :, :4])
        scores = yolo_output[:, :, 4] *  yolo_output[:, :, 5] # multiply obj score by person confidence
        return boxes, scores

    # taken from ultralytics yolo
    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
