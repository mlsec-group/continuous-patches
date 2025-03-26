import torch
import cv2
import numpy as np

# taken from ultralytics yolo
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# load model
model_path = "reid/resnet50_market1501_aicity156.pth"
model = torch.load(model_path, weights_only=False)
model.eval()

# ultralytics nms
from torchvision.ops import nms

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
        #     break  # time limit exceeded

    return output



# label description: https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#reidentification-market1501-format
'''
The root directory of the dataset contains sub-directories for training, testing, and query. 
Each sub-directory has the cropped images of different identities. For example, the image 0001_c1s1_01_00.jpg is from the first sequence s1 of camera c1. 
01 indicates the first frame in the sequence c1s1. 0001 is the unique ID assigned to the object. The contents after the third _ are ignored. 
There is no label file required.
'''
# file paths look like this: reid/Market-1501/gt_bbox/0001_c1s1_001051_00.jpg



# https://developer.ridgerun.com/wiki/index.php/ReID_and_Target_Reassociation_using_NVIDIA_Deepstream#Re-ID
'''
The Re-ID similarity between a detector object and a target is the cosine similarity between the detector object’s Re-ID feature 
and its nearest neighbor in the target’s feature gallery, whose value is in range [0.0, 1.0]. Specifically, each Re-ID feature in the 
target’s gallery takes the dot product with the detector object’s Re-ID feature. The maximum of all the dot products is the similarity score. 
'''


# load two images
image_1 = cv2.imread("reid/Market-1501/gt_bbox/0001_c1s1_001051_00.jpg")
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
# print(image_1.shape) # (128, 64, 3) h, w, c
#resize to match model input (3, 256, 128)
image_1 = cv2.resize(image_1, (256, 128))
# transform to tensor
image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float()
print(image_1.shape) # torch.Size([1, 3, 128, 256])

image_2 = cv2.imread("reid/Market-1501/gt_bbox/0001_c1s1_002301_00.jpg")
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
image_2 = cv2.resize(image_2, (256, 128))
image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float()
print(image_2.shape) # torch.Size([1, 3, 128, 256])


features_image_1 = model(image_1)
features_image_2 = model(image_2)

# calculate cosine similarity
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = cos(features_image_1, features_image_2)
print(similarity)


# load two dji images
image_1 = cv2.imread("data/dji_images/0045.jpg")
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_1 = cv2.resize(image_1, (640, 320))       # yolo has input shape (bs, 3, 320, 640) (bs, channels, height, width)
print(image_1.shape)
image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float() / 255.
print(image_1.shape)

image_2 = cv2.imread("data/dji_images/0175.jpg")
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
image_2 = cv2.resize(image_2, (640, 320))
image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float() /255.
# print(image_2.min(), image_2.max())

# load yolov5 model for bounding box detection
yolo = torch.hub.load("ultralytics/yolov5", "yolov5n", autoshape=False)
yolo.eval()


out_1 = yolo(image_1)[0]
# print(out_1.shape)
boxes = out_1[0, :, :4]
print(boxes.shape)
objectness = out_1[0, :, 4]
person_scores = out_1[0, :, 5]

# print(torch.argmax(person_scores))
# print(torch.argmax(objectness))

labels = torch.argmax(out_1[0, :, 5:], dim=1)
print(labels.shape, labels[:10])

# print(best_box)


# print(objectness.shape, person_scores.shape)

scores = (objectness * person_scores)
print(scores.min(), scores.max(), scores.shape)
# replace entries with 0 if score < 0.5
# scores = torch.where(scores < 0.5, torch.zeros_like(scores), scores)
# print(scores.shape)
# print(torch.sort(scores, descending=True)[:20])

# remove boxes with score < 0.5
# boxes = torch.where(scores[..., None] < 0.5, torch.zeros_like(boxes), boxes)
# print(boxes.shape)

# best_box = out_1[0, torch.argmax(scores), :4]

# best_box_xyxy = xywh2xyxy(best_box)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

# print(image_1.min(), image_1.max())
# print(image_1[0].shape)
# plt.imshow(image_1[0].permute(1, 2, 0).numpy() / 255.) 

# fig, ax = plt.subplots(1)
# ax.imshow(image_1[0].permute(1, 2, 0).numpy())
# # for box in nms_boxes:
# #     x1, y1, x2, y2 = box
# #     rect = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
# #     ax.add_patch(rect)
# x1, y1, x2, y2 = best_box_xyxy
# rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)

# plt.show()


# print(objectness.min(), objectness.max())
# print(person_scores.min(), person_scores.max())

# scores = objectness * person_scores
# print(scores.min(), scores.max())
# print(torch.sort(scores, descending=True)[:20])




boxes_xyxy = xywh2xyxy(boxes)

# boxes_normalized = torch.nn.functional.sigmoid(boxes_xyxy)

# # print(boxes_normalized.min(), boxes_normalized.max())

# print(scores.shape, boxes_xyxy.shape)

# nms_indices = nms(boxes_xyxy, scores, iou_threshold=0.45)
nms_boxes = non_max_suppression(out_1, conf_thres=0.5, iou_thres=0.45, classes=[0], max_det=200) # 0 is person class
# A list of length batch_size, where each element is a tensor of shape (num_boxes, 6 + num_masks) 
# containing the kept boxes, with columns (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).

print(nms_boxes[0].shape)
print(nms_boxes)



# print(nms_boxes[0][:4].min(), nms_boxes[0][:4].max())


# nms_boxes = boxes_xyxy[nms_indices]

# print(nms_boxes.shape)

# # # draw bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image_1[0].permute(1, 2, 0).numpy())
for box in nms_boxes[0]:
    x1, y1, x2, y2, class_idx, mask_1  = box
    print(x1, y1, x2, y2)
    print(class_idx, mask_1)
    rect = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


plt.show()




