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

from torchvision.ops import nms

out_1 = yolo(image_1)[0]
# print(out_1.shape)
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
scores = torch.where(scores < 0.5, torch.zeros_like(scores), scores)
print(scores.shape)
print(torch.sort(scores, descending=True)[:20])

best_box = out_1[0, torch.argmax(scores), :4]

best_box_xyxy = xywh2xyxy(best_box)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

# print(image_1.min(), image_1.max())
# print(image_1[0].shape)
# plt.imshow(image_1[0].permute(1, 2, 0).numpy() / 255.) 

fig, ax = plt.subplots(1)
ax.imshow(image_1[0].permute(1, 2, 0).numpy())
# for box in nms_boxes:
#     x1, y1, x2, y2 = box
#     rect = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
x1, y1, x2, y2 = best_box_xyxy
rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.show()


# print(objectness.min(), objectness.max())
# print(person_scores.min(), person_scores.max())

# scores = objectness * person_scores
# print(scores.min(), scores.max())
# print(torch.sort(scores, descending=True)[:20])



boxes = out_1[0, :, :4]

boxes_xyxy = xywh2xyxy(boxes)

boxes_normalized = torch.nn.functional.sigmoid(boxes_xyxy)

# # print(boxes_normalized.min(), boxes_normalized.max())

# print(scores.shape, boxes_xyxy.shape)

nms_indices = nms(boxes_xyxy, scores, iou_threshold=0.45)

nms_boxes = boxes_xyxy[nms_indices]

print(nms_boxes.shape)

# # draw bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image_1[0].permute(1, 2, 0).numpy())
for box in nms_boxes:
    x1, y1, x2, y2 = box
    rect = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


plt.show()

