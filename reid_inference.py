import torch
import cv2
import numpy as np

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
image_1 = cv2.resize(image_1, (128, 256))
# transform to tensor
image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float()
print(image_1.shape) # torch.Size([1, 3, 128, 64])

image_2 = cv2.imread("reid/Market-1501/gt_bbox/0001_c1s1_002301_00.jpg")
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
image_2 = cv2.resize(image_2, (128, 256))
image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float()
print(image_2.shape) # torch.Size([1, 3, 128, 64])


features_image_1 = model(image_1)
features_image_2 = model(image_2)

# calculate cosine similarity
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = cos(features_image_1, features_image_2)
print(similarity)