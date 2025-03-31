import torch
import cv2
import numpy as np

from tqdm import trange
from collections import deque

# ultralytics nms
from util import non_max_suppression, xywh2xyxy

def get_features(person_box, reid):
    # print("Person box shape: ", person_box.shape)
    if len(person_box.shape) < 4:
        person_box = person_box.unsqueeze(0)
    if person_box.shape[2:] != (256, 128):
        person_box = torch.nn.functional.interpolate(person_box, size=(256, 128), mode='bilinear')
    # print("Person box shape after interpolation: ", person_box.shape)
    return reid(person_box)


def feature_similarity(feature_1, feature_2, metric='cosine'):
    if metric == 'cosine':
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(feature_1, feature_2)
        return similarity
    else:
        "Invalid metric (yet). Please use 'cosine'."

def get_bounding_boxes(image, yolo, class_idx=0):
    out = yolo(image)[0]
    nms_boxes = non_max_suppression(out, conf_thres=0.5, iou_thres=0.45, classes=[class_idx], max_det=200) # 0 is person class
    
    # list of len(batch size), each element is a tensor of shape (num detections, 6 + num_masks) containing the kept boxes, with columns (x1, y1, x2, y2, confidence, class, [mask1, mask2, ...])
    return nms_boxes
        
def cut_bounding_box(image, bb):
    x1, y1, x2, y2 = bb.int()
    person_box = image[0, :, y1:y2, x1:x2]
    return person_box


if __name__ == "__main__":
    # load reid
    reid_path = "reid/resnet50_market1501_aicity156.pth"
    reid = torch.load(reid_path, weights_only=False)
    reid.eval()

    yolo = torch.hub.load("ultralytics/yolov5", "yolov5n", autoshape=False)
    yolo.eval()



    image = cv2.imread("/home/pia/dji_videos/flightspace_human/0538.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 320))
    image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.

    nms_boxes = get_bounding_boxes(image_t, yolo)[0]

    max_people = 5
    max_feature_budget = 20
    feature_dim = 256
    cosine_similarity_matrix  = np.zeros((max_people, max_feature_budget, feature_dim))

    camera = cv2.VideoCapture('data/dji_fly_20250121_135614_0011_1737464791745_video.mp4')
    ret, _ = camera.read()
    ret, frame = camera.read()

    # print(frame.shape)



    # person_ids = []
    # for detected_person in range(nms_boxes.shape[0]):
    #     person_box = cut_bounding_box(image_t, nms_boxes[detected_person, :4])
    #     person_features = get_features(person_box, reid)
    #     person_ids.append(person_features)

    # print(image.shape)

    # nms_boxes = get_bounding_boxes(image_t, yolo)[0]
    # print(nms_boxes.shape)

    # boxes = []
    # features = []
    # for detected_person in range(nms_boxes.shape[0]):
    #     cv2.rectangle(image, (int(nms_boxes[detected_person, 0]), int(nms_boxes[detected_person, 1])), (int(nms_boxes[detected_person, 2]), int(nms_boxes[detected_person, 3])), (0, 255, 0), 2)
    #     person_box = cut_bounding_box(image_t, nms_boxes[detected_person, :4])
    #     boxes.append(person_box)
    #     person_features = get_features(person_box, reid)
    #     features.append(person_features)

    # print("Similarity: ", feature_similarity(features[0], features[1]))
    # cv2.imshow("Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # if cv2.waitKey(0) == 27:
    #     cv2.destroyAllWindows()

    
    # image_2 = cv2.imread("/home/pia/dji_videos/flightspace_human/0536.jpg")
    # image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    # image_2 = cv2.resize(image_2, (640, 320))
    # image_2_t = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float() / 255.

    # nms_boxes_2 = get_bounding_boxes(image_2_t, yolo)[0]
    # print(nms_boxes_2.shape)

    # boxes_2 = []
    # features_2 = []
    # for detected_person in range(nms_boxes_2.shape[0]):
    #     cv2.rectangle(image_2, (int(nms_boxes_2[detected_person, 0]), int(nms_boxes_2[detected_person, 1])), (int(nms_boxes_2[detected_person, 2]), int(nms_boxes_2[detected_person, 3])), (0, 255, 0), 2)
    #     person_box = cut_bounding_box(image_2_t, nms_boxes_2[detected_person, :4])
    #     boxes_2.append(person_box)
    #     person_features = get_features(person_box, reid)
    #     features_2.append(person_features)

    # print("Similarity: ", feature_similarity(features_2[0], features_2[1]))
    # cv2.imshow("Detection", cv2.cvtColor(image_2, cv2.COLOR_RGB2BGR))
    # if cv2.waitKey(0) == 27:
    #     cv2.destroyAllWindows()

    # # print("Similarity image_0,box_0 to image_1, box_0: ", feature_similarity(features[0], features_2[0]))
    # # print("Similarity image_0,box_0 to image_1, box_1: ", feature_similarity(features[0], features_2[1]))
    # # print("Similarity image_0,box_1 to image_1, box_0: ", feature_similarity(features[1], features_2[0]))
    # # print("Similarity image_0,box_1 to image_1, box_1: ", feature_similarity(features[1], features_2[1]))

    # similarity_matrix = np.zeros((len(features), len(features_2)))
    # for i, feature_1 in enumerate(features):
    #     for j, feature_2 in enumerate(features_2):
    #         similarity_matrix[i, j] = feature_similarity(feature_1, feature_2)
        
    # print(similarity_matrix)



    # camera = cv2.VideoCapture('data/dji_fly_20250121_135614_0011_1737464791745_video.mp4')
    # ret, img = camera.read()

    # camera.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    # num_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    # camera.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    # # print(num_frames)

    # person_dictionary = {}   # will safe a queue of features for last 5 occurences of that person in frames
    # # code will be similar to Nearest Neighbour Distance Metric from https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification/blob/main/deep_sort/nn_matching.py#L99

    # detection_images = []

    # for frame_idx in trange(int(num_frames)):
    #     ret, frame = camera.read()
    #     if ret:
    #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         image = cv2.resize(image, (640, 320))
    #         image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.

    #         # print(image.shape)

    #         nms_boxes = get_bounding_boxes(image, yolo)[0]  # 0 because it's a single image always
    #         for detected_person in range(nms_boxes.shape[0]):
    #             # print(detected_person, nms_boxes.shape)
    #             det_image = cv2.rectangle((image.permute(0, 2, 3, 1).numpy()[0]), (int(nms_boxes[detected_person, 0]), int(nms_boxes[detected_person, 1])), (int(nms_boxes[detected_person, 2]), int(nms_boxes[detected_person, 3])), (0, 255, 0), 2)
    #             # print(det_image.shape)
    #             # detection_images.append(cv2.cvtColor(det_image, cv2.COLOR_RGB2BGR))
    #             person_box = cut_bounding_box(image, nms_boxes[detected_person, :4])
    #             # if person_box is very small, skip
    #             if person_box.shape[1] * person_box.shape[2] <= 20: # leq 20 pixels
    #                 continue

    #             person_features = get_features(person_box, reid)
                
    #             # check if dictionary is empty
    #             if not person_dictionary:
    #                 person_dictionary[detected_person] = deque([person_features], maxlen=40)
    #                 print(f"New person added: {len(person_dictionary) -1}")
    #                 detected_person_idx = 0
    #             elif len(person_dictionary) < nms_boxes.shape[0]:
    #                 person_dictionary[len(person_dictionary)] = deque([person_features], maxlen=40)
    #                 print(f"New person added: {len(person_dictionary) -1}")
    #                 detected_person_idx = len(person_dictionary) - 1

    #             else:
    #                 # for each person in dictionary, calculate similarity matrix for all features in deque
    #                 # create empty similarity matrix of shape (len(person_dictionary), 5)
    #                 similarity_matrix = np.zeros((len(person_dictionary), 40))
    #                 for person_idx, feature_deque in enumerate(person_dictionary.values()):
    #                     for feature_idx, current_saved_feature in enumerate(feature_deque):
    #                         similarity_matrix[person_idx, feature_idx] = feature_similarity(person_features, current_saved_feature)

    #                 # if nms_boxes.shape[0] > 1:
    #                 #     print("Similarity matrix: ", similarity_matrix)
                    
    #                 # find the person with the highest similarity
    #                 highest_similarity = np.max(similarity_matrix)
    #                 if highest_similarity < 0.2: # that's the threshold they use here: https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification/blob/main/demo.py#L70
    #                     # update dictionary with new person
    #                     # get current num of persons in dictionary
    #                     num_persons = len(person_dictionary)
    #                     person_dictionary[num_persons] = deque([person_features], maxlen=40)
    #                     print("New person added: ", num_persons)
    #                     detected_person_idx = len(person_dictionary) - 1
    #                 else:
    #                     # get the index of the person with the highest similarity
    #                     person_idx, feature_idx = np.where(similarity_matrix == highest_similarity)
    #                     # print(person_idx, feature_idx)
    #                     # print("Detected person: ", person_idx[0])
    #                     person_dictionary[person_idx[0]].append(person_features)
    #                     detected_person_idx = person_idx[0]
                    
    #                 # add number to bounding box in det_image
    #                 det_image = cv2.putText(det_image, str(person_idx), (int(nms_boxes[detected_person, 0]), int(nms_boxes[detected_person, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #                 detection_images.append(cv2.cvtColor(det_image, cv2.COLOR_RGB2BGR))
    #                 cv2.imshow("Detection", detection_images[-1])
    #                 cv2.waitKey(1)
            



    # # save video
    # detection_video = cv2.VideoWriter("DJI_Detect.avi", 0, 1, (640,320))
    # for frame in detection_images:
    #     detection_video.write(frame.astype(np.uint8))
    # detection_video.release()        
        
                    # for person_id, feature_deque in person_dictionary.items():

    # # load two dji images
    # image_1 = cv2.imread("data/dji_images/0045.jpg")
    # image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    # image_1 = cv2.resize(image_1, (640, 320))       # yolo has input shape (bs, 3, 320, 640) (bs, channels, height, width)
    # # print(image_1.shape)
    # image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float() / 255.
    # # print(image_1.shape)

    # image_2 = cv2.imread("data/dji_images/0175.jpg")
    # image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    # image_2 = cv2.resize(image_2, (640, 320))
    # image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float() /255.
    # # print(image_2.min(), image_2.max())

    # # get bounding boxes
    # nms_boxes_1 = get_bounding_boxes(image_1, yolo)
    # nms_boxes_2 = get_bounding_boxes(image_2, yolo)

    # # cut bounding box out of image
    # person_box_1 = cut_bounding_box(image_1, nms_boxes_1[0][0, :4])
    # person_box_2 = cut_bounding_box(image_2, nms_boxes_2[0][0, :4])

    # # print(person_box_1.shape, person_box_2.shape)

    # # calc reid features
    # similarity = feature_similarity(person_box_1, person_box_2, reid)

    # print("Similarity: ", similarity)
