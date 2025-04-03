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
        if not torch.is_tensor(feature_1):
            feature_1 = torch.from_numpy(feature_1)
        if not torch.is_tensor(feature_2):
            feature_2 = torch.from_numpy(feature_2)
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


    overall_detected_persons = 0
    max_people = 3
    max_feature_budget = 20
    feature_dim = 256
    feature_history  = np.zeros((max_people, max_feature_budget, feature_dim))

    camera = cv2.VideoCapture('data/dji_fly_20250121_135614_0011_1737464791745_video.mp4')
    ret, _ = camera.read()
    camera.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    num_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    camera.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    # ret, frame = camera.read()


    for frame_idx in trange(int(num_frames)):
        ret, frame = camera.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (640, 320))
        image = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.

        nms_boxes = get_bounding_boxes(image, yolo)[0]
        # print(nms_boxes.shape)

        

        for detected_person_idx in range(nms_boxes.shape[0]):
            person_box = cut_bounding_box(image, nms_boxes[detected_person_idx, :4])
            if person_box.shape[1] * person_box.shape[2] <= 20: # person box should be larger than 20 pixels
                continue
            person_features = get_features(person_box, reid)

            det_img = cv2.rectangle(cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), (int(nms_boxes[detected_person_idx, 0]), int(nms_boxes[detected_person_idx, 1])), (int(nms_boxes[detected_person_idx, 2]), int(nms_boxes[detected_person_idx, 3])), (0, 255, 0), 2)

            if detected_person_idx >= overall_detected_persons:
                print("Adding new person to dictionary")
                feature_history[overall_detected_persons, 0] = person_features.detach().numpy()
                det_img = cv2.putText(det_img, str(overall_detected_persons), (int(nms_boxes[overall_detected_persons, 0]), int(nms_boxes[overall_detected_persons, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                overall_detected_persons += 1
                # print(feature_history[:overall_detected_persons, :2, 0])
            else:
                # print("Detected person: ", detected_person_idx)
                cosine_similarities = np.zeros((max_people, max_feature_budget))
                for person_idx in range(max_people):
                    for feature_idx in range(max_feature_budget):
                        cosine_similarities[person_idx, feature_idx] = feature_similarity(person_features, feature_history[person_idx, feature_idx])
                

                calculated_person_idx = np.unravel_index(cosine_similarities.argmax(), cosine_similarities.shape)[0]


                feature_history[calculated_person_idx, frame_idx % max_feature_budget] = person_features.detach().cpu().numpy()

                det_img = cv2.putText(det_img, str(calculated_person_idx), (int(nms_boxes[detected_person_idx, 0]), int(nms_boxes[detected_person_idx, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Detection", det_img)
            cv2.waitKey(1)

    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

