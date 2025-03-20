import MNN
import MNN.numpy as np
import MNN.cv as cv2

import cv2 as cv2_original

config = {}
config['precision'] = 'normal'  # inference precision: normal, low, high, lowBF')
config['backend'] = 'CPU' # CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI')
config['numThread'] = 4

# scale = 2

rt = MNN.nn.create_runtime_manager((config,))
net = MNN.nn.load_module_from_file('/home/pia/bb_FAP/HoverAir/models/mnn/elan_yolo_192x192_20230718_830.mnn', [], [], runtime_manager=rt)


original_image = cv2.imread('/home/pia/bb_FAP/HoverAir/data/our_img/0000.jpg')
ih, iw, _ = original_image.shape
length = max((ih, iw))
scale = length / 192
print(scale)
image = cv2.resize(original_image, (192, 192), 0., 0., cv2.INTER_LINEAR, -1, [0., 0., 0.], [1./255., 1./255., 1./255.])
print(image.shape)

input_var = np.expand_dims(image, 0)
input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
output_var = net.forward(input_var)

output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
print(output_var.shape)

output_var = np.array(output_var)

output_var = output_var.reshape(-1, 7)
print(output_var.shape)

has_object = output_var[:, 4] > 0.1
idx = MNN.expr.where(has_object)
output_var = output_var[idx]

print(output_var.shape)

output_var = output_var.reshape(7, -1)
print(output_var.shape)

cx = output_var[0]
# print(cx.shape)

cy = output_var[1]

w = output_var[2]
h = output_var[3]

# conf = output_var[4]
# probs = output_var[5:]

x0 = cx - w * 0.5
y0 = cy - h * 0.5
x1 = cx + w * 0.5
y1 = cy + h * 0.5

boxes = np.stack([x0, y0, x1, y1], axis=-1)


probs = output_var[4:]

scores = np.max(probs, axis=0)
class_ids = np.argmax(probs, axis=0)
result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
result_boxes = boxes[result_ids]
result_scores = scores[result_ids]
result_class_ids = class_ids[result_ids]

print(result_boxes[:10] * 20)

for i in range(len(result_boxes)):
    x0, y0, x1, y1 = result_boxes[i].read_as_tuple()
    y0 = int(y0 * scale)
    y1 = int(y1 * scale)
    x0 = int(x0 * scale)
    x1 = int(x1 * scale)
    # print(result_class_ids[i])
    cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
cv2.imwrite('res.jpg', original_image)






# print(boxes.shape)

# print(boxes[:10])

# print(conf)
# print(probs)

# top_scores = np.max(probs, axis=0)
# print(top_scores)
# class_ids = np.argmax(probs, axis=0)
# print(class_ids)

# result_ids = MNN.expr.nms(boxes, conf, 100, 0.45, 0.25)
# print(result_ids.shape)

# result_boxes = boxes[result_ids]
# result_scores = top_scores[result_ids]
# result_class_ids = class_ids[result_ids]

# for i in range(len(boxes)):
#         x0, y0, x1, y1 = boxes[i].read_as_tuple()
#         y0 = int(y0 * scale)
#         # print(y0)
#         y1 = int(y1 * scale)
#         x0 = int(x0 * scale)
#         x1 = int(x1 * scale)
#         # print(result_class_ids[i])
#         cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
# cv2.imwrite('res.jpg', original_image)