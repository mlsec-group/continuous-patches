import numpy as np
import os
import subprocess
import shlex



# inference
# network_path = 'HoverAir/models/snpe/elan_yolo_192x192_20230718_830_ptq_quant.dlc'
# input_path = 'HoverAir/data/192x192/raw_list.txt'
# output_path = 'HoverAir/data/192x192/output/'
# metadata_path = 'HoverAir/data/192x192/metadata/'
# os.makedirs(output_path, exist_ok=True)
# os.makedirs(metadata_path, exist_ok=True)

# command = f"snpe-net-run --container {network_path} --input_list {input_path} --output_dir={output_path} --storage_dir={metadata_path} --debug"
# subprocess.run(shlex.split(command))


# eval results for one image
# output-0.raw
file_path = 'HoverAir/data/192x192/output/Result_0/output-0.raw'
output_0 = np.fromfile(file_path, dtype=np.float32)

file_path = 'HoverAir/data/192x192/output/Result_0/output-1.raw'
output_1 = np.fromfile(file_path, dtype=np.float32)

file_path = 'HoverAir/data/192x192/output/Result_0/output-2.raw'
output_2 = np.fromfile(file_path, dtype=np.float32)

# reshape output_0 according to config
batch_size = 3
output_channels = 24
output_height = 24
output_width = 7
scale_x_y = 2.

# output_0 = output_0.reshape(batch_size, output_channels, output_height, output_width)

# output_1 = output_1.reshape(batch_size, 12, 12, output_width)

# output_2 = output_2.reshape(batch_size, 6, 6, output_width)





# 
# print(output_0.shape)
# Print a few sample values (first 3 grid cells in the first batch)
# print("Sample bounding box values:")
# print(output_0[0, :3, :3, :])  # First batch, first 3x3 grid cells



# import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def decode_yolo_output(yolo_output, anchors, strides, img_size=192, conf_threshold=0.3):
#     """
#     Decodes YOLO output into bounding boxes.
    
#     yolo_output: list of numpy arrays [(3, 24, 24, 7), (3, 12, 12, 7), (3, 6, 6, 7)]
#     anchors: numpy array of shape (3,3,2)  # 3 scales, 3 anchors per scale, (width, height)
#     strides: [8, 16, 32]  # Downsampling factors
#     img_size: 192
#     conf_threshold: 0.3
#     """
#     bboxes = []

#     for scale_idx, output in enumerate(yolo_output):  # Loop through 3 scales
#         num_anchors, grid_H, grid_W, _ = output.shape
#         stride = strides[scale_idx]
#         anchor_set = anchors[scale_idx]  # Shape (3,2) -> 3 anchors at this scale
#         print(anchor_set)

#         # Generate grid coordinates
#         grid_x, grid_y = np.meshgrid(np.arange(grid_W), np.arange(grid_H), indexing='xy')
#         grid_x = np.expand_dims(grid_x, axis=0)  # Shape (1, grid_H, grid_W)
#         grid_y = np.expand_dims(grid_y, axis=0)  # Shape (1, grid_H, grid_W)

#         # Apply sigmoid to cx, cy, w, h, confidence
#         output[..., 0:5] = sigmoid(output[..., 0:5])

#         # Compute bounding box center
#         bx = (output[:, :, :, 0] * 2. - (2. - 1)+ grid_x) * stride  # Center x
#         by = (output[:, :, :, 1] * 2. - (2. - 1) + grid_y) * stride  # Center y

#         # Compute width and height using anchors
#         bw = (output[:, :, :, 2] * 2) ** 2 * anchor_set[:, 0, None, None]  # Width
#         bh = (output[:, :, :, 3] * 2) ** 2 * anchor_set[:, 1, None, None]  # Height

#         # Convert to (x_min, y_min, x_max, y_max)
#         x_min = bx - bw / 2
#         y_min = by - bh / 2
#         x_max = bx + bw / 2
#         y_max = by + bh / 2

#         # Get confidence and class probabilities
#         conf = output[:, :, :, 4]
#         class_probs = output[:, :, :, 5:]  # Shape (3, grid_H, grid_W, num_classes)
#         class_id = np.argmax(class_probs, axis=-1)  # Select highest class probability
#         class_score = np.max(class_probs, axis=-1)

#         # Flatten arrays
#         x_min, y_min, x_max, y_max = x_min.flatten(), y_min.flatten(), x_max.flatten(), y_max.flatten()
#         conf, class_id, class_score = conf.flatten(), class_id.flatten(), class_score.flatten()

#         print(x_min[:5], y_min[:5])

#         # Filter boxes based on confidence threshold
#         # mask = conf > conf_threshold
#         # for i in np.where(mask)[0]:
#         #     bboxes.append((x_min[i], y_min[i], x_max[i], y_max[i], conf[i], class_id[i], class_score[i]))
#         bboxes.append((x_min, y_min, x_max, y_max, conf, class_id, class_score))
#         break

#     return bboxes  # List of (x_min, y_min, x_max, y_max, confidence, class_id, class_score)


# # Example usage
# # Assuming `reshaped_data` contains 3 output scales: [(3, 24, 24, 7), (3, 12, 12, 7), (3, 6, 6, 7)]
# # yolo_output = [reshaped_data[0], reshaped_data[1], reshaped_data[2]]
# yolo_output = [output_0, output_1, output_2]

# # Define **your** anchors per scale
# anchors = np.array([
#     [[9.29, 13.18], [17.18, 24.45], [25.31, 44.04]],  # Small scale (24x24)
#     [[27.57, 110.71], [43.93, 56.54], [51.07, 96.06]],  # Medium scale (12x12)
#     [[53.32, 158.74], [88.83, 104.43], [112.46, 169.59]]  # Large scale (6x6)
# ])

# # Strides based on feature map sizes
# strides = [8, 16, 32]

# # Decode the bounding boxes
# bboxes = decode_yolo_output(yolo_output, anchors, strides)
# print(bboxes.shape)

# Print results
# for box in bboxes:
#     print(f"Bounding Box: {box[:4]}, Confidence: {box[4]:.2f}, Class: {box[5]}, Class Score: {box[6]:.2f}")
