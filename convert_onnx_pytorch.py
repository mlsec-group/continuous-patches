import onnx
import torch
from onnx2torch import convert
import onnxruntime as ort
import numpy as np

from torchsummary import summary

# got model from NVIDIA NGC:
# wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/deployable_v1.2/files?redirect=true&path=resnet50_market1501_aicity156.onnx' -O resnet50_market1501_aicity156.onnx
# input shape (1,3,256,128)


# Path to ONNX model
onnx_model_path = "reid/resnet50_market1501_aicity156.onnx"
ort_session = ort.InferenceSession(onnx_model_path)
input_shape = ort_session.get_inputs()[0].shape
print(input_shape) # sanity check: shape is (bs, 3, 256, 128)

# You can pass the path to the onnx model to convert it or...
torch_model = convert(onnx_model_path)

x = torch.ones(4, 3, 256, 128)
print(x.shape)

out_torch = torch_model(x)


out_onnx = np.array(ort_session.run(None, {'input': x.numpy()}))[0]

print(out_onnx.dtype, out_torch.dtype)
print(out_onnx.shape, out_torch.shape)


# Check the Onnx output against PyTorch
print(np.max(np.abs(out_onnx - out_torch.detach().numpy())))
print(np.allclose(out_onnx, out_torch.detach().numpy(), atol=1.0e-7))

# Save the PyTorch model
torch.save(torch_model.state_dict(), "reid/resnet50_market1501_aicity156.pth")