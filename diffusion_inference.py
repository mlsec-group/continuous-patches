import torch
import numpy as np
import matplotlib.pyplot as plt
# add subfolder diffusion to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'diffusion'))
from diffusion_model import DiffusionModel, construct_T_matrix
from attack_minimal_single import project_patch
from util import load_model, normalize_yaw_t, load_dataset

import argparse

def check_generalization(prediction_model_name='frontnet'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

    patch_size = (45, 80) if prediction_model_name == 'frontnet' else (150, 320)
    
    # 1. Load Model
    model = DiffusionModel(device=device, patch_size=patch_size, prediction_model_name=prediction_model_name)
    model.load(f"overfit_results/diffusion_model_{prediction_model_name}_1000.pth")
    model.model.eval()
    
    # 2. Define a "New" Condition (Arbitrary values inside valid range)
    # sf=0.8, tx=50, ty=30 (Patch Position)
    # x=1.0, y=0.0, z=0.0, yaw=0.0 (Drone Target)
    new_cond = torch.tensor([[0.8, 50.0, 30.0, 1.0, 0.0, 0.0, 0.0]], device=device)
    
    print(f"Testing Unseen Condition: {new_cond.cpu().numpy()}")

    # 3. Generate Patch
    patch = model.sample(n_samples=1, targets=new_cond, device=device, n_steps=50)
    
    # 4. Validate with Frontnet
    # Load Victim
    if prediction_model_name == 'frontnet':
        frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
        frontnet.eval()
    elif prediction_model_name == 'yolov5':
        from yolo_bounding import YOLOBox
        yolo = YOLOBox()
        yolo.model.eval()
        
    # Create fake background (grey)
    # bg = torch.ones((1, 1, 96, 160), device=device) * 0.5
    bg_dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = 1, shuffle = True, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)
    bg_img, _ = next(iter(bg_dataset))
    bg = bg_img.to(device) / 255.0  # Normalize to [0, 1]
    
    if prediction_model_name == 'yolov5':
        bg = torch.nn.functional.interpolate(bg, size=(320, 640), mode='bilinear', align_corners=False)
    # Project
    T = construct_T_matrix(new_cond[:,0], new_cond[:,1], new_cond[:,2])

    # print(bg.requires_grad)
    # print(patch.requires_grad)
    # print(T.requires_grad)
    manipulated = project_patch(patch, T, bg).clamp(0, 1)
    # print(manipulated.requires_grad)
    
    # Predict
    if prediction_model_name == 'frontnet':
        x, y, z, yaw = frontnet(manipulated * 255.)
        pred = torch.stack([x, y, z, yaw]).squeeze(2).mT.detach().cpu().numpy()
    elif prediction_model_name == 'yolov5':
        manipulated = manipulated.repeat_interleave(3, dim=1)
        predicted_boxes, _ = yolo(manipulated, target_anchor=None)
        scaled_box = predicted_boxes.clone()
        # scale back to 160x96
        scaled_box[:, [0, 2]] *= (160.0 / 640.)  # x coords
        scaled_box[:, [1, 3]] *= (96.0 / 320.)   # y coords
        pred = yolo.cam.batch_xyz_from_boxes(scaled_box).detach().cpu().numpy()
        print(pred.shape, pred)


    
    print(f"Goal: [1.00, 0.00, 0.00, 0.00]")
    print(f"Pred: {pred}")

    error = np.abs(pred - np.array([1.0, 0.0, 0.0, 0.0]))
    print(f"Error: {error}")
    
    # Plot
    plt.imshow(manipulated[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title(f"Generalization Test\nErr : {error}")
    # plt.show()
    plt.savefig("generalization_test.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Inference for Drone Patch Generation')
    parser.add_argument('-m', '--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet',)
    
    args = parser.parse_args()
    check_generalization(args.model)