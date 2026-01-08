import torch
import numpy as np
import matplotlib.pyplot as plt
# add subfolder diffusion to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'diffusion'))
from diffusion_model import DiffusionModel, construct_T_matrix
from attack_minimal_single import project_patch
from util import load_model, normalize_yaw_t

def check_generalization():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = DiffusionModel(device=device, patch_size=(45, 80), prediction_model_name='frontnet')
    model.load("overfit_results/diffusion_model.pth")
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
    frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
    frontnet.eval()
    
    # Create fake background (grey)
    bg = torch.ones((1, 1, 96, 160), device=device) * 0.5
    
    # Project
    T = construct_T_matrix(new_cond[:,0], new_cond[:,1], new_cond[:,2])

    # print(bg.requires_grad)
    # print(patch.requires_grad)
    # print(T.requires_grad)
    manipulated = project_patch(patch, T, bg).clamp(0, 1)
    # print(manipulated.requires_grad)
    
    # Predict
    x, y, z, yaw = frontnet(manipulated * 255.)
    pred = torch.stack([x, y, z, yaw], dim=1).detach().cpu().numpy()[0].mT
    
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
    check_generalization()