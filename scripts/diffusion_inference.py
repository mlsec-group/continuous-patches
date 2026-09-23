import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.diffusion.diffusion_model import DiffusionModel, construct_T_matrix
from src.diffusion.diffusion_overfit import normalize_condition
from src.attacks import project_patch
from src.util import load_model, normalize_yaw_t, load_dataset
from src.simulation import T_matrix, calc_heading_vec
import cv2
import time
import glob

import argparse

def _resolve_checkpoint(project_root, prediction_model_name, checkpoint_path=None):
    if checkpoint_path is not None:
        return checkpoint_path

    ckpt_dir = os.path.join(project_root, "flipped_diffusion", prediction_model_name)
    preferred = os.path.join(ckpt_dir, f"diffusion_model_{prediction_model_name}_1000.pth")
    if os.path.exists(preferred):
        return preferred

    candidates = sorted(glob.glob(os.path.join(ckpt_dir, f"diffusion_model_{prediction_model_name}_*.pth")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return candidates[-1]


def _frontnet_abs_error(pred_t, target_t):
    return torch.mean(torch.abs(pred_t - target_t), dim=1)


def _yolo_abs_error(pred_t, target_t):
    return torch.mean(torch.abs(pred_t - target_t), dim=1)


def check_generalization(
    prediction_model_name='frontnet',
    n_steps=25,
    n_candidates=1,
    checkpoint_path=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    patch_size = (45, 80) if prediction_model_name == 'frontnet' else (150, 320)
    
    # 1. Load Model
    model = DiffusionModel(device=device, patch_size=patch_size, prediction_model_name=prediction_model_name)
    model_path = _resolve_checkpoint(project_root, prediction_model_name, checkpoint_path)
    print(f"Loading checkpoint: {model_path}")
    model.load(model_path)
    model.model.eval()

    if os.path.exists(f"{project_root}/{prediction_model_name}/corpus_{prediction_model_name}.npz"):
        # ==========================================================================
        # 1. LOAD PRE-SAVED CORPUS
        # ==========================================================================
        print(f"Loading pre-saved corpus for {prediction_model_name}...")
        data = np.load(f"{project_root}/{prediction_model_name}/corpus_{prediction_model_name}.npz")
        patches_np = data['patches'].astype(np.float32)[:]
        targets_np = data['targets'].astype(np.float32)[:]
        conds_np = data['conds'].astype(np.float32)[:]
        patch_h, patch_w = patches_np.shape[1], patches_np.shape[2]
        print(f"Loaded dataset: {patches_np.shape[0]} samples of size ({patch_h}, {patch_w})")


    cond_vector = np.concatenate([conds_np, targets_np], axis=1)  # (B, 7)


    original_patches = patches_np[:5]
    for i, patch in enumerate(original_patches):
        cv2.imwrite(f"example_corpus_{prediction_model_name}_{i}.png", (patch * 255).astype(np.uint8))

    example_conditions = cond_vector[:5]

    
    # 4. Validate with Frontnet
        # Load Victim
    if prediction_model_name == 'frontnet':
        frontnet = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
        frontnet.eval()
    elif prediction_model_name == 'yolov5':
        from src.yolo_bounding import YOLOBox
        yolo = YOLOBox()
        yolo.model.eval()
        
    # Create fake background (grey)
    # bg = torch.ones((1, 1, 96, 160), device=device) * 0.5
    bg_dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = 1, shuffle = True, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)
    bg_img, _ = next(iter(bg_dataset))
    bg = bg_img.to(device) / 255.0  # Normalize to [0, 1]
    
    if prediction_model_name == 'yolov5':
        bg = torch.nn.functional.interpolate(bg, size=(320, 640), mode='bilinear', align_corners=False)
    # 2. Define a "New" Condition (Arbitrary values inside valid range)
    # sf=0.8, tx=50, ty=30 (Patch Position)
    # x=1.0, y=0.0, z=0.0, yaw=0.0 (Drone Target)
    # new_cond = torch.tensor([[0.8, 50.0, 30.0, 1.0, 0.0, 0.0, 0.0]], device=device)
    # condition from experiment:
    # base_cond = torch.tensor([[ 0.76900554, 45.857277, 47.99321, 0.95535207, -0.09457839, -0.0215925, -0.29849893]], device=device)
    # print("Original Condition: ", base_cond.cpu().numpy())

    for i, cond in enumerate(example_conditions):
        
        base_cond = torch.tensor(cond, device=device).unsqueeze(0)
        new_cond = normalize_condition(base_cond, model=prediction_model_name)


        print(f"Testing Unseen Condition: {new_cond.cpu().numpy()}")

        time_start = time.time()
        # 3. Generate Patch
        with torch.no_grad():
            patch_candidates = model.sample(
                n_samples=n_candidates,
                targets=new_cond,
                device=device,
                n_steps=n_steps,
            )

        # Project all candidates with the same transform/background.
        T = construct_T_matrix(base_cond[:,0], base_cond[:,1], base_cond[:,2])
        T_candidates = T.repeat(n_candidates, 1, 1)
        bg_candidates = bg.repeat(n_candidates, 1, 1, 1)
        manipulated_candidates = project_patch(patch_candidates, T_candidates, bg_candidates).clamp(0, 1)

        target_t = base_cond[:, 3:].repeat(n_candidates, 1)

        if prediction_model_name == 'frontnet':
            x, y, z, yaw = frontnet(manipulated_candidates * 255.)
            pred_t = torch.stack([x, y, z, yaw]).squeeze(2).mT
            candidate_errors = _frontnet_abs_error(pred_t, target_t)
        elif prediction_model_name == 'yolov5':
            yolo_inputs = manipulated_candidates.repeat_interleave(3, dim=1)
            predicted_boxes, _ = yolo(yolo_inputs, target_anchor=None)
            scaled_box = predicted_boxes.clone()
            scaled_box[:, [0, 2]] *= (160.0 / 640.)
            scaled_box[:, [1, 3]] *= (96.0 / 320.)
            pred_t = yolo.cam.batch_xyz_from_boxes(scaled_box)
            candidate_errors = _yolo_abs_error(pred_t, target_t)

        best_idx = int(torch.argmin(candidate_errors).item())
        patch = patch_candidates[best_idx:best_idx+1]
        manipulated = manipulated_candidates[best_idx:best_idx+1]
        pred = pred_t[best_idx:best_idx+1].detach().cpu().numpy()
        
        time_end = time.time()
        print(f"Diffusion Sampling Time: {time_end - time_start:.2f} seconds")
        print(f"Best candidate index: {best_idx}, mean absolute error: {candidate_errors[best_idx].item():.6f}")

        print(patch.shape)
        print("Generated patch stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(patch.min().item(), patch.max().item(), patch.mean().item()))

        print("T matrix:\n", T.cpu().numpy())


        
        print(f"Goal: {base_cond[0,3:].cpu().numpy()}")
        print(f"Pred: {pred}")

        error = np.abs(pred - base_cond[0,3:].cpu().numpy())
        print(f"Error: {error}")
        
        # Plot
        plt.imshow(manipulated[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Generalization Test\nErr : {error}")
        # plt.show()
        plt.savefig(f"generalization_test_{i}.png")
        plt.close()

        cv2.imwrite(f"example_diffusion_{prediction_model_name}_{i}.png", (patch.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Inference for Drone Patch Generation')
    parser.add_argument('-m', '--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet',)
    parser.add_argument('--steps', type=int, default=25, help='Number of diffusion denoising steps at sampling time')
    parser.add_argument('--candidates', type=int, default=1, help='Sample K candidates and keep the best by victim-model error')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint path override')
    
    args = parser.parse_args()
    check_generalization(
        args.model,
        n_steps=args.steps,
        n_candidates=max(1, args.candidates),
        checkpoint_path=args.checkpoint,
    )