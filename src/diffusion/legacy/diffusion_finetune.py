"""DEPRECATED: Legacy experiment script from earlier development phases.

This script is not part of the released AISec'26 artifact and is not
maintained: it references modules or workflows that are no longer present
in this repository. It is kept for reference only.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm

# --- Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffusion_model import DiffusionModel, construct_T_matrix
from util import load_model, load_dataset
from attack_minimal_single import project_patch, normalize_yaw_t
from diffusion_overfit import PatchDataset
from torch.utils.data import DataLoader

# Ensure parent folder is on sys.path so we can import util from there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project root: ", project_root)
sys.path.insert(0, project_root)


def train_adversarial_finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # ==========================================================================
    # 1. LOAD OVERFIT DATA (The Anchors)
    # ==========================================================================
    if not os.path.exists("overfit_data.npz"):
        print("Error: overfit_data.npz not found. Run train_overfit.py first.")
        return

    data = np.load("overfit_data.npz")
    patches_np = data['patches']  # (B, H, W)
    targets_np = data['targets']  # (B, 4)
    conds_np = data['conds']      # (B, 3)

    # Prepare Tensors
    # Anchors (Gold Patches) for stability
    anchor_patches = torch.tensor(patches_np, dtype=torch.float32, device=device).unsqueeze(1)
    anchor_patches = anchor_patches * 2.0 - 1.0 # Normalize to [-1, 1]

    # Conditioning
    cond_vector = np.concatenate([conds_np, targets_np], axis=1)
    cond_t = torch.tensor(cond_vector, dtype=torch.float32, device=device) # (B, 7)
    
    # Explicit Targets for Loss Calculation
    target_pose_t = torch.tensor(targets_np, dtype=torch.float32, device=device) # (B, 4)

    batch_size = 128
    print(f"Loaded {len(patches_np)} samples for fine-tuning.")

    # ==========================================================================
    # 2. SETUP MODELS
    # ==========================================================================
    
    # A. Diffusion Model (The Student)
    model_wrapper = DiffusionModel(
        device=device,
        patch_size=(45, 80),
        prediction_model_name='frontnet'
    )
    # Load the overfit weights
    if os.path.exists("overfit_results/diffusion_model.pth"):
        model_wrapper.model.load_state_dict(torch.load("overfit_results/diffusion_model.pth"))
        print("Loaded overfit weights.")
    else:
        print("Warning: No pre-trained weights found. Starting from scratch (hard).")

    # B. Frontnet (The Critic/Victim)
    frontnet_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Models/Frontnet160x32.pt")
    frontnet = load_model(frontnet_path, device, config="160x32")
    frontnet.eval()
    # Freeze Frontnet completely (we only want gradients to flow through it to the patch)
    for p in frontnet.parameters():
        p.requires_grad = False

    # C. Background Images (For Projection)
    # data_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle")
    bg_dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)

    patch_dataset = PatchDataset(patches_np*2 - 1.0, targets_np, conds_np)
    patch_dataloader = DataLoader(patch_dataset, batch_size=128, shuffle=True, drop_last=True)
    

    # Optimizer (Low LR for fine-tuning)
    optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=1e-4)

    # ==========================================================================
    # 3. ADVERSARIAL FINE-TUNING LOOP
    # ==========================================================================
    print("Starting adversarial fine-tuning...")
    model_wrapper.model.train()

    # Get T-Matrices for projection (Static for this batch)
    # cond_t has [sf, tx, ty, ...]
    # T_matrices = construct_T_matrix(cond_t[:, 0], cond_t[:, 1], cond_t[:, 2]) # (B, 3, 3)
    # if T_matrices.dim() == 2: T_matrices = T_matrices.unsqueeze(0)

    
    for i in tqdm(range(500), desc="Fine-tuning"):
        epoch_loss = 0.0
        for bg_imgs, _ in bg_dataset:
            # print("BG batch shape: ", bg_imgs.shape)
            a_patch_batch, a_cond_batch = next(iter(patch_dataloader))
            # Move batch to device
            anchor_patches = a_patch_batch.to(device)
            cond_t = a_cond_batch.to(device)
            optimizer.zero_grad()

            # --- A. Generate Patches (One-Step Dreaming) ---
            # Sample random noise levels
            rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
            sigmas = (rnd_normal * 1.2 - 1.2).exp()
            
            # Add noise to anchors (we start exploration *near* the known good patches)
            noise = torch.randn_like(anchor_patches)
            noisy_input = anchor_patches + noise * sigmas
            
            # Predict the clean patch
            # This gradient path goes: Loss -> Frontnet -> Projection -> Denoised_Patch -> UNet
            denoised_patch = model_wrapper.denoised_prediction(noisy_input, cond_t, sigmas)
            
        # --- B. Project onto Scene ---
        # Get background batch
        # try:
        #     bg_imgs, _ = next(bg_iter)
        # except StopIteration:
        #     bg_iter = iter(bg_dataset)
        #     bg_imgs, _ = next(bg_iter)
        
        # Handle batch size mismatch if dataset end reached
        # if bg_imgs.shape[0] != batch_size:
        #     bg_imgs = bg_imgs[:batch_size]
        #     if bg_imgs.shape[0] < batch_size:
        #         # pad or skip (skipping for simplicity in this minimal script)
        #         continue 
                
            bg_imgs = bg_imgs.to(device) / 255.0

            jitter_sf = cond_t[0, 0] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-0.3, 0.3)
            jitter_tx = cond_t[0, 1] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-10.0, 10.0)
            jitter_ty = cond_t[0, 2] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-10.0, 10.0)
            jitter_x = cond_t[0, 3] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-0.2, 0.2)
            jitter_y = cond_t[0, 4] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-0.2, 0.2)
            jitter_z = cond_t[0, 5] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-0.2, 0.2)
            jitter_yaw = cond_t[0, 6] + torch.empty(1, device=device, dtype=torch.float32).uniform_(-0.01, 0.01)

            T_matrices = construct_T_matrix(jitter_sf, jitter_tx, jitter_ty) # (1, 3, 3)
            # print("T_matrix: ", T_matrices)
            T_matrices = T_matrices.repeat(batch_size, 1, 1)
            # print("T_matrices shape: ", T_matrices.shape)

            
            condition_single = torch.stack([jitter_sf, jitter_tx, jitter_ty, jitter_x, jitter_y, jitter_z, jitter_yaw], dim=1)
            # print("condition_single: ", condition_single)
            # print("condition_single shape: ", condition_single.shape)
            sampled_patches = model_wrapper.sample(n_samples=1, targets=condition_single, device=device, patch_size=(45, 80), n_steps=2)
            sampled_patches = sampled_patches.repeat(batch_size, 1, 1, 1)
            # print("Sampled patches shape: ", sampled_patches.shape, sampled_patches.dtype)
            # print("Sampled patches min/max: ", sampled_patches.min().item(), sampled_patches.max().item())

            manimpulated_images = project_patch(sampled_patches, T_matrices, bg_imgs)
            manipulated_images = manimpulated_images.clamp(0., 1.)
            # print("Manipulated images shape: ", manipulated_images.shape, manipulated_images.dtype)
            # print("Manipulated images min/max: ", manipulated_images.min().item(), manipulated_images.max().item())
            x, y, z, yaw = frontnet(manipulated_images * 255.)
            pred_pose = torch.stack([x, y, z, yaw], dim=1).squeeze(2) # (B, 4)
            # print("Examples of predicted poses: ", pred_pose[:5, :])
            # print("Predicted pose shape: ", pred_pose.shape)

            target_pose = condition_single[0, 3:]  # (1, 4)
            # print("Target pose: ", target_pose)
            target_yaw = target_pose[3]
            target_pose = target_pose.repeat(batch_size, 1)
            target_yaw = target_yaw.repeat(batch_size)

            # cond_t = torch.stack([jitter_sf, jitter_tx, jitter_ty, jitter_x, jitter_y, jitter_z, jitter_yaw], dim=1)
            # print("cond_t shape: ", cond_t.shape)
            # sampled_patches = model_wrapper.sample(n_samples=batch_size, targets=cond_t, device=device, patch_size=(45, 80), n_steps=1)
            # print("Sampled patches shape: ", sampled_patches.shape, sampled_patches.dtype)
            # # Project the DENOISED guess
            # manipulated_images = project_patch(sampled_patches, T_matrices, bg_imgs)
            # manipulated_images = manipulated_images.clamp(0., 1.)
            # print("Manipulated images shape: ", manipulated_images.shape)
            # # --- C. Victim Forward Pass ---
            # # Frontnet expects [0, 255]
            # x, y, z, yaw = frontnet(manipulated_images * 255.)
            # pred_pose = torch.stack([x, y, z, yaw], dim=1).squeeze(2) # (B, 4)
            # print("Predicted pose shape: ", pred_pose.shape)
            # # --- D. Calculate Losses ---
            
            # 1. Adversarial Loss (The Goal)
            # Minimize distance between Frontnet Prediction and Target Condition
            # Target is cond_t[:, 3:] -> (x, y, z, yaw)
            # target_pose = cond_t[:, 3:]  # (B, 4)
            # target_yaw = target_pose[:, 3]

            # print("Target pose shape: ", target_pose.shape)
            # print("Target yaw shape: ", target_yaw.shape)
            dist_loss = F.mse_loss(pred_pose[:, :3], target_pose[:, :3])
            # print("dist_loss: ", dist_loss.item())
            ang_loss = (1 - torch.cos(normalize_yaw_t(pred_pose[:, 3]) - normalize_yaw_t(target_yaw))).mean()
            # print("ang_loss: ", ang_loss.item())
            adv_loss = dist_loss + ang_loss
            # print("adv_loss: ", adv_loss.item())

            # 2. Anchor Loss (The Stabilizer)
            # Keep the patch close to the "Gold Standard" we mined earlier.
            # This prevents mode collapse or generating pure noise.
            # We assume the "Gold" patches were good, we just want to polish them.
            rec_loss = F.mse_loss(denoised_patch, anchor_patches)
            # print("rec_loss: ", rec_loss.item())
            # Total Loss
            # We weight adversarial loss higher now since we want to improve performance
            loss = adv_loss + (0.2 * rec_loss)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(bg_dataset)

        if i % 10 == 0:
            tqdm.write(f"Step {i} | Adv Loss: {adv_loss.item():.4f} | Rec Loss: {rec_loss.item():.4f} | Avg Loss: {avg_epoch_loss:.4f}")

    # ==========================================================================
    # 4. SAVE & EXIT
    # ==========================================================================
    print("Fine-tuning complete.")
    torch.save(model_wrapper.model.state_dict(), "overfit_results/diffusion_finetuned.pth")

if __name__ == "__main__":
    train_adversarial_finetune()