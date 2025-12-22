import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# --- Imports ---
from diffusion_model import DiffusionModel

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    PIXEL_MAX = 2.0  # normalized [-1, 1] range = 2.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def train_batch_overfit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # ==========================================================================
    # 1. LOAD DATA
    # ==========================================================================
    sample_paths = sorted(glob.glob("temp_pid_*/sample_*.npz"))
    if not sample_paths:
        print("Error: no samples found in temp_pid_* folders.")
        return

    max_samples = 100
    patches_list, targets_list, conds_list = [], [], []

    print(f"Loading {min(len(sample_paths), max_samples)} samples...")
    for p in sample_paths[:max_samples]:
        try:
            data = np.load(p, allow_pickle=True)
            patches_list.append(data['patch'].astype(np.float32))
            targets_list.append(data['target'].astype(np.float32))
            conds_list.append(data['condition'].astype(np.float32))
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")

    if not patches_list: return

    # Stack to Numpy Arrays
    patches_np = np.stack(patches_list, axis=0)  # (B, H, W)
    targets_np = np.stack(targets_list, axis=0)  # (B, 4)
    conds_np = np.stack(conds_list, axis=0)      # (B, 3)

    # Save corpus for later reuse
    np.savez("overfit_data.npz", patches=patches_np, targets=targets_np, conds=conds_np)

    # ==========================================================================
    # 2. PREPARE TENSORS (BATCH)
    # ==========================================================================
    # Normalize patches to [-1, 1] for diffusion
    patches_t = torch.tensor(patches_np, dtype=torch.float32, device=device).unsqueeze(1) # (B, 1, H, W)
    patches_t = patches_t * 2.0 - 1.0

    # Create Conditioning Vectors: [sf, tx, ty, x, y, z, yaw]
    cond_vector = np.concatenate([conds_np, targets_np], axis=1) # (B, 7)
    cond_t = torch.tensor(cond_vector, dtype=torch.float32, device=device) # (B, 7)

    batch_size = patches_t.shape[0]
    print(f"Batch Size: {batch_size}")
    print(f"Conditioning Vector Shape: {cond_t.shape}")

    # ==========================================================================
    # 3. INITIALIZE MODEL
    # ==========================================================================
    model_wrapper = DiffusionModel(
        device=device,
        patch_size=(45, 80),
        prediction_model_name='frontnet'
    )
    optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=1e-3)

    # ==========================================================================
    # 4. VECTORIZED TRAINING LOOP
    # ==========================================================================
    print("Starting vectorized training...")
    model_wrapper.model.train()
    
    # 2000 Steps
    for i in tqdm(range(2000), desc="Training"):
        optimizer.zero_grad()
        
        # A. Sample Random Noise Levels (Batch)
        # Log-normal distribution for sigmas
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
        sigmas = (rnd_normal * 1.2 - 1.2).exp()
        
        # B. Add Noise (Batch)
        noise = torch.randn_like(patches_t)
        noisy_patches = patches_t + noise * sigmas
        
        # C. Predict Clean Patch (Batch Denoising)
        # The UNet processes the entire batch (B, 1, 45, 80) in parallel
        denoised_guess = model_wrapper.denoised_prediction(noisy_patches, cond_t, sigmas)
        
        # D. Loss (MSE over entire batch)
        # Weighting by sigma is standard in EDM but simple MSE works for overfitting
        loss = F.mse_loss(denoised_guess, patches_t)
        
        loss.backward()
        optimizer.step()
        
        if i % 200 == 0:
            tqdm.write(f"Step {i}: Loss {loss.item():.6f}")

    # ==========================================================================
    # 5. EVALUATION
    # ==========================================================================
    print("Training done. Generating samples...")
    model_wrapper.model.eval()
    os.makedirs("overfit_results", exist_ok=True)
    torch.save(model_wrapper.model.state_dict(), "overfit_results/diffusion_model.pth")

    all_psnr = []
    
    # Generate 3 samples per condition
    # We can do this in batches too, but for plotting/saving logic, iterating is fine.
    # To speed up, we can batch the 3 samples per condition.
    
    with torch.no_grad():
        for idx in tqdm(range(batch_size), desc="Sampling"):
            # Prepare batch of 3 identical conditions
            test_cond = cond_t[idx].unsqueeze(0).repeat(3, 1) # (3, 7)
            
            # Sample (3, 1, 45, 80)
            samples = model_wrapper.sample(n_samples=3, targets=test_cond, device=device, n_steps=25)
            samples_np = samples.detach().cpu().numpy() # [0, 1] range

            # Visualize
            fig, ax = plt.subplots(1, 4, figsize=(12, 3))
            
            # GT (Un-normalize from [-1, 1] to [0, 1] for display if needed, 
            # but patches_np is already [0, 1] if loaded directly)
            ax[0].imshow(patches_np[idx], cmap='gray')
            ax[0].set_title("Ground Truth")
            
            mean_psnr = 0.0
            for j in range(3):
                ax[j+1].imshow(samples_np[j, 0], cmap='gray')
                ax[j+1].set_title(f"Gen {j+1}")
                mean_psnr += psnr(samples_np[j, 0], patches_np[idx])
            
            mean_psnr /= 3.0
            all_psnr.append(mean_psnr)
            
            plt.tight_layout()
            plt.savefig(f"overfit_results/result_{idx}.png")
            plt.close()

    all_psnr = np.array(all_psnr)
    print(f"Overall Mean PSNR: {all_psnr.mean():.2f} dB")
    np.save("overfit_results/psnr_scores.npy", all_psnr)

if __name__ == "__main__":
    train_batch_overfit()