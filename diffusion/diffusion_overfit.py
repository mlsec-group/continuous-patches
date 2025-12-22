import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Imports (assuming you saved the previous class definitions) ---
# If you haven't split files, paste the UNet/DiffusionModel classes here.
from diffusion_model import DiffusionModel, construct_T_matrix


# calc similarity score (psnr)
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0  # since we normalized to [-1, 1]
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def train_single_overfit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. Load Data
    path = "temp_pid_364256/sample_4.npz"
    if not os.path.exists(path):
        print(f"Error: {path} not found. Please check path.")
        return

    data = np.load(path, allow_pickle=True)
    
    # Extract raw data
    raw_patch = data['patch']      # (45, 80)
    raw_target = data['target']    # (4,) -> x, y, z, yaw
    raw_cond = data['condition']   # (3,) -> sf, tx, ty

    # 2nd patch:
    path2 = 'temp_pid_364256/sample_1.npz'
    data2 = np.load(path2, allow_pickle=True)
    raw_patch2 = data2['patch']
    raw_target2 = data2['target']
    raw_cond2 = data2['condition']

    patches = [raw_patch, raw_patch2]
    targets = [raw_target, raw_target2]
    conds = [raw_cond, raw_cond2]
    
    # 2. Prepare Tensors (Single Batch)
    # Normalize patch to [0, 1] if not already
    patch_tensor = torch.tensor(patches, dtype=torch.float32, device=device).unsqueeze(1) # (B, 1, H, W)
    # normalize patch to [-1, 1]
    patch_tensor = patch_tensor * 2 - 1
    
    # Construct Conditioning: [sf, tx, ty, x, y, z, yaw]
    # We explicitly concat them to match the training logic
    cond_vector = np.concatenate([conds, targets], axis=1) # (2, 7)
    cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device) # (2, 7)

    print(f"Target Patch Range: [{patch_tensor.min():.2f}, {patch_tensor.max():.2f}]")
    print(f"Conditioning Vector: {cond_vector}")

    # 3. Initialize Model
    # We use a smaller model configuration for this quick test
    model_wrapper = DiffusionModel(
        device=device,
        patch_size=(45, 80),
        prediction_model_name='frontnet' # Not used here, but required by init
    )
    
    optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=1e-3)

    # 4. Overfit Loop
    print("Starting overfit training...")
    model_wrapper.model.train()
    
    for i in range(1000): # 1000 steps should be plenty for 1 sample
        for patch, condition in zip(patch_tensor, cond_tensor):
            patch = patch.unsqueeze(0)  # (1, 1, H, W)
            condition = condition.unsqueeze(0)  # (1, 7)
            
            # Zero Grad
            optimizer.zero_grad()
            
            # A. Sample Random Noise Level (Sigma)
            rnd_normal = torch.randn([1, 1, 1, 1], device=device)
            sigma = (rnd_normal * 1.2 - 1.2).exp()
            
            # B. Add Noise
            noise = torch.randn_like(patch)
            noisy_patch = patch + noise * sigma
            
            # C. Predict Clean Patch (Denoising)
            denoised_guess = model_wrapper.denoised_prediction(noisy_patch, condition, sigma)
            
            # D. Loss (Direct Reconstruction)
            loss = F.mse_loss(denoised_guess, patch)
            
            loss.backward()
            optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")

    # 5. Verify / Sample
    print("Training done. Sampling...")
    model_wrapper.model.eval()
    
    # Generate 3 samples using the SAME conditioning
    # We repeat the condition vector 3 times
    test_cond = cond_tensor[0].repeat(3, 1)
    
    samples = model_wrapper.sample(n_samples=3, targets=test_cond, device=device, n_steps=10)
    samples = samples.detach().cpu().numpy()

    # 6. Visualize
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    
    # Ground Truth
    ax[0].imshow(raw_patch, cmap='gray')
    ax[0].set_title("Ground Truth")
    
    # Generated
    for j in range(3):
        ax[j+1].imshow(samples[j, 0], cmap='gray')
        ax[j+1].set_title(f"Gen {j+1}")
        
    plt.tight_layout()
    # plt.show()
    plt.savefig("overfit_result0.png")
    plt.close()

    test_cond2 = cond_tensor[1].repeat(3, 1)
    samples2 = model_wrapper.sample(n_samples=3, targets=test_cond2, device=device, n_steps=10)
    samples2 = samples2.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(raw_patch2, cmap='gray')
    ax[0].set_title("Ground Truth 2")
    for j in range(3):
        ax[j+1].imshow(samples2[j, 0], cmap='gray')
        ax[j+1].set_title(f"Gen2 {j+1}")
    plt.tight_layout()
    plt.savefig("overfit_result1.png")
    plt.close()
    # print("If 'Gen' looks like 'Ground Truth', the pipeline works.")

    for j in range(3):
        score = psnr(samples[j, 0], raw_patch * 2 - 1)
        print(f"Sample {j+1} PSNR: {score:.2f} dB")
        score2 = psnr(samples2[j, 0], raw_patch2 * 2 - 1)
        print(f"Sample2 {j+1} PSNR: {score2:.2f} dB")

if __name__ == "__main__":
    train_single_overfit()