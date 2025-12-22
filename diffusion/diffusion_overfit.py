import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Imports (assuming you saved the previous class definitions) ---
# If you haven't split files, paste the UNet/DiffusionModel classes here.
from diffusion_model import DiffusionModel, construct_T_matrix

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
    
    # 2. Prepare Tensors (Single Batch)
    # Normalize patch to [0, 1] if not already
    patch_tensor = torch.tensor(raw_patch, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    # normalize patch to [-1, 1]
    patch_tensor = patch_tensor * 2 - 1
    
    # Construct Conditioning: [sf, tx, ty, x, y, z, yaw]
    # We explicitly concat them to match the training logic
    cond_vector = np.concatenate([raw_cond, raw_target])
    cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0) # (1, 7)

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
        optimizer.zero_grad()
        
        # A. Sample Random Noise Level (Sigma)
        # We want to train on ALL noise levels, so we sample randomly
        rnd_normal = torch.randn([1, 1, 1, 1], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()
        
        # B. Add Noise
        noise = torch.randn_like(patch_tensor)
        noisy_patch = patch_tensor + noise * sigma
        
        # C. Predict Clean Patch (Denoising)
        # We ask: "Given this noisy version and the condition, what is the original?"
        denoised_guess = model_wrapper.denoised_prediction(noisy_patch, cond_tensor, sigma)
        
        # D. Loss (Direct Reconstruction)
        loss = F.mse_loss(denoised_guess, patch_tensor)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")

    # 5. Verify / Sample
    print("Training done. Sampling...")
    model_wrapper.model.eval()
    
    # Generate 3 samples using the SAME conditioning
    # We repeat the condition vector 3 times
    test_cond = cond_tensor.repeat(3, 1)
    
    samples = model_wrapper.sample(n_samples=3, targets=test_cond, device=device, n_steps=50)
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
    plt.savefig("overfit_result.png")
    plt.close()
    print("If 'Gen' looks like 'Ground Truth', the pipeline works.")

if __name__ == "__main__":
    train_single_overfit()