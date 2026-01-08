import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- Imports ---
from diffusion_model import DiffusionModel


class PatchDataset(Dataset):
    """Custom Dataset for patches and conditioning"""
    def __init__(self, patches_np, targets_np, conds_np):
        self.patches = torch.tensor(patches_np, dtype=torch.float32).unsqueeze(1)  # (B, 1, H, W)
        self.patches = self.patches * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Create Conditioning Vectors: [sf, tx, ty, x, y, z, yaw]
        cond_vector = np.concatenate([conds_np, targets_np], axis=1)  # (B, 7)
        self.conds = torch.tensor(cond_vector, dtype=torch.float32)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.conds[idx]

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
    sample_paths = sorted(glob.glob("frontnet/temp_pid_*/sample_*.npz"))
    if not sample_paths:
        print("Error: no samples found in temp_pid_* folders.")
        return

    max_samples = 1000
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

    # create histogram for scale factors, translation x,y
    sf_tx_ty = conds_np[:, :3]
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.hist(sf_tx_ty[:,0], bins=20)
    plt.title("Scale Factor Histogram")
    plt.subplot(1,3,2)
    plt.hist(sf_tx_ty[:,1], bins=20)
    plt.title("Translation X Histogram")
    plt.subplot(1,3,3)
    plt.hist(sf_tx_ty[:,2], bins=20)
    plt.title("Translation Y Histogram")
    plt.tight_layout()
    plt.savefig("overfit_results/overfit_data_histograms.png")
    plt.close()

    # create 3d scatter plot for target positions
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(targets_np[:,0], targets_np[:,1], targets_np[:,2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Target Position Scatter Plot')
    plt.savefig("overfit_results/overfit_data_scatter.png")
    plt.close()

    # ==========================================================================
    # 2. CREATE DATALOADER
    # ==========================================================================
    dataset = PatchDataset(patches_np, targets_np, conds_np)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")

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
    # 4. TRAINING LOOP WITH DATALOADER
    # ==========================================================================
    print("Starting training with DataLoader...")
    model_wrapper.model.train()
    
    num_epochs = 2000  # Multiple epochs to process 1000 samples
    step_count = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_patches, batch_conds in dataloader:
            # Move batch to device
            batch_patches = batch_patches.to(device)
            batch_conds = batch_conds.to(device)
            batch_size = batch_patches.shape[0]
            
            optimizer.zero_grad()
            
            # A. Sample Random Noise Levels (Batch)
            # Log-normal distribution for sigmas
            rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
            sigmas = (rnd_normal * 1.2 - 1.2).exp()
            
            # B. Add Noise (Batch)
            noise = torch.randn_like(batch_patches)
            noisy_patches = batch_patches + noise * sigmas
            
            # C. Predict Clean Patch (Batch Denoising)
            denoised_guess = model_wrapper.denoised_prediction(noisy_patches, batch_conds, sigmas)
            
            # D. Loss (MSE over entire batch)
            loss = F.mse_loss(denoised_guess, batch_patches)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # step_count += 1
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        if epoch % 10 == 0:
                tqdm.write(f"Step {epoch}, Avg Loss: {avg_epoch_loss:.6f}")
        
        # print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")
    
    # ==========================================================================
    # 5. EVALUATION
    # ==========================================================================
    print("Training done. Generating samples...")
    model_wrapper.model.eval()
    os.makedirs("overfit_results", exist_ok=True)
    torch.save(model_wrapper.model.state_dict(), "overfit_results/diffusion_model.pth")

    all_psnr = []
    
    # Generate samples for evaluation
    n_samples = 10
    total_samples = len(dataset)
    
    with torch.no_grad():
        for idx in tqdm(range(total_samples), desc="Sampling"):
            # Prepare batch of n_samples identical conditions
            test_cond = dataset.conds[idx].unsqueeze(0).repeat(n_samples, 1)  # (n_samples, 7)
            
            # Sample (n_samples, 1, 45, 80)
            samples = model_wrapper.sample(n_samples=n_samples, targets=test_cond, device=device, n_steps=25)
            samples_np = samples.detach().cpu().numpy() # [0, 1] range
            print(samples_np.shape, samples_np.min(), samples_np.max())

            # Visualize
            fig, ax = plt.subplots(1, n_samples + 1, figsize=(12, 3))
            
            # GT (denormalize from [-1, 1] to [0, 1] for display)
            gt_patch = (dataset.patches[idx, 0].cpu().numpy() + 1.0) / 2.0
            ax[0].imshow(gt_patch, cmap='gray')
            ax[0].set_title("Ground Truth")
            
            mean_psnr = 0.0
            for j in range(n_samples):
                ax[j+1].imshow(samples_np[j, 0], cmap='gray')
                ax[j+1].set_title(f"Gen {j+1}")
                mean_psnr += psnr(samples_np[j, 0], gt_patch)
            
            mean_psnr /= n_samples
            all_psnr.append(mean_psnr)
            
            plt.tight_layout()
            plt.savefig(f"overfit_results/result_{idx}.png")
            plt.close()

    all_psnr = np.array(all_psnr)
    print(f"Overall Mean PSNR: {all_psnr.mean():.2f} dB")
    np.save("overfit_results/psnr_scores.npy", all_psnr)

if __name__ == "__main__":
    train_batch_overfit()