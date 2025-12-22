import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Imports (assuming you saved the previous class definitions) ---
# If you haven't split files, paste the UNet/DiffusionModel classes here.
from diffusion_model import DiffusionModel, construct_T_matrix, project_patch


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
    raw_patch_t = torch.tensor(raw_patch, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
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

    bg_img = model_wrapper.train_set.dataset.data[0][0] / 255. # (H, W)
    bg_img_t = bg_img.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    # print(f"Background image shape: {bg_img.shape}")
    # bg_img_t = torch.tensor(bg_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
     # gt sanity check
    for i, (patch, condition) in enumerate(zip(patch_tensor, cond_tensor)):
        patch = patch.unsqueeze(0)  # (1, 1, H, W)
        patch = (patch + 1) / 2  # convert back to [0, 1] for projection
        condition = condition.unsqueeze(0)  # (1, 7)
        T = construct_T_matrix(*condition[0, :3])  # (1, 3, 3)
        manipulated_image = project_patch(patches=patch, T_matrices=T, images=bg_img_t)
        x, y, z, yaw = model_wrapper.prediction_model(manipulated_image*255.)
        pred = torch.stack([x, y, z, yaw])
        pred = pred.squeeze(2).mT
        print(f"Prediction from frontnet (sanity check): {pred.detach().cpu().numpy()}")
        gt_pred = condition[0, 3:].detach().cpu().numpy()
        print(f"Ground Truth target: {gt_pred}")
        dist = np.linalg.norm(pred.detach().cpu().numpy() - gt_pred)
        print(f"Prediction error (L2 norm): {dist:.4f}")
   

        cond_tensor[i, 3:] = pred[0]  # use the prediction as target to see if it can overfit better


    print("Updated Conditioning Vector with Frontnet Predictions:")
    print(cond_tensor.detach().cpu().numpy())


    optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=1e-3)

    # 4. Overfit Loop
    print("Starting overfit training...")
    model_wrapper.model.train()
    
    for i in range(2000): # 1000 steps should be plenty for 1 sample
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
    samples2 = model_wrapper.sample(n_samples=3, targets=test_cond2, device=device, n_steps=25)
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
        score = psnr(samples[j, 0], raw_patch)
        print(f"Sample {j+1} PSNR: {score:.2f} dB")
        score2 = psnr(samples2[j, 0], raw_patch2)
        print(f"Sample2 {j+1} PSNR: {score2:.2f} dB")

    # prediction frontnet model test
   
    samples_t = torch.tensor(samples[0], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 1, H, W)
    print(test_cond)
    T = construct_T_matrix(*test_cond[0, :3])  # (1, 3, 3)
    
    print(f"T matrix shape: {T.shape}, {T}")
    manipulated_image = project_patch(patches=samples_t, T_matrices=T, images=bg_img_t)
    manipulated_image_gt = project_patch(patches=raw_patch_t, T_matrices=T, images=bg_img_t)
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(bg_img, cmap='gray')
    ax[0].set_title("Background Image")
    ax[1].imshow(manipulated_image[0, 0].detach().cpu().numpy(), cmap='gray')
    ax[1].set_title("Manipulated Image with Generated Patch")
    ax[2].imshow(manipulated_image_gt[0, 0].detach().cpu().numpy(), cmap='gray')
    ax[2].set_title("Manipulated Image with GT Patch")
    plt.tight_layout()
    plt.savefig("manipulated_image.png")
    plt.close()
    x, y, z, yaw = model_wrapper.prediction_model(manipulated_image*255.)
    pred = torch.stack([x, y, z, yaw])
    pred = pred.squeeze(2).mT
    print(f"Prediction from frontnet: {pred.detach().cpu().numpy()}")
    gt_pred = test_cond[0, 3:].detach().cpu().numpy()
    print(f"Ground Truth target: {gt_pred}")
    dist = np.linalg.norm(pred.detach().cpu().numpy() - gt_pred)
    print(f"Prediction error (L2 norm): {dist:.4f}")

   

if __name__ == "__main__":
    train_single_overfit()