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

import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from util import load_model, load_dataset, normalize_yaw_t
from simulation_refactor import T_matrix

def _perspective_grid(
coeffs: list[float], 
w: int, h: int, 
ow: int, oh: int, 
dtype: torch.dtype, 
device: torch.device,
center = None,
) -> torch.Tensor:
    # source: https://github.com/pytorch/pytorch/issues/100526#issuecomment-1610226058
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    batch_size = coeffs.shape[0]
    theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)

    theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0) # theta2 is a matrix of shape [2, 3], it is the last row of the original transformation matrix repeated 2x
    theta2 = theta2.reshape(batch_size, 2, 3) # reshape from [batch_size*2, 3] to [batch_size, 2, 3] 

    d = 0.5
    base_grid = torch.empty(batch_size, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    shape = (batch_size, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    if center is not None:
        center = torch.tensor(center, dtype=dtype, device=device)
    else:
        center = 1.0

    output_grid = output_grid1.div_(output_grid2).sub_(center)
    return output_grid.view(batch_size, oh, ow, 2)
    
def project_patch(patches, T_matrices, images):
    """ Project the patches on the batch of camera images.
    Args:
        patches: Tensor of shape [B, C, H_p, W_p] (batch of patches).
        T_matrices: Tensor of shape [B, 3, 3] (batch of transformation matrices).
        images: Tensor of shape [B, C, H_i, W_i] (batch of images).
    Returns:
        Tensor of manipulated images of shape [B, C, H_i, W_i].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patches = patches.to(device)
    T_matrices = T_matrices.to(device)
    images = images.to(device)

    batch_size, _, p_height, p_width = patches.shape
    # _, _, i_height, i_width = images.shape

    i_height, i_width = images.shape[-2::]
    while len(images.shape) < 4:
        images = images.unsqueeze(0)

    # Create masks for patches
    masks = torch.ones_like(patches, dtype=torch.float32, device=device)

    # print(T_matrices[0])

    # Invert transformation matrices
    inv_T_matrices = torch.inverse(T_matrices)

    # Flatten transformation matrices for grid computation
    # print(inv_T_matrices.shape)
    coeffs = inv_T_matrices.reshape(batch_size, -1)
    # print(coeffs.shape)

    # Generate perspective grids for batch
    grids = _perspective_grid(
        coeffs, w=p_width, h=p_height, ow=i_width, oh=i_height, 
        dtype=torch.float32, device=device, center=[1., 1.]
    )

    # Apply grid sampling for patches and masks
    transformed_patches = torch.nn.functional.grid_sample(
        patches, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    )
    bit_masks = torch.nn.functional.grid_sample(
        masks, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    ).bool()

    # Combine transformed patches with original images
    manipulated_images = images * ~bit_masks
    manipulated_images += transformed_patches

    return manipulated_images

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

def train_batch_overfit(model_name='frontnet', corpus_size=1000, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    if os.path.exists(f"{project_root}/{model_name}/corpus_{model_name}.npz"):
        # ==========================================================================
        # 1. LOAD PRE-SAVED CORPUS
        # ==========================================================================
        print(f"Loading pre-saved corpus for {model_name}...")
        data = np.load(f"{project_root}/{model_name}/corpus_{model_name}.npz")
        patches_np = data['patches'].astype(np.float32)[:corpus_size]
        targets_np = data['targets'].astype(np.float32)[:corpus_size]
        conds_np = data['conds'].astype(np.float32)[:corpus_size]
        patch_h, patch_w = patches_np.shape[1], patches_np.shape[2]
        print(f"Loaded dataset: {patches_np.shape[0]} samples of size ({patch_h}, {patch_w})")


    else:
        # ==========================================================================
        # 1. LOAD DATA
        # ==========================================================================
        sample_paths = sorted(glob.glob(f"{project_root}/{model_name}/temp_pid_*/sample_*.npz"))
        if not sample_paths:
            print("Error: no samples found in temp_pid_* folders.")
            return

        
        patches_list, targets_list, conds_list = [], [], []

        print(f"Loading {len(sample_paths)} samples...")
        for p in sample_paths:
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

        patch_h, patch_w = patches_np.shape[1], patches_np.shape[2]
        print(f"Loaded dataset: {patches_np.shape[0]} samples of size ({patch_h}, {patch_w})")

        # Save corpus for later reuse
        np.savez(f"{project_root}/{model_name}/corpus_{model_name}.npz", patches=patches_np, targets=targets_np, conds=conds_np)

        max_samples = 1000
        patches_np = patches_np[:max_samples]
        targets_np = targets_np[:max_samples]
        conds_np = conds_np[:max_samples]
        corpus_size = patches_np.shape[0]

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
        plt.savefig(f"{project_root}/overfit_results/{model_name}__{corpus_size}_overfit_data_histograms.png")
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
        plt.savefig(f"{project_root}/overfit_results/{model_name}__{corpus_size}_overfit_data_scatter.png")
        plt.close()

    # ==========================================================================
    # 2. CREATE DATALOADER
    # ==========================================================================
    dataset = PatchDataset(patches_np, targets_np, conds_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    # B. Frontnet (The Critic/Victim)
    if model_name == 'frontnet':
        frontnet_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Models/Frontnet160x32.pt")
        frontnet = load_model(frontnet_path, device, config="160x32")
        frontnet.eval()
        # Freeze Frontnet completely (we only want gradients to flow through it to the patch)
        for p in frontnet.parameters():
            p.requires_grad = False
    if model_name == 'yolov5':
        from yolo_bounding import YOLOBox
        yolo = YOLOBox()
        yolo.model.eval()
        for p in yolo.model.parameters():
            p.requires_grad = False

    # C. Background Images (For Projection)
    # data_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle")
    bg_dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)


    # ==========================================================================
    # 3. INITIALIZE MODEL
    # ==========================================================================
    model_wrapper = DiffusionModel(
        device=device,
        patch_size=(patch_h, patch_w),
        prediction_model_name=model_name
    )
    optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=1e-3)

    # ==========================================================================
    # 4. TRAINING LOOP WITH DATALOADER
    # ==========================================================================
    print("Starting training with DataLoader...")
    model_wrapper.model.train()
    
    num_epochs = 1000  # Multiple epochs to process 1000 samples
    step_count = 0

    # book keeping
    all_losses = []
    all_avg_losses = []
    all_recon_losses = []
    all_control_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_patches, batch_conds in dataloader:
            # print(step_count)
            # step_count += 1
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
            reconstruction_loss = F.mse_loss(denoised_guess, batch_patches)
            # print("Reconstruction loss:", reconstruction_loss.item())

            bg_imgs, _ = next(iter(bg_dataset))
            bg_imgs = bg_imgs.to(device)
            manipulated_images = []
            for i in range(batch_size):
                img = bg_imgs[i].unsqueeze(0) / 255. # (1, C, H, W)
                # print("Background image min/max:", img.min().item(), img.max().item(), img.shape)
                patch = denoised_guess[i].unsqueeze(0)  # (1, 1, H_p, W_p), between -1 and 1
                # print("Patch min/max:", patch.min().item(), patch.max().item())
                
                patch_normalized = (patch + 1.0) / 2.0  # Normalize patch to [0, 1] for projection
                # print("Patch normalized min/max:", patch_normalized.min().item(), patch_normalized.max().item(), patch_normalized.shape)

                sf = batch_conds[i, 0].item() + torch.randn(1).item() * 0.05  # slight noise to scale factor
                tx = batch_conds[i, 1].item() + torch.randn(1).item() * 2.0   # slight noise to translation x
                ty = batch_conds[i, 2].item() + torch.randn(1).item() * 2.0   # slight noise to translation y
                
                # Construct Transformation Matrix T
                T = torch.tensor([
                    [sf, 0., tx],
                    [0., sf, ty],
                    [0., 0., 1.]
                ], dtype=torch.float32, device=device)

                if model_name == 'yolov5':
                    img = torch.nn.functional.interpolate(img, size=(320, 640), mode='bilinear', align_corners=False)
                
                # Project patch onto image
                manipulated_img = project_patch(patch_normalized, T.unsqueeze(0), img.unsqueeze(0))  # (1, C, H, W)
                manipulated_images.append(manipulated_img.squeeze(0))

            
            manipulated_images = torch.stack(manipulated_images, dim=0)  # (B, C, 1, H, W)
            manipulated_images = manipulated_images.squeeze(2)  # (B, C, H, W)
            # add gaussian noise to manipulated images
            manipulated_images = manipulated_images + torch.randn_like(manipulated_images) * 0.01
            manipulated_images = torch.clamp(manipulated_images, 0., 1.)
            if model_name == 'yolov5':
                manipulated_images = manipulated_images.repeat_interleave(3, dim=1)  # (B, 3, H, W)
            
            # print("Manipulated images min/max:", manipulated_images.min().item(), manipulated_images.max().item(), manipulated_images.shape) # (B, C, H, W)
            
            target_positions = batch_conds[:, 3:7]  # (B, 4)
            # E. Frontnet Prediction on Manipulated Images
            if model_name == 'frontnet':
                x, y, z, yaw = frontnet(manipulated_images*255.)
                # print("x, y, z, yaw:", x, y, z, yaw)
                prediction = torch.stack([x, y, z, yaw])
                prediction = prediction.squeeze(2).mT
                x_loss = F.mse_loss(prediction[:, 0], target_positions[:, 0])
                y_loss = F.mse_loss(prediction[:, 1], target_positions[:, 1])
                z_loss = F.mse_loss(prediction[:, 2], target_positions[:, 2])

                dist_loss = x_loss + y_loss + (10.0 * z_loss)
                ang_loss = (1 - torch.cos(normalize_yaw_t(prediction[:, 3]) - normalize_yaw_t(target_positions[:, 3]))).mean()
                control_loss = dist_loss + ang_loss

            if model_name == 'yolov5':
                random_drone_pose = torch.zeros(batch_size, 4, device=device) # pose in world frame
                random_drone_pose[:, 0] = random_drone_pose[:, 0].uniform_(-1.0, 1.0)  # x
                random_drone_pose[:, 1] = random_drone_pose[:, 1].uniform_(-1.0, 1.0)  # y
                random_drone_pose[:, 2] = random_drone_pose[:, 2].uniform_(0.5, 1.5)  # z
                random_drone_pose[:, 3] = random_drone_pose[:, 3].uniform_(-0.3, 0.3)  # yaw
                predicted_boxes, _ = yolo(manipulated_images, target_anchor=None)
                scaled_box = predicted_boxes.clone()
                # scale back to 160x96
                scaled_box[:, [0, 2]] *= (160.0 / 640.)  # x coords
                scaled_box[:, [1, 3]] *= (96.0 / 320.)   # y coords

                prediction = yolo.cam.batch_xyz_from_boxes(scaled_box)
                
                yaw_list = []
                for pred, drone_pose in zip(prediction, random_drone_pose):
                    T_pred_drone = T_matrix(pred)
                    T_drone_world = T_matrix(drone_pose)
                    T_pred_world = T_drone_world @ T_pred_drone
                    delta = T_pred_world[:3, 3] - drone_pose[:3]
                    yaw = torch.atan2(delta[1], delta[0]) * -1.
                    yaw_list.append(yaw)
                yaw_tensor = torch.stack(yaw_list).to(device)
                prediction = torch.cat([prediction[:, :3], yaw_tensor.unsqueeze(1)], dim=1)
                # print("Prediction:", prediction)
                
                control_loss = F.mse_loss(prediction[:, :3], target_positions[:, :3])
                angular_loss = (1 - torch.cos(normalize_yaw_t(prediction[:, 3]) - normalize_yaw_t(target_positions[:, 3]))).mean()
                control_loss = control_loss + angular_loss
                # control_loss = control_loss * 20.
           

            all_control_losses.append(control_loss.item())
            all_recon_losses.append(reconstruction_loss.item())

            loss = control_loss#reconstruction_loss + control_loss
            
            all_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # step_count += 1
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        all_avg_losses.append(avg_epoch_loss)
        if epoch % 10 == 0:
                tqdm.write(f"Step {epoch}, Avg Loss: {avg_epoch_loss:.6f}, Last reconstruction Loss: {reconstruction_loss.item():.6f}, Last Control Loss: {control_loss.item():.6f}")
        
        # print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")


    # ==========================================================================
    # 5. SAVE LOSS CURVES
    # ==========================================================================

    plt.figure(figsize=(10,5))
    plt.plot(all_losses, label='Total Loss', alpha=0.5)
    plt.plot(all_recon_losses, label='Reconstruction Loss', alpha=0.5)
    plt.plot(all_control_losses, label='Control Loss', alpha=0.5)
    plt.plot(all_avg_losses, label='Avg Epoch Loss', color='blue', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{project_root}/overfit_results/loss_curves_{model_name}_{corpus_size}.png")
    plt.close()

    np.savez(f"{project_root}/overfit_results/loss_values_{model_name}_{corpus_size}.npz",
             total_loss=np.array(all_losses),
             recon_loss=np.array(all_recon_losses),
             control_loss=np.array(all_control_losses),
             avg_epoch_loss=np.array(all_avg_losses)
    )
    
    
    # ==========================================================================
    # 5. EVALUATION
    # ==========================================================================
    print("Training done. Generating samples...")
    model_wrapper.model.eval()
    os.makedirs(f"{project_root}/overfit_results", exist_ok=True)
    torch.save(model_wrapper.model.state_dict(), f"{project_root}/overfit_results/diffusion_model_{model_name}_{corpus_size}.pth")

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
            # print(samples_np.shape, samples_np.min(), samples_np.max())

            # Visualize
            fig, ax = plt.subplots(1, n_samples + 1, figsize=(12, 4))
            
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
            plt.savefig(f"{project_root}/overfit_results/result_{model_name}_{corpus_size}_{idx}.png")
            plt.close()

    all_psnr = np.array(all_psnr)
    print(f"Overall Mean PSNR: {all_psnr.mean():.2f} dB")
    np.save(f"{project_root}/overfit_results/psnr_scores_{model_name}_{corpus_size}.npy", all_psnr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet', help='Prediction model name')
    parser.add_argument('--corpus_size', type=int, default=1000, help='Number of samples to use for overfitting')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()

    model_name = args.model

    train_batch_overfit(model_name=model_name, corpus_size=args.corpus_size, batch_size=args.batch_size)