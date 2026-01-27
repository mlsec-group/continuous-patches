import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle
from simulation import T_matrix, normalize_yaw, calc_heading_vec
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# --- Differentiable Grid Sample ---
def _perspective_grid(coeffs, w, h, ow, oh, dtype, device):
    batch_size = coeffs.shape[0]
    theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)
    theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0).reshape(batch_size, 2, 3)

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

    return output_grid1.div_(output_grid2).sub_(1.0).view(batch_size, oh, ow, 2)

def project_patch(patches, T_matrices, images):
    """Differentiable projection of patches onto images."""
    device = patches.device
    batch_size, _, p_height, p_width = patches.shape
    i_height, i_width = images.shape[-2:]
    
    inv_T = torch.inverse(T_matrices)
    coeffs = inv_T.reshape(batch_size, -1)
    
    grids = _perspective_grid(coeffs, p_width, p_height, i_width, i_height, torch.float32, device)
    
    transformed_patches = F.grid_sample(patches, grids, align_corners=False, padding_mode='zeros')
    masks = torch.ones_like(patches)
    transformed_masks = F.grid_sample(masks, grids, align_corners=False, padding_mode='zeros')
    
    return images * (1 - transformed_masks) + transformed_patches

# --- Pose Calculation Helper (Shared) ---
def get_pose_from_prediction(model_wrapper, image, current_drone_pose, cam_params):
    """
    Standardized logic to go from (Image) -> (Model Output) -> (World Pose).
    Used by BOTH the Optimizer (for loss) and the Simulation Loop (for control).
    """
    device = current_drone_pose.device
    
    # 1. Frontnet Logic
    if model_wrapper.name == 'frontnet':
        # Output: (B, 4) -> x, y, z, yaw
        pred_raw = model_wrapper.predict(image)
        
        T_drone = T_matrix(current_drone_pose)
        T_pred_drone = T_matrix(pred_raw[0])
        T_pred_world = T_drone @ T_pred_drone
        
        # Yaw logic: Frontnet predicts relative yaw, calculate heading vector
        target_yaw = normalize_yaw(pred_raw[0, 3])
        T_dir = torch.eye(4, device=device)
        T_dir[:3, 3] = calc_heading_vec(1.0, normalize_yaw(target_yaw - torch.pi), device).squeeze()
        
        T_setpoint = T_dir @ T_pred_world
        final_yaw = target_yaw
        
    # 2. YOLO Logic
    elif model_wrapper.name == 'yolov5':
        # Output: Boxes
        boxes, _ = model_wrapper.predict(image)
        
        if boxes.shape[0] == 0: 
            # Fallback if detection lost (stay in place)
            return current_drone_pose 

        # Scale 640 -> 160 metric (Intrinsics match 160x96)
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= (160.0 / 640.0)
        scaled_boxes[:, [1, 3]] *= (96.0 / 320.0)
        
        # Box -> XYZ (in Drone Frame)
        xyz_yaw = cam_params.batch_xyz_from_boxes(scaled_boxes)
        
        T_drone = T_matrix(current_drone_pose)
        T_rec_drone = T_matrix(xyz_yaw[0])
        T_rec_world = T_drone @ T_rec_drone
        
        # Calculate Vector from Drone -> Box World Position
        delta = T_rec_world[:3, 3] - current_drone_pose[:3]
        
        # Drone should face the box
        final_yaw = torch.atan2(delta[1], delta[0]) * -1.0
        
        # Heading Vector
        T_dir = torch.eye(4, device=device)
        T_dir[:3, 3] = calc_heading_vec(1.0, normalize_yaw(final_yaw - torch.pi), device).squeeze()
        
        T_setpoint = T_dir @ T_rec_world

    return torch.cat([T_setpoint[:3, 3], final_yaw.unsqueeze(0)])

# --- Attacker Class ---
class Attacker:
    def __init__(self, args, model_wrapper, device, cam_params):
        self.args = args
        self.mode = args.patch_mode
        self.model = model_wrapper
        self.device = device
        self.cam_params = cam_params
        
        self.patch_size = (150, 320) if args.model == 'yolov5' else (45, 80)
        self.last_patch = None
        
        # Load Diffusion/Corpus models
        if self.mode == 'diffusion':
            from diffusion.diffusion_model import DiffusionModel
            self.diff_model = DiffusionModel(device=device, patch_size=self.patch_size, prediction_model_name=args.model)
            weight_path = f"{project_root}/overfit_results/diffusion_model_{args.model}_1000.pth"
            self.diff_model.load(weight_path)
            self.diff_model.model.eval()
            if self.model.name == "frontnet":
                self.num_denoising_steps = 2
            elif self.model.name == "yolov5":
                num_denoising_steps = 50
            
        if self.mode in ['corpus', 'interpolation']:
            data = np.load(f"{project_root}/{self.model.name}/corpus_{self.model.name}.npz")
            patches_np = data['patches'].astype(np.float32)[:self.args.corpus_size]
            targets_np = data['targets'].astype(np.float32)[:self.args.corpus_size]
            conds_np = data['conds'].astype(np.float32)[:self.args.corpus_size]
            cond_vector = np.concatenate([conds_np, targets_np], axis=1) 
            patch_h, patch_w = patches_np.shape[1], patches_np.shape[2]
            assert (patch_h, patch_w) == self.patch_size, "Corpus patch size mismatch."

            self.corpus_patches = torch.tensor(patches_np, device=device, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
            self.corpus_conds = torch.tensor(cond_vector, device=device, dtype=torch.float32)  # (N, 7)

            # print(self.corpus_patches.shape, self.corpus_conds.shape)
            
    def generate(self, base_img, T, drone_pose, target_pose):
        """
        Generates and returns the patch.
        If mode is 'optimal', runs the optimization loop here.
        """
        # 1. No Patch Mode (Clean)
        if self.mode == 'none':
            return None

        # 2. Optimization Loop
        if self.mode in ['velo', 'timeout']:
            return self._optimize(base_img, T, drone_pose, target_pose)

        # 3. Generative / Baseline (Single Step)
        patch = self._get_static_patch(T, target_pose)
        return patch

    def _get_static_patch(self, T, target_pose):
        if self.mode == 'black': return torch.zeros((1, 1, *self.patch_size), device=self.device)
        if self.mode == 'white': return torch.ones((1, 1, *self.patch_size), device=self.device)
        if self.mode == 'random': return torch.rand((1, 1, *self.patch_size), device=self.device)

        if self.mode in ['diffusion', 'corpus', 'interpolation']:
            sf, tx, ty = T[0,0], T[0,2], T[1,2]
            cond = torch.tensor([[sf, tx, ty, *target_pose]], device=self.device) 
            
            if self.mode == 'diffusion':
                with torch.no_grad(): 
                    return self.diff_model.sample(1, cond, self.device, n_steps=self.num_denoising_steps)
            if self.mode == 'corpus':
                dists = torch.norm(self.corpus_conds - cond, dim=1)
                return self.corpus_patches[torch.argmin(dists)].unsqueeze(0)

        return torch.rand((1, 1, *self.patch_size), device=self.device)

    def _optimize(self, base_img, T, drone_pose, target_pose, max_iters=3000):
        """
        The Optimization Loop. 
        Minimizes distance between (Drone's Perceived Pose) and (Target Pose).
        """
        # Initialize patch
        if self.args.temperature == 'warm' and self.last_patch is not None:
            patch = self.last_patch.clone().detach().requires_grad_(True)
        else:
            patch = torch.rand((1, 1, *self.patch_size), device=self.device, requires_grad=True)
            
        opt = torch.optim.Adam([patch], lr=0.03)
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, end_factor=1., total_iters=max_iters//10)

        timeout = 1.0 / self.args.timeout if self.args.timeout != 0 else 1e10
        # print("Time limit for optimization (s):", timeout)
        start_t = time.time()
        
        best_patch = patch.detach().clone()
        best_dist = float('inf')

        # Optimization Loop
        for _ in range(max_iters): 
            if time.time() - start_t > timeout: break
            opt.zero_grad()

            if self.model.name == 'yolov5':
                base_img = torch.nn.functional.interpolate(base_img, size=(320, 640), mode='bilinear', align_corners=False)
            
            # print("base_img shape: ", base_img.shape, base_img.min(), base_img.max())
            # print("patch shape: ", patch.shape, patch.min(), patch.max())
            # print("T shape: ", T)


            # 1. Project
            manipulated = project_patch(patch, T.unsqueeze(0), base_img).clamp(0, 1)
            
            # 2. Predict (Get World Pose)
            # We use the SHARED logic so the optimizer minimizes the EXACT metric used for control
            pred_pose = get_pose_from_prediction(self.model, manipulated, drone_pose, self.cam_params)
            # print("pred_pose: ", pred_pose)

            # 3. Loss (Targeting)
            # We want the drone to think it is at 'target_pose' (Kidnapping)
            # or we want it to think it is somewhere else? 
            # In 'optimal' mode as requested: "current point of target_trajectory is predicted"
            # This implies Loss = Dist(Predicted, Target)
            
            dist = torch.dist(pred_pose[:3], target_pose[:3])
            ang_loss = 1 - torch.cos(normalize_yaw(pred_pose[3]) - normalize_yaw(target_pose[3]))
            
            loss = dist + ang_loss
            
            if dist < best_dist:
                best_dist = dist.item()
                best_patch = patch.detach().clone()

            if dist < 0.01:
                best_dist = dist.item()
                best_patch = patch.detach().clone()
                break
                
            loss.backward()
            patch.grad = patch.grad.sign()
            opt.step()
            patch.data.clamp_(0., 1.)
            scheduler.step()

            print(f"Optimization Step: Loss={loss.item():.4f}, Pos Error={dist.item():.4f}, Ang Error={ang_loss.item():.4f}", end='\r')
            
        self.last_patch = best_patch
        return best_patch