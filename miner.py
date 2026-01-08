import os
import sys
import argparse
import multiprocessing as mp
import torch
import traceback
import numpy as np
import pickle
import torch.nn.functional as F
from tqdm import tqdm

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from util import load_model, load_dataset
from attack_minimal_single import project_patch, normalize_yaw_t

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def construct_T_matrix(sf, tx, ty, device):
    if not isinstance(sf, torch.Tensor): sf = torch.tensor(sf, device=device)
    if not isinstance(tx, torch.Tensor): tx = torch.tensor(tx, device=device)
    if not isinstance(ty, torch.Tensor): ty = torch.tensor(ty, device=device)
    
    if sf.ndim == 0:
        sf = sf.unsqueeze(0); tx = tx.unsqueeze(0); ty = ty.unsqueeze(0)
        
    B = sf.shape[0]
    T = torch.zeros((B, 3, 3), device=device)
    T[:, 0, 0] = sf
    T[:, 1, 1] = sf
    T[:, 0, 2] = tx
    T[:, 1, 2] = ty
    T[:, 2, 2] = 1.0
    return T

# ==============================================================================
# 2. WORKER FUNCTION
# ==============================================================================

def worker_mine_batch(args_dict):
    """
    Independent worker process.
    """
    pid = os.getpid()
    
    # --- 1. Thread Balancing ---
    # We set this INSIDE the process so it only affects this worker
    threads_per_worker = args_dict.get('threads', 1)
    torch.set_num_threads(threads_per_worker)
    
    try:
        import cv2
        cv2.setNumThreads(0) # Keep OpenCV single-threaded to avoid conflict
    except ImportError:
        pass

    print("PID", pid, "- Starting mining for", args_dict['n_samples'], "samples with batch size", args_dict['batch_size'], "and", threads_per_worker, "threads.", flush=True)
    # create temp pid dir for intermediate results (per model)
    model_name = args_dict.get('model', 'frontnet')
    os.makedirs(f"{model_name}/temp_pid_{pid}", exist_ok=True)
    try:
        # Unpack arguments
        n_samples = args_dict['n_samples']
        batch_size = args_dict['batch_size']
        threshold = args_dict['threshold']
        
        # Choose device: use CUDA for YOLO if available, otherwise CPU
        if model_name == 'yolov5' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            if model_name == 'yolov5':
                print(f"Warning: requested yolov5 but CUDA unavailable, running on CPU.")

        # --- 2. Load Model ---
        model_name = args_dict.get('model', 'frontnet')

        if model_name == 'frontnet':
            model_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Models/Frontnet160x32.pt")
            # Suppress load output
            sys_stdout_backup = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                frontnet = load_model(model_path, device, config="160x32")
            finally:
                sys.stdout = sys_stdout_backup
            frontnet.eval()
            for p in frontnet.parameters():
                p.requires_grad = False
            model = frontnet
        elif model_name == 'yolov5':
            # YOLOv5 wrapper defined in yolo_bounding.YOLOBox
            from yolo_bounding import YOLOBox
            model = YOLOBox()
            model.eval()
            # ensure cam on correct device
            try:
                model.cam.device = device
            except Exception:
                pass

        # --- 3. Load Data ---
        data_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle")
        
        # num_workers=0 is MANDATORY to prevent recursive forking inside a worker
        bg_dataset = load_dataset(
            data_path, 
            batch_size=batch_size, 
            shuffle=True, 
            train=True, 
            IMRC=True, 
            num_workers=0,
            drop_last=True  
        )
        #bg_iter = iter(bg_dataset)
        
        # local_gold_data = []
        succes_count = 0
        
        # --- 4. Mining Loop ---
        while succes_count < n_samples:
            
            # A. Conditioning
            base_sf = np.random.uniform(0.5, 1.3)
            if model_name == 'frontnet':
                base_tx = np.random.uniform(-20., 160.)
                base_ty = np.random.uniform(-10., 96.)
            if model_name == 'yolov5':
                base_tx = np.random.uniform(-40., 320.)
                base_ty = np.random.uniform(-20., 160.)
            
            x_t = np.random.uniform(0., 1.5)
            y_t = np.random.uniform(-1, 1)
            z_t = np.random.uniform(-0.5, 0.5)
            yaw_t = np.random.uniform(-0.1, 0.1)
            target_vec = torch.tensor([x_t, y_t, z_t, yaw_t], device=device)
            
            # B. Init Patch
            if model_name == 'yolov5':
                patch = torch.rand((1, 1, 150, 320), device=device, requires_grad=True)
            else:
                patch = torch.rand((1, 1, 45, 80), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([patch], lr=1e-2)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1., total_iters=300)
            
            success = False
            
            # C. Optimization (150 steps)
            for step in range(1000): 
                optimizer.zero_grad()
                
                # # Get Batch
                # try:
                #     batch_bg_imgs, _ = next(bg_iter)
                # except StopIteration:
                #     bg_iter = iter(bg_dataset)
                #     batch_bg_imgs, _ = next(bg_iter)
                
                # # Check for uneven batch size at end of epoch
                # if batch_bg_imgs.shape[0] != batch_size:
                #     continue # Skip partial batches to keep shapes consistent
                batch_bg_imgs, _ = next(iter(bg_dataset))

                batch_bg_imgs = batch_bg_imgs.to(device) / 255.0
                # curr_batch = batch_bg_imgs.shape[0]
                
                # Jitter
                jitter_sf = torch.tensor(base_sf, device=device) + torch.empty(batch_size, device=device).uniform_(-0.01, 0.01)
                jitter_tx = torch.tensor(base_tx, device=device) + torch.empty(batch_size, device=device).uniform_(-0.3, 0.3)
                jitter_ty = torch.tensor(base_ty, device=device) + torch.empty(batch_size, device=device).uniform_(-0.3, 0.3)
                
                T_matrices = construct_T_matrix(jitter_sf, jitter_tx, jitter_ty, device)

                patch_batch = patch.repeat(batch_size, 1, 1, 1)
                
                # Forward
                if model_name == 'yolov5':
                    batch_bg_imgs = F.interpolate(batch_bg_imgs, size=(320, 640), mode='bilinear', align_corners=False)
                manipulated_batch = project_patch(patch_batch, T_matrices, batch_bg_imgs)
                manipulated_batch = manipulated_batch + torch.randn_like(manipulated_batch) * 0.01
                manipulated_batch.clamp_(0., 1.)

                # Model-specific forward + loss
                if model_name == 'frontnet':
                    x, y, z, yaw = model(manipulated_batch * 255.)
                    preds = torch.stack([x, y, z, yaw], dim=1).squeeze(2)
                    targets_batch = target_vec.unsqueeze(0).expand(batch_size, -1)
                    dist_loss = F.mse_loss(preds[:, :3], targets_batch[:, :3])
                    ang_loss = (1 - torch.cos(normalize_yaw_t(preds[:, 3]) - normalize_yaw_t(targets_batch[:, 3]))).mean()
                    total_loss = dist_loss + ang_loss
                if model_name == 'yolov5':
                    
                    manipulated_rgb = manipulated_batch.repeat_interleave(3, dim=1)

                    # Move inputs to model device if needed
                    manipulated_rgb = manipulated_rgb.to(device)

                    # YOLOBox returns (boxes_xyxy, scores)
                    predicted_boxes, predicted_scores = model(manipulated_rgb)

                    # If no boxes detected, give high loss and continue
                    if predicted_boxes is None or predicted_boxes.shape[0] == 0:
                        total_loss = torch.tensor(1e3, device=device)
                    else:
                        # scale predictions back to 160x96
                        scaled_box = predicted_boxes.clone()
                        scaled_box[:, [0, 2]] *= (160.0 / 640.0)
                        scaled_box[:, [1, 3]] *= (96.0 / 320.0)

                        # Convert boxes to xyz+yaw via camera model
                        try:
                            recovered = model.cam.batch_xyz_from_boxes(scaled_box)
                        except Exception:
                            recovered = None

                        if recovered is None:
                            total_loss = torch.tensor(1e3, device=device)
                        else:
                            # recovered: (B,4) -> x,y,z,yaw
                            preds = recovered.to(device)
                            targets_batch = target_vec.unsqueeze(0).expand(preds.shape[0], -1).to(device)
                            dist_loss = F.mse_loss(preds[:, :3], targets_batch[:, :3])
                            ang_loss = (1 - torch.cos(normalize_yaw_t(preds[:, 3]) - normalize_yaw_t(targets_batch[:, 3]))).mean()
                            total_loss = dist_loss + ang_loss
                
                # If loss is a tensor, get scalar safely
                try:
                    loss_val = float(total_loss.item())
                except Exception:
                    loss_val = float(total_loss)

                if loss_val < threshold:
                    success = True
                    break
                
                total_loss.backward()
                patch.grad = patch.grad.sign()
                optimizer.step()
                patch.data.clamp_(0., 1.)
                scheduler.step()
            
                if step % 50 == 0:
                    print(f"[PID {pid}] Step {step} - Loss: {loss_val:.4f}", flush=True)
                
            if success:
                patch_np = patch.detach().cpu().numpy().squeeze()
                target_np = target_vec.cpu().numpy()
                condition_np = np.array([base_sf, base_tx, base_ty])
                # local_gold_data.append((patch_np, target_np, condition_np))
                # Save intermediate results
                np.savez_compressed(f"{model_name}/temp_pid_{pid}/sample_{succes_count}.npz", patch=patch_np, target=target_np, condition=condition_np)
                succes_count += 1
            print(f"[PID {pid}] Local Gold Samples: {succes_count}/{n_samples} - Last Loss: {total_loss.item():.4f}", flush=True)

        return succes_count

    except Exception as e:
        print(f"\n[PID {pid}] CRITICAL FAILURE:", file=sys.stderr)
        traceback.print_exc()
        return succes_count

# ==============================================================================
# 3. MANAGER
# ==============================================================================

def run_parallel_mining(total_samples, output_path, threads_per_worker=2, batch_size=32, model_name='frontnet'):
    # 1. Detect Resources
    total_cores = os.cpu_count()
    
    # 2. Calculate Split
    # Leave 2 cores free for system overhead
    # available_cores = max(1, total_cores - 10)
    available_cores = 10
    
    # Calculate number of workers based on threads per worker
    # e.g. 110 cores / 2 threads = 55 workers
    num_workers = max(1, available_cores // threads_per_worker)
    
    # Recalculate samples per worker
    samples_per_worker = total_samples // num_workers
    remainder = total_samples % num_workers
    
    print(f"\n=== Resource Allocation ===")
    print(f"  Total System Cores:   {total_cores}")
    print(f"  Threads per Worker:   {threads_per_worker}")
    print(f"  Concurrent Workers:   {num_workers}")
    print(f"  Total Cores Utilized: {num_workers * threads_per_worker}")
    print(f"  Samples per Worker:   ~{samples_per_worker}")
    print(f"===========================\n")

    # 3. Prevent Global Thread Pools (Critical for main process too)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

    # Set Start Method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 
    
    # Create arguments
    worker_args = []
    for i in range(num_workers):
        count = samples_per_worker + (1 if i < remainder else 0)
        worker_args.append({
            'n_samples': count,
            'batch_size': batch_size,
            'threshold': 0.01,
            'threads': threads_per_worker
        })
        # include model_name in args so workers can load correct model
        worker_args[-1]['model'] = model_name
    
    print("Spawning pool...", flush=True)
    
    with mp.Pool(processes=num_workers) as pool:
        # Tqdm tracks completed WORKERS (lists of patches), not individual patches
        results = list(tqdm(pool.imap_unordered(worker_mine_batch, worker_args), total=num_workers))
    
    final_gold_data = []
    for res in results:
        final_gold_data.extend(res)
        
    print(f"Mining complete. Saving {len(final_gold_data)} samples to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(final_gold_data, f)
    print("Done.")


def run_yolo_sequential(start_idx, end_idx, threshold=0.01):
    """Run YOLO mining sequentially on the main process using CUDA.
    start_idx and end_idx define inclusive indexes; total patches = end_idx - start_idx + 1
    """
    if start_idx is None or end_idx is None:
        raise ValueError("start_idx and end_idx must be provided for yolov5 runs")
    if end_idx < start_idx:
        raise ValueError("end_idx must be >= start_idx")

    total = end_idx - start_idx + 1
    print(f"Running YOLO sequential mining for {total} patches (indexes {start_idx}..{end_idx}) on CUDA if available.")

    # Build args dict for a single worker run
    args = {
        'n_samples': total,
        'batch_size': 128,
        'threshold': threshold,
        'threads': 1,
        'model': 'yolov5'
    }

    # Run worker in-process (sequential)
    counter = worker_mine_batch(args)

    # Save results
    # final_gold_data = []
    # final_gold_data.extend(results)
    # os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    # with open(output_path, 'wb') as f:
    #     pickle.dump(final_gold_data, f)

    print(f"YOLO mining complete. Saved {counter} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", type=str, default="gold_corpus_robust.pickle")
    # New argument to control thread balancing
    parser.add_argument("--threads", type=int, default=2, help="Number of threads per worker process.")
    parser.add_argument("--model", type=str, choices=["frontnet", "yolov5"], default="frontnet", help="Model to use for prediction.")
    parser.add_argument("--start-index", type=int, default=None, help="Start index for yolov5 patch generation (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index for yolov5 patch generation (inclusive)")
    
    args = parser.parse_args()
    
    if args.model == 'frontnet':
        # default batch size kept high for frontnet
        run_parallel_mining(args.samples, args.output, threads_per_worker=args.threads, batch_size=128, model_name=args.model)
    else:
        # yolov5: run sequentially on CUDA and use start/end indices to control how many patches
        start_idx = args.start_index
        end_idx = args.end_index
        if start_idx is None or end_idx is None:
            # derive from samples if not provided
            start_idx = 0
            end_idx = max(0, args.samples - 1)
        run_yolo_sequential(start_idx, end_idx, threshold=0.05)












# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import argparse
# import pickle
# import os
# import sys
# from tqdm import tqdm
# from pathlib import Path

# # --- Path Setup ---
# # Ensure we can import from parent/sibling directories
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # --- Local Imports ---
# # Assuming these exist in your project structure based on previous messages
# from util import load_model, load_dataset
# from attack_minimal_single import project_patch, normalize_yaw_t

# # ==============================================================================
# # 1. HELPER FUNCTIONS
# # ==============================================================================

# def construct_T_matrix(sf, tx, ty, device):
#     """Creates a batch of transformation matrices."""
#     # sf, tx, ty can be scalars or tensors.
#     # If they are tensors of shape (B,), output is (B, 3, 3)
    
#     if not isinstance(sf, torch.Tensor): sf = torch.tensor(sf, device=device)
#     if not isinstance(tx, torch.Tensor): tx = torch.tensor(tx, device=device)
#     if not isinstance(ty, torch.Tensor): ty = torch.tensor(ty, device=device)
    
#     # Handle batch dimension if inputs are 1D tensors
#     if sf.ndim == 0:
#         sf = sf.unsqueeze(0)
#         tx = tx.unsqueeze(0)
#         ty = ty.unsqueeze(0)
        
#     B = sf.shape[0]
#     T = torch.zeros((B, 3, 3), device=device)
    
#     # Scale
#     T[:, 0, 0] = sf
#     T[:, 1, 1] = sf
    
#     # Translation
#     T[:, 0, 2] = tx
#     T[:, 1, 2] = ty
#     T[:, 2, 2] = 1.0
    
#     return T

# def get_frontnet(device):
#     """Loads the Frontnet model in eval mode."""
#     # Adjust path as necessary
#     model_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Models/Frontnet160x32.pt")
#     model = load_model(model_path, device, config="160x32")
#     model.eval()
#     return model

# # ==============================================================================
# # 2. MINING LOGIC (The Robust Teacher)
# # ==============================================================================

# def mine_robust_patches(
#     n_samples=5000, 
#     save_path="gold_corpus_robust.pickle", 
#     batch_size=32,      # Number of random variations per optimization step
#     max_steps=200,      # Max steps to find a solution
#     loss_threshold=0.15 # Loss to accept a patch as "Gold"
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Mining on {device}...")

#     # 1. Load Victim Model
#     frontnet = get_frontnet(device)
    
#     # 2. Load Background Dataset (for robustness)
#     # Adjust path as necessary
#     data_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle")
#     bg_dataset = load_dataset(data_path, batch_size=batch_size, shuffle=True, train=True, IMRC=True)
    
#     gold_data = []
    
#     print(f"Starting mining loop. Goal: {n_samples} patches.")
#     print(f"Robustness: EoT {batch_size}x (jitter & background) per step.")

#     # We use a progress bar that updates manually when we find a good sample
#     pbar = tqdm(total=n_samples)
    
#     while len(gold_data) < n_samples:
        
#         # --- A. Generate Random Scenario (Conditioning) ---
#         # The Diffusion model needs to learn: "Given this pos/target, what works?"
        
#         # Position Conditioning
#         base_sf = np.random.uniform(0.5, 1.0)
#         base_tx = np.random.uniform(0., 120.) # Valid range for Frontnet 160 width
#         base_ty = np.random.uniform(0., 60.)  # Valid range for Frontnet 96 height
        
#         # Target Output (The behavior we want to force)
#         # Randomly sample a valid drone state
#         x_t = np.random.uniform(0., 1.5)
#         y_t = np.random.uniform(-1, 1)
#         z_t = np.random.uniform(-0.5, 0.5)
#         yaw_t = np.random.uniform(-0.3, 0.3)
        
#         target_vec = torch.tensor([x_t, y_t, z_t, yaw_t], device=device)
        
#         # --- B. Initialize Patch ---
#         # Start from random noise
#         patch = torch.rand((1, 1, 45, 80), device=device, requires_grad=True)
#         optimizer = torch.optim.Adam([patch], lr=0.05)
        
#         success = False

#         while len(gold_data) < n_samples:
        

# #             optimizer.zero_grad()
# #             
            
# #             # --- Expectation over Transformation (EoT) ---
# #             # Instead of optimizing for 1 background, we average loss over 'eot_samples'
# #             # variations. This creates a patch that works "in general".
            
# #             # # 1. Sample 'eot_samples' random backgrounds indices
# #             # bg_indices = np.random.randint(0, len(bg_dataset), size=eot_samples)
# #             # batch_bg_imgs = []
# #             # for idx in bg_indices:
# #             #     img = bg_dataset.dataset[idx][0].to(device) # (1, 96, 160)
# #             #     batch_bg_imgs.append(img)
# #             # batch_bg_imgs = torch.stack(batch_bg_imgs).float() / 255.0 # (B, 1, 96, 160)
            
# #             # 2. Create Jittered Transformations (Robustness to position error)
# #             # Add small noise to sf, tx, ty for each sample in the batch
# #             jitter_sf = torch.tensor(base_sf, device=device) + torch.empty(32, device=device).uniform_(-0.1, 0.1)
# #             jitter_tx = torch.tensor(base_tx, device=device) + torch.empty(32, device=device).uniform_(-3.0, 3.0)
# #             jitter_ty = torch.tensor(base_ty, device=device) + torch.empty(32, device=device).uniform_(-3.0, 3.0)
            
# #             T_matrices = construct_T_matrix(jitter_sf, jitter_tx, jitter_ty, device)

# #             patch_batch = patch.repeat(32, 1, 1, 1)  # (B, 1, H, W)
            
# #             # 3. Project Patch (Batch Operation)
# #             # Patch (1,1,H,W) expands to match batch size in project_patch
# #             manipulated_batch = project_patch(patch_batch, T_matrices, batch_bg_imgs)
        
#         # --- C. Robust Optimization Loop ---
#         for step in range(max_steps):
#             optimizer.zero_grad()
            
#             # --- Expectation over Transformation (EoT) ---
#             # Instead of optimizing for 1 background, we average loss over 'eot_samples'
#             # variations. This creates a patch that works "in general".
            
#             # # 1. Sample 'eot_samples' random backgrounds indices
#             # bg_indices = np.random.randint(0, len(bg_dataset), size=eot_samples)
#             # batch_bg_imgs = []
#             # for idx in bg_indices:
#             #     img = bg_dataset.dataset[idx][0].to(device) # (1, 96, 160)
#             #     batch_bg_imgs.append(img)
#             # batch_bg_imgs = torch.stack(batch_bg_imgs).float() / 255.0 # (B, 1, 96, 160)
#             batch_bg_imgs, _ = next(iter(bg_dataset))
#             batch_bg_imgs = batch_bg_imgs.to(device).float() / 255.0  # (B, 1, 96, 160)
            
#             # 2. Create Jittered Transformations (Robustness to position error)
#             # Add small noise to sf, tx, ty for each sample in the batch
#             jitter_sf = torch.tensor(base_sf, device=device) + torch.empty(batch_size, device=device).uniform_(-0.05, 0.05)
#             jitter_tx = torch.tensor(base_tx, device=device) + torch.empty(batch_size, device=device).uniform_(-3.0, 3.0)
#             jitter_ty = torch.tensor(base_ty, device=device) + torch.empty(batch_size, device=device).uniform_(-3.0, 3.0)
            
#             T_matrices = construct_T_matrix(jitter_sf, jitter_tx, jitter_ty, device)

#             patch_batch = patch.repeat(batch_size, 1, 1, 1)  # (B, 1, H, W)
            
#             # 3. Project Patch (Batch Operation)
#             # Patch (1,1,H,W) expands to match batch size in project_patch
#             manipulated_batch = project_patch(patch_batch, T_matrices, batch_bg_imgs)
            
#             # 4. Add visual noise (Sensor noise robustness)
#             manipulated_batch = manipulated_batch + torch.randn_like(manipulated_batch) * 0.02
#             manipulated_batch.clamp_(0., 1.)
            
#             # 5. Victim Forward Pass
#             # Frontnet expects [0, 255]
#             x, y, z, yaw = frontnet(manipulated_batch * 255.)
#             preds = torch.stack([x, y, z, yaw], dim=1).squeeze(2) # (B, 4)
            
#             # 6. Loss Calculation (Average over batch)
#             # Expand target to batch size
#             targets_batch = target_vec.unsqueeze(0).expand(batch_size, -1)
            
#             dist_loss = F.mse_loss(preds[:, :3], targets_batch[:, :3])
#             ang_loss = (1 - torch.cos(normalize_yaw_t(preds[:, 3]) - normalize_yaw_t(targets_batch[:, 3]))).mean()
            
#             total_loss = dist_loss + ang_loss
            
#             # 7. Check Success (on the robust average)
#             if total_loss.item() < loss_threshold:
#                 success = True
#                 break
                
#             total_loss.backward()
#             patch.grad = patch.grad.sign() # Fast Gradient Sign method for cleaner patterns
#             optimizer.step()
#             patch.data.clamp_(0., 1.)
            
#         # --- D. Save Data ---
#         if success:
#             # We save the CLEAN patch and the BASE condition.
#             # We do NOT save the jittered position. The diffusion model should learn
#             # that for this base position, this patch is the robust solution.
            
#             patch_np = patch.detach().cpu().numpy().squeeze() # (45, 80)
#             target_np = target_vec.cpu().numpy()              # (4,)
#             condition_np = np.array([base_sf, base_tx, base_ty]) # (3,)
            
#             gold_data.append((patch_np, target_np, condition_np))
#             pbar.update(1)
#             pbar.set_description(f"Loss: {total_loss.item():.4f}")
            
#             # Periodically save to disk to avoid data loss
#             if len(gold_data) % 500 == 0:
#                 with open(save_path, "wb") as f:
#                     pickle.dump(gold_data, f)

#     pbar.close()
    
#     # Final Save
#     print(f"Mining complete. Saving {len(gold_data)} samples to {save_path}...")
#     with open(save_path, "wb") as f:
#         pickle.dump(gold_data, f)
#     print("Done.")

# # ==============================================================================
# # 3. ENTRY POINT
# # ==============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Mine robust adversarial patches for Diffusion training.")
    
#     parser.add_argument("--samples", type=int, default=5000, help="Number of successful patches to mine.")
#     parser.add_argument("--output", type=str, default="gold_corpus_robust.pickle", help="Output pickle filename.")
#     parser.add_argument("--eot", type=int, default=5, help="Batch size for EoT (robustness). Higher = more robust but slower.")
#     parser.add_argument("--threshold", type=float, default=0.15, help="Loss threshold to accept a patch.")
    
#     args = parser.parse_args()
    
#     # Create directory if it doesn't exist
#     output_dir = os.path.dirname(args.output)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
        
#     mine_robust_patches(
#         n_samples=args.samples,
#         save_path=args.output,
#         eot_samples=args.eot,
#         loss_threshold=args.threshold
#     )

# Let's say I want to parallelize this. Load the model once but parallelize the optimization of the patches?

# Yes, absolutely. The most efficient way to parallelize this—given that the Frontnet model is extremely small—is to use CPU multiprocessing.

# Since Frontnet is lightweight, running multiple copies on the CPU avoids the complex overhead and potential errors of managing CUDA contexts across processes (which often leads to "CUDA initialization" errors or memory contention).

# Here is miner_parallel.py. It uses Python's multiprocessing to spawn independent workers.
# miner_parallel.py
# Python








# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import argparse
# import pickle
# import os
# import sys
# from tqdm import tqdm
# from pathlib import Path

# # --- Path Setup ---
# # Ensure we can import from parent/sibling directories
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # --- Local Imports ---
# # Assuming these exist in your project structure based on previous messages
# from util import load_model, load_dataset
# from attack_minimal_single import project_patch, normalize_yaw_t

# # ==============================================================================
# # 1. HELPER FUNCTIONS
# # ==============================================================================

# def construct_T_matrix(sf, tx, ty, device):
#     """Creates a batch of transformation matrices."""
#     # sf, tx, ty can be scalars or tensors.
#     # If they are tensors of shape (B,), output is (B, 3, 3)
    
#     if not isinstance(sf, torch.Tensor): sf = torch.tensor(sf, device=device)
#     if not isinstance(tx, torch.Tensor): tx = torch.tensor(tx, device=device)
#     if not isinstance(ty, torch.Tensor): ty = torch.tensor(ty, device=device)
    
#     # Handle batch dimension if inputs are 1D tensors
#     if sf.ndim == 0:
#         sf = sf.unsqueeze(0)
#         tx = tx.unsqueeze(0)
#         ty = ty.unsqueeze(0)
        
#     B = sf.shape[0]
#     T = torch.zeros((B, 3, 3), device=device)
    
#     # Scale
#     T[:, 0, 0] = sf
#     T[:, 1, 1] = sf
    
#     # Translation
#     T[:, 0, 2] = tx
#     T[:, 1, 2] = ty
#     T[:, 2, 2] = 1.0
    
#     return T

# def get_frontnet(device):
#     """Loads the Frontnet model in eval mode."""
#     # Adjust path as necessary
#     model_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Models/Frontnet160x32.pt")
#     model = load_model(model_path, device, config="160x32")
#     model.eval()
#     return model

# # ==============================================================================
# # 2. MINING LOGIC (The Robust Teacher)
# # ==============================================================================

# def mine_robust_patches(
#     n_samples=5000, 
#     save_path="gold_corpus_robust.pickle", 
#     eot_samples=5,      # Number of random variations per optimization step
#     max_steps=1000,      # Max steps to find a solution
#     loss_threshold=0.05 # Loss to accept a patch as "Gold"
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Mining on {device}...")

#     # 1. Load Victim Model
#     frontnet = get_frontnet(device)
    
#     # 2. Load Background Dataset (for robustness)
#     # Adjust path as necessary
#     data_path = os.path.join(project_root, "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle")
#     bg_dataset = load_dataset(data_path, batch_size=32, shuffle=True, train=True, IMRC=True)
    
#     gold_data = []
    
#     print(f"Starting mining loop. Goal: {n_samples} patches.")
#     print(f"Robustness: EoT {eot_samples}x (jitter & background) per step.")

#     # We use a progress bar that updates manually when we find a good sample
#     pbar = tqdm(total=n_samples)
    
#     while len(gold_data) < n_samples:
        
#         # --- A. Generate Random Scenario (Conditioning) ---
#         # The Diffusion model needs to learn: "Given this pos/target, what works?"
        
#         # Position Conditioning
#         base_sf = np.random.uniform(0.5, 1.0)
#         base_tx = np.random.uniform(-20., 120.) # Valid range for Frontnet 160 width
#         base_ty = np.random.uniform(-10., 60.)  # Valid range for Frontnet 96 height
        
#         # Target Output (The behavior we want to force)
#         # Randomly sample a valid drone state
#         x_t = np.random.uniform(0., 1.5)
#         y_t = np.random.uniform(-1, 1)
#         z_t = np.random.uniform(-0.5, 0.5)
#         yaw_t = np.random.uniform(-0.3, 0.3)
        
#         target_vec = torch.tensor([x_t, y_t, z_t, yaw_t], device=device)
        
#         # --- B. Initialize Patch ---
#         # Start from random noise
#         patch = torch.rand((1, 1, 45, 80), device=device, requires_grad=True)
#         optimizer = torch.optim.Adam([patch], lr=0.01)
#         scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=100)
        
#         success = False
        
#         # --- C. Robust Optimization Loop ---
#         for step in range(max_steps):
#             optimizer.zero_grad()
#             batch_bg_imgs, _ = next(iter(bg_dataset))
#             batch_bg_imgs = batch_bg_imgs.to(device).float() / 255.0  # (B, 1, 96, 160)
            
#             # --- Expectation over Transformation (EoT) ---
#             # Instead of optimizing for 1 background, we average loss over 'eot_samples'
#             # variations. This creates a patch that works "in general".
            
#             # # 1. Sample 'eot_samples' random backgrounds indices
#             # bg_indices = np.random.randint(0, len(bg_dataset), size=eot_samples)
#             # batch_bg_imgs = []
#             # for idx in bg_indices:
#             #     img = bg_dataset.dataset[idx][0].to(device) # (1, 96, 160)
#             #     batch_bg_imgs.append(img)
#             # batch_bg_imgs = torch.stack(batch_bg_imgs).float() / 255.0 # (B, 1, 96, 160)
            
#             # 2. Create Jittered Transformations (Robustness to position error)
#             # Add small noise to sf, tx, ty for each sample in the batch
#             jitter_sf = torch.tensor(base_sf, device=device) + torch.empty(32, device=device).uniform_(-0.1, 0.1)
#             jitter_tx = torch.tensor(base_tx, device=device) + torch.empty(32, device=device).uniform_(-3.0, 3.0)
#             jitter_ty = torch.tensor(base_ty, device=device) + torch.empty(32, device=device).uniform_(-3.0, 3.0)
            
#             T_matrices = construct_T_matrix(jitter_sf, jitter_tx, jitter_ty, device)

#             patch_batch = patch.repeat(32, 1, 1, 1)  # (B, 1, H, W)
            
#             # 3. Project Patch (Batch Operation)
#             # Patch (1,1,H,W) expands to match batch size in project_patch
#             manipulated_batch = project_patch(patch_batch, T_matrices, batch_bg_imgs)
            
#             # 4. Add visual noise (Sensor noise robustness)
#             manipulated_batch = manipulated_batch + torch.randn_like(manipulated_batch) * 0.02
#             manipulated_batch.clamp_(0., 1.)
            
#             # 5. Victim Forward Pass
#             # Frontnet expects [0, 255]
#             x, y, z, yaw = frontnet(manipulated_batch * 255.)
#             preds = torch.stack([x, y, z, yaw], dim=1).squeeze(2) # (B, 4)
            
#             # 6. Loss Calculation (Average over batch)
#             # Expand target to batch size
#             targets_batch = target_vec.expand(32, -1)

#             dist_loss = F.mse_loss(preds[:, :3], targets_batch[:, :3])
#             ang_loss = (1 - torch.cos(normalize_yaw_t(preds[:, 3]) - normalize_yaw_t(targets_batch[:, 3]))).mean()
            
#             total_loss = dist_loss + ang_loss

#             # print(f"Step {step}, Loss: {total_loss.item():.4f}")
            
#             # 7. Check Success (on the robust average)
#             if total_loss.item() < loss_threshold:
#                 success = True
#                 patch.data.clamp_(0., 1.)
#                 print(f"Found robust patch! Step {step}, Loss: {total_loss.item():.4f}")
#                 break
                
#             total_loss.backward()
#             patch.grad = patch.grad.sign() # Fast Gradient Sign method for cleaner patterns
#             optimizer.step()
#             scheduler.step()
#             patch.data.clamp_(0., 1.)
            
#         # --- D. Save Data ---
#         if success:
#             # We save the CLEAN patch and the BASE condition.
#             # We do NOT save the jittered position. The diffusion model should learn
#             # that for this base position, this patch is the robust solution.
            
#             patch_np = patch.detach().cpu().numpy().squeeze() # (45, 80)
#             target_np = target_vec.cpu().numpy()              # (4,)
#             condition_np = np.array([base_sf, base_tx, base_ty]) # (3,)
            
#             gold_data.append((patch_np, target_np, condition_np))
#             pbar.update(1)
#             pbar.set_description(f"Loss: {total_loss.item():.4f}")
            
#             # Periodically save to disk to avoid data loss
#             if len(gold_data) % 500 == 0:
#                 with open(save_path, "wb") as f:
#                     pickle.dump(gold_data, f)

#     pbar.close()
    
#     # Final Save
#     print(f"Mining complete. Saving {len(gold_data)} samples to {save_path}...")
#     with open(save_path, "wb") as f:
#         pickle.dump(gold_data, f)
#     print("Done.")

# # ==============================================================================
# # 3. ENTRY POINT
# # ==============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Mine robust adversarial patches for Diffusion training.")
    
#     parser.add_argument("--samples", type=int, default=5000, help="Number of successful patches to mine.")
#     parser.add_argument("--output", type=str, default="gold_corpus_robust.pickle", help="Output pickle filename.")
#     parser.add_argument("--eot", type=int, default=5, help="Batch size for EoT (robustness). Higher = more robust but slower.")
#     parser.add_argument("--threshold", type=float, default=0.15, help="Loss threshold to accept a patch.")
    
#     args = parser.parse_args()
    
#     # Create directory if it doesn't exist
#     output_dir = os.path.dirname(args.output)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
        
#     mine_robust_patches(
#         n_samples=args.samples,
#         save_path=args.output,
#         eot_samples=args.eot,
#         loss_threshold=args.threshold
#     )










# miner.py
# import torch
# import numpy as np
# import pickle
# import os
# from tqdm import tqdm
# from util import load_dataset, load_model
# from camera import Camera
# from attack_minimal_single import project_patch, normalize_yaw_t, construct_T_matrix

# from matplotlib import pyplot as plt

# # ... (Insert ModelWrapper and PController classes from previous refactor if needed) ...
# # Or just load Frontnet directly:
# def get_frontnet(device):
#     model = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
#     model.eval()
#     for param in model.parameters():
#         param.requires_grad = False
#     return model

# def mine_data(n_batches=150, batch_size=32, save_path="gold_corpus.pickle"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     frontnet = get_frontnet(device)
    
#     # Dataset for backgrounds: use batch_size=128
#     bg_dataset = load_dataset("pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", 
#                               batch_size=batch_size, shuffle=True, train=True, IMRC=True)
    
#     gold_data = []
    
#     print(f"Mining from {n_batches} random batches (batch_size={batch_size})...")
    
#     # Iterate over DataLoader batches and stop after n_batches
#     for trial in range(5000): # Max 200 steps to find a solution
#         # For each image in the batch, run the existing per-image optimization
#         # for img_idx in range(bg_imgs.shape[0]):
#         # 1. Random Conditioning
#         sf = np.random.uniform(0.5, 1.1)
#         tx = np.random.uniform(-20., 140.)
#         ty = np.random.uniform(-10., 80.)
        
#         # Target Pose
#         x_t = np.random.uniform(0., 1.5)
#         y_t = np.random.uniform(-1, 1)
#         z_t = np.random.uniform(-0.5, 0.5)
#         yaw_t = np.random.uniform(-0.1, 0.1)
        
#         target_vec = torch.tensor([x_t, y_t, z_t, yaw_t], device=device)
        
#         # 2. Optimization Loop (Standard Pixel Optimization)
#         # Initialize random patch
#         patch = torch.rand((1, 1, 45, 80), device=device, requires_grad=True)
#         opt = torch.optim.Adam([patch], lr=1e-3)
#         scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=1000)
        
#         # T Matrix
#         T = construct_T_matrix(torch.tensor(sf), torch.tensor(tx), torch.tensor(ty)).unsqueeze(0).to(device)

#         print("Patch position: ", sf, tx, ty)
#         print("Target vector: ", x_t, y_t, z_t, yaw_t)
#         print("T_matrix: ", T)
        
#         patch_batch = patch.repeat(batch_size, 1, 1, 1)  # shape: (batch_size, 1, 45, 80)
#         T_batch = T.repeat(batch_size, 1, 1)  # shape: (batch_size, 4, 4)
#         target_vec_batch = target_vec.repeat(batch_size, 1)  # shape: (batch_size, 4)

                    
#         success = False
#         best_loss = torch.inf
#         best_patch_state = None

#         # ... inside mine_data loop ...
        
#         # # 1. Define the Goal
#         # # e.g., We want a patch that makes the drone think it's at (1.0, 0, 0, 0)
#         # # regardless of slight shifts in the patch position or background.
#         # target_vec = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device) # Fixed target for this sample
        
#         # # 2. Define Universal Constraints
#         # # We define a "center" position (sf, tx, ty) but we will jitter it
#         # base_sf = np.random.uniform(0.5, 0.9)
#         # base_tx = np.random.uniform(20., 100.)
#         # base_ty = np.random.uniform(10., 60.)
        
#         # Initialize one patch to rule them all
#         patch = torch.rand((1, 1, 45, 80), device=device, requires_grad=True)
#         opt = torch.optim.Adam([patch], lr=0.05)
        
#         # 3. Robust Optimization Loop
#         for step in range(300): # More steps needed for universal patches
#             opt.zero_grad()
            
#             loss_accum = 0
            
#             # Mini-batch of robustness (Expectation over Transformation - EoT)
#             # We test the SAME patch against 5 random variations at once
#             for _ in range(5):
#                 # A. Random Background
#                 bg_img = bg_dataset.dataset[np.random.randint(len(bg_dataset))][0].to(device).unsqueeze(0) / 255.0
                
#                 # B. Random Jitter (Position Noise)
#                 # The patch must work even if it shifts by +/- 5 pixels or scales by +/- 5%
#                 jit_sf = base_sf + np.random.uniform(-0.05, 0.05)
#                 jit_tx = base_tx + np.random.uniform(-5, 5)
#                 jit_ty = base_ty + np.random.uniform(-5, 5)
                
#                 T = construct_T_matrix(torch.tensor(jit_sf), torch.tensor(jit_tx), torch.tensor(jit_ty)).unsqueeze(0).to(device)
                
#                 # C. Project & Predict
#                 manipulated = project_patch(patch, T, bg_img).clamp(0, 1)
#                 x, y, z, yaw = frontnet(manipulated * 255.)
#                 pred = torch.stack([x, y, z, yaw]).squeeze(2).mT[0]
                
#                 # D. Accumulate Loss
#                 loss_accum += torch.dist(pred[:3], target_vec[:3]) + (1 - torch.cos(normalize_yaw_t(pred[3]) - normalize_yaw_t(target_vec[3])))
            
#             # Average loss over the mini-batch
#             loss_final = loss_accum / 5.0
            
#             if loss_final < 0.15: # Slightly higher threshold for universal
#                 success = True
#                 best_patch_state = patch.detach().cpu().numpy()
#                 break
                
#             loss_final.backward()
#             patch.grad = patch.grad.sign()
#             opt.step()
#             patch.data.clamp_(0, 1)

#         # 4. Save
#         # Note: We save the BASE position as the condition.
#         # The Diffusion model learns: "At this base position, generate a patch that is ROBUST to jitter."
#         if success:
#              gold_data.append((best_patch_state.squeeze(), target_vec.cpu().numpy(), np.array([base_sf, base_tx, base_ty])))
#         # for epoch in range(2000):
#         #     for batch_idx, batch in enumerate(bg_dataset):
            
#         #         # batch expected to be a tuple like (images, labels, ...)
#         #         # take first element as images and normalize
#         #         bg_imgs = batch[0].to(device) / 255.0  # shape: (batch_size, C, H, W)
                    
#         #         for step in range(1): # Max 200 steps to find a solution
#         #             opt.zero_grad()
                    
#         #             # Project
#         #             manipulated = project_patch(patch_batch, T_batch, bg_imgs).clamp(0, 1)
#         #             # # save all manipulated images for debugging in grid
#         #             # fig, axs = plt.subplots(4, 8, figsize=(16, 8))
#         #             # for i in range(batch_size):
#         #             #     ax = axs[i // 8, i % 8]
#         #             #     ax.imshow(manipulated[i].detach().cpu().squeeze(), cmap='gray')
#         #             #     ax.axis('off')
#         #             # plt.tight_layout()
#         #             # plt.savefig(f"debug_manipulated_epoch{epoch}_batch{batch_idx}.png")
#         #             # plt.close()
                    
#         #             # Predict
#         #             x, y, z, yaw = frontnet(manipulated * 255.)
#         #             pred = torch.stack([x, y, z, yaw]).squeeze(2).mT[0]
                    
#         #             # Loss
#         #             losses = torch.dist(pred[:3], target_vec_batch[:, :3]) + (1 - torch.cos(normalize_yaw_t(pred[3]) - normalize_yaw_t(target_vec_batch[:, 3])))
#         #             sorted_losses = torch.sort(losses, descending=True)[0]
#         #             top5_losses = sorted_losses[:5]
#         #             loss = torch.mean(top5_losses)


#         #             if losses.clone().mean().detach().cpu().item() < 0.1: # Threshold for "Gold Standard"
#         #                 success = True
#         #                 best_patch_state = patch.clone().detach().cpu().numpy()
#         #                 best_patch_state = np.clip(best_patch_state, 0, 1)
#         #                 break
                    
#         #             loss.backward()
#         #             patch.grad = patch.grad.sign()
#         #             opt.step()
#         #             scheduler.step()
#         #             patch.data.clamp_(0, 1)
                
#         #         # 3. Save if successful
#         #         if success:
#         #             print(f"Epoch {epoch}, Found gold sample with loss {loss.item():.4f}")
#         #             pos_vec = np.array([sf, tx, ty])
#         #             target_vec_np = np.array([x_t, y_t, z_t, yaw_t])
                    
#         #             gold_data.append((best_patch_state.squeeze(), target_vec_np, pos_vec))

#         #     if epoch % 10 == 0:    
#         #         print(f"Epoch {epoch},: Completed optimization, final loss {loss.item():.4f}, success: {success}")

#         print(f"Mining complete. Found {len(gold_data)} usable samples across {min(n_batches, batch_idx+1)} batches.")

#     with open(save_path, "wb") as f:
#         pickle.dump(gold_data, f)

# if __name__ == "__main__":
#     mine_data()