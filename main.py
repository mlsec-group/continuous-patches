import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

# Local Imports
from simulation import DroneSimulation
from attacks import Attacker, project_patch, get_pose_from_prediction
from util import load_dataset, load_model, gen_target_trajectory 
from yolo_bounding import YOLOBox 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

class ModelWrapper:
    def __init__(self, name, device):
        self.name = name
        self.device = device
        if name == 'frontnet':
            self.model = load_model(f"{project_root}/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
            self.model.eval()
        elif name == 'yolov5':
            self.model = YOLOBox()
            
    def predict(self, image):
        if self.name == 'frontnet':
            x, y, z, yaw = self.model(image * 255.)
            return torch.stack([x, y, z, yaw]).squeeze(2).mT
        elif self.name == 'yolov5':
            #img_resize = F.interpolate(image, size=(320, 640), mode='bilinear', align_corners=False)
            img_resize = image.repeat_interleave(3, dim=1)
            return self.model(img_resize, target_anchor=None) 

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup
    sim = DroneSimulation(device, display_size_inch=args.display_size)
    model = ModelWrapper(args.model, device)
    attacker = Attacker(args, model, device, sim.cam) 
    
    dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size=1, shuffle=False, train=True, IMRC=True)
    target_traj = gen_target_trajectory(args.trajectory).to(device)

    sim.pose = target_traj[0].clone().detach()
    # print(f"Starting pose: {sim.pose.cpu().numpy()}")
    
    if args.pic_mode == 'idx':
        if args.temperature != 'none':
            output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.temperature}/{args.trajectory}/{args.display_size}/image_{args.img_idx}/{args.seed}')
        else:
            output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.trajectory}/{args.display_size}/image_{args.img_idx}/{args.seed}')
    else: # random
        if args.temperature != 'none':
            output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.temperature}/{args.trajectory}/{args.display_size}/random/{args.seed}')
        else:
            output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.trajectory}/{args.display_size}/random/{args.seed}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history_pose = []
    time_per_step = []
    history_vel_cmd = []
    target_idx = 1
    dt = 1.0 / args.timeout if args.timeout else 1/10.0
    print(f"Using dt={dt:.4f}s based on timeout={args.timeout}")

    patch_size = (150, 320) if args.model == 'yolov5' else (45, 80)
    img_size = (320, 640) if args.model == 'yolov5' else (96, 160)

    optim_step = 0
    max_retries = 10
    retry = 0
    # 2. Main Loop
    with tqdm(total=len(target_traj)) as pbar:
        while target_idx < len(target_traj) - 1:
            step_start = time.time()
            target_pose = target_traj[target_idx]
            retry += 1
            if retry > max_retries:
                # print("Max retries reached, moving to next target.\n")
                target_idx += 1
                pbar.update(1)
                retry = 0
                continue
            
            # A. Check Target Status
            dist = torch.dist(sim.pose[:3], target_pose[:3])
            # print(f"Step {optim_step}: Target idx {target_idx}, Distance to target: {dist:.4f}m")
            # print("Current Pose: ", sim.pose.cpu().numpy())
            # print("Target Pose: ", target_pose.cpu().numpy())
            if dist < 0.05:
                target_idx += 1 
                pbar.update(1)
                sim.update_physics(vel_cmd, dt=dt)
                continue
            if dist > 1.5 and args.trajectory not in ['slingshot_left', 'slingshot_right', 'slingshot_forward']:
                target_idx += 1
                pbar.update(1)
                sim.update_physics(vel_cmd, dt=dt)
                continue

            if sim.pose[0] < -2. or sim.pose[0] > 2. or sim.pose[1] < -2. or sim.pose[1] > 2. or sim.pose[2] < 0.1 or sim.pose[2] > 2.:
                # Out of bounds
                print("Drone crashed")
                break

                
            # B. Get Background Image
            img_idx = np.random.randint(len(dataset)) if args.pic_mode == 'random' else args.img_idx
            base_img = dataset.dataset[img_idx][0].to(device).unsqueeze(0) / 255.0
            
            # C. Check Geometry
            T, monitor_corners, is_visible = sim.get_view_geometry(patch_size, img_size)
            
            # D. Generate & Apply Attack
            patch = None
            manipulated_img = base_img
            
            # 'none' mode: skip all generation logic, just use base image
            if args.patch_mode == 'none' or args.patch_mode == 'optimal':
                pass
            # standard modes: check visibility first
            elif not is_visible:
                # Monitor not visible, cannot attack -> fly on clean image
                pass 
            else:
                # Generate Patch (Optimization or Diffusion happens inside here)
                patch = attacker.generate(base_img, T, sim.pose, target_pose)
                if args.model == 'yolov5':
                    base_img = F.interpolate(base_img, size=(320, 640), mode='bilinear', align_corners=False)
                    manipulated_img = base_img.clone()
                if patch is not None:
                    manipulated_img = project_patch(patch, T.unsqueeze(0), base_img).clamp(0, 1)

            # E. Perception (What does the drone see?)
            # Use the SHARED logic from attacks.py to get consistent World Pose
            if args.patch_mode == 'optimal':
                # Simulating attacker 'predicts' the optimal == target pose
                perceived_pose = target_pose
            else:
                perceived_pose = get_pose_from_prediction(model, manipulated_img, sim.pose, sim.cam)
            
            # F. Control (Fly based on Perceived Pose vs Target Pose)
            # Construct state vector for controller (XYZ + Yaw)
            current_state_for_control = torch.cat([sim.pose[:3], perceived_pose[3].unsqueeze(0)])
            
            # Step Controller
            _, vel_cmd = sim.controller.step(current_state_for_control, perceived_pose, dt=dt)
            
            # Update Real Physics
            sim.update_physics(vel_cmd, dt=dt)
            
            # G. Log
            step_end = time.time()
            current_pose = sim.pose.cpu().numpy()
            history_pose.append(current_pose)
            time_per_step.append(step_end - step_start)
            history_vel_cmd.append(vel_cmd.detach().clone().cpu().numpy())
            optim_step += 1

            # Plot Intermediate Results
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(manipulated_img[0, 0].detach().cpu().numpy(), cmap='gray')            
            # monitor_corners are in format (ul_x, ul_y), (ur_x, ur_y), (ll_x, ll_y), (lr_x, lr_y)
            try:
                monitor_corners = monitor_corners.detach().cpu().numpy()
                axs[0].plot([monitor_corners[0, 0], monitor_corners[1, 0]], [monitor_corners[0, 1], monitor_corners[1, 1]], 'r--')  # top edge
                axs[0].plot([monitor_corners[0, 0], monitor_corners[2, 0]], [monitor_corners[0, 1], monitor_corners[2, 1]], 'r--')  # left edge
                axs[0].plot([monitor_corners[1, 0], monitor_corners[3, 0]], [monitor_corners[1, 1], monitor_corners[3, 1]], 'r--')  # right edge
                axs[0].plot([monitor_corners[2, 0], monitor_corners[3, 0]], [monitor_corners[2, 1], monitor_corners[3, 1]], 'r--')  # bottom edge    
            except:
                pass
            axs[1].plot(target_traj[:, 0].detach().cpu().numpy(), target_traj[:, 1].detach().cpu().numpy(), 'r--')
            axs[1].plot(np.array(history_pose)[:, 0], np.array(history_pose)[:, 1])
            axs[1].scatter(sim.monitor_world[:, 0].detach().cpu().numpy(), sim.monitor_world[:, 1].detach().cpu().numpy(), c='b', label='Projector corners')
            axs[1].scatter(current_pose[0], current_pose[1], color='black')
            axs[1].arrow(current_pose[0], current_pose[1],
                            0.3 * np.cos(current_pose[3]), 0.3 * np.sin(current_pose[3]),
                            head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            
            axs[1].set_xlim(-2, 2)
            axs[1].set_ylim(-2, 2)
            plt.tight_layout()

            plt.savefig(output_dir / f'optim_step{optim_step}.png')
            plt.close()
            
            if len(history_pose) % 50 == 0:
                np.save(output_dir / "all_drone_poses.npy", np.array(history_pose))
                np.save(output_dir / "time_per_step.npy", np.array(time_per_step))
                np.save(output_dir / "all_velocity_commands.npy", np.array(history_vel_cmd))

    np.save(output_dir / "all_drone_poses.npy", np.array(history_pose))
    np.save(output_dir / "time_per_step.npy", np.array(time_per_step))
    np.save(output_dir / "all_velocity_commands.npy", np.array(history_vel_cmd))
    print(f"Done. Mean time per step: {np.mean(time_per_step):.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet', help='Model to use for prediction')
    parser.add_argument('-t', '--trajectory', type=str, choices=['figure8', 'square', 'circle', 'line_x', 'line_y', 'diagonal_line', 'triangle', 'c', 's', 'u', 'slingshot_left', 'slingshot_right', 'slingshot_forward'], default='figure8', help='Target Trajectory')
    parser.add_argument('--display_size', type=int, default=60, help='Size of the display in pixels (default: 60")')
    parser.add_argument('--patch_mode', type=str, choices=['none', 'optimal', 'velo', 'timeout', 'black', 'white', 'random', 'fap', 'diffusion', 'interpolation', 'corpus'], default='none', help='Mode to initialize the patch: optimal, timeout, black, white, random')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--pic_mode', type=str, choices=['random', 'idx'], default='idx', help='Mode to select image: random or specific index')
    parser.add_argument('--img_idx', type=int, default=0, help='Index of the image to use from the dataset')
    parser.add_argument('--temperature', type=str, choices=['warm', 'cold', 'none'], default='none', help='Either restart from random patch (cold) or from the last patch (warm)')
    parser.add_argument('--corpus_size', type=int, choices=[1000, 2000, 3000], default=1000, help='Number of patches in the corpus (only for corpus/interpolation/diffusion patch mode)')
    parser.add_argument('--timeout', type=int, default=None)
    
    args = parser.parse_args()
    run_experiment(args)