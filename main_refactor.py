import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

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
            return torch.stack([x, y, z, yaw], dim=1).squeeze(2).mT
        elif self.name == 'yolov5':
            img_resize = F.interpolate(image, size=(320, 640), mode='bilinear', align_corners=False)
            img_resize = img_resize.repeat_interleave(3, dim=1)
            return self.model(img_resize, target_anchor=None) 

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup
    sim = DroneSimulation(device, display_size_inch=args.display_size)
    model = ModelWrapper(args.model, device)
    attacker = Attacker(args, model, device, sim.cam) 
    
    dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size=1, shuffle=False, train=True, IMRC=True)
    target_traj = gen_target_trajectory(args.trajectory).to(device)
    
    if args.pic_mode == 'image':
        output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.trajectory}/{args.display_size}/image_{args.img_idx}/{args.seed}')
    else: # random
        output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.trajectory}/{args.display_size}/random/{args.seed}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history_pose = []
    time_per_step = []
    target_idx = 1
    dt = 1.0 / args.timeout if args.timeout else 1/30.0

    # 2. Main Loop
    with tqdm(total=len(target_traj)) as pbar:
        while target_idx < len(target_traj) - 1:
            step_start = time.time()
            target_pose = target_traj[target_idx]
            
            # A. Check Target Status
            dist = torch.dist(sim.pose[:3], target_pose[:3])
            if dist < 0.05:
                target_idx += 1; pbar.update(1); continue
            if dist > 1.5:
                target_idx += 1; pbar.update(1); continue
                
            # B. Get Background Image
            img_idx = np.random.randint(len(dataset)) if args.pic_mode == 'random' else args.img_idx
            base_img = dataset.dataset[img_idx][0].to(device).unsqueeze(0) / 255.0
            
            # C. Check Geometry
            patch_size = (150, 320) if args.model == 'yolov5' else (45, 80)
            img_size = (320, 640) if args.model == 'yolov5' else (96, 160)
            T, _, is_visible = sim.get_view_geometry(patch_size, img_size)
            
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
            # The drone *thinks* it is at 'perceived_pose'. It wants to go to 'target_pose'.
            # Note: For 'optimal' attack, perceived_pose should be close to target_pose, so velocity -> 0.
            
            # Construct state vector for controller (XYZ + Yaw)
            current_state_for_control = torch.cat([sim.pose[:3], perceived_pose[3].unsqueeze(0)])
            
            # Step Controller
            _, vel_cmd = sim.controller.step(current_state_for_control, perceived_pose, dt=dt)
            
            # Update Real Physics
            sim.update_physics(vel_cmd, dt=dt)
            
            # G. Log
            history_pose.append(sim.pose.cpu().numpy())
            step_end = time.time()
            time_per_step.append(step_end - step_start)
            
            if len(history_pose) % 50 == 0:
                np.save(output_dir / "all_drone_poses.npy", np.array(history_pose))

    np.save(output_dir / "all_drone_poses.npy", np.array(history_pose))
    np.save(output_dir / "time_per_step.npy", np.array(time_per_step))
    print(f"Done. Mean time per step: {np.mean(time_per_step):.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='frontnet')
    parser.add_argument('-t', '--trajectory', default='figure8')
    parser.add_argument('--display_size', type=int, default=60)
    parser.add_argument('--patch_mode', default='optimal')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pic_mode', default='random')
    parser.add_argument('--img_idx', type=int, default=0)
    parser.add_argument('--temperature', default='warm')
    parser.add_argument('--corpus_size', type=int, default=1000)
    parser.add_argument('--timeout', type=int, default=30)
    
    args = parser.parse_args()
    run_experiment(args)