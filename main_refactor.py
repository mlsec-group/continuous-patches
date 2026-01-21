import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Local Imports
from simulation import DroneSimulation
from attacks import Attacker, get_pose_from_prediction
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
            return self.model(img_resize, target_anchor=None) # No ghost anchor needed

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup
    sim = DroneSimulation(device, display_size_inch=args.display_size)
    model = ModelWrapper(args.model, device)
    # Pass cam params (sim.cam) to attacker for the shared pose logic
    attacker = Attacker(args, model, device, sim.cam) 
    
    dataset = load_dataset(f"{project_root}/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size=1, shuffle=False, train=True, IMRC=True)
    target_traj = gen_target_trajectory(args.trajectory).to(device)
    
    if args.pic_mode == 'image':
        output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.trajectory}/{args.display_size}/image_{args.img_idx}/{args.seed}')
    if args.pic_mode == 'random':
        output_dir = Path(f'{project_root}/paper_results/{args.model}/{args.patch_mode}/{args.trajectory}/{args.display_size}/random/{args.seed}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history_pose = []
    target_idx = 1

    dt = 1 / args.timeout  # Simulation timestep

    time_per_steps = []
    
    # 2. Main Loop
    with tqdm(total=len(target_traj)) as pbar:
        while target_idx < len(target_traj) - 1:
            start_time = time.time()
            target_pose = target_traj[target_idx]
            
            # A. Check Target Status
            dist = torch.dist(sim.pose[:3], target_pose[:3])
            if dist < 0.05:
                # print("Target reached.")
                target_idx += 1; pbar.update(1); continue
            if dist > 1.5:
                # print("Target too far, skipping.")
                target_idx += 1; pbar.update(1); continue
                
            # B. Get Image & Geometry
            img_idx = np.random.randint(len(dataset)) if args.pic_mode == 'random' else args.img_idx
            base_img = dataset.dataset[img_idx][0].to(device).unsqueeze(0) / 255.0
            
            patch_size = (150, 320) if args.model == 'yolov5' else (45, 80)
            img_size = (320, 640) if args.model == 'yolov5' else (96, 160)
            
            T, _, is_visible = sim.get_view_geometry(patch_size, img_size)
            
            if not is_visible:
                # Monitor not visible, just fly normally towards target
                # (Use un-attacked image logic or skip?)
                # Monolith skipped. Let's skip to keep behavior.
                target_idx += 1
                continue

            # C. Generate Attack
            # Returns manipulated image and patch
            # Note: The optimization loop is now fully inside generate()
            attacked_img, patch = attacker.generate(base_img, T, sim.pose, target_pose)
            
            # D. Perception (What does the drone see?)
            # Uses the exact same logic as the optimizer to calculate world pose
            perceived_pose = get_pose_from_prediction(model, attacked_img, sim.pose, sim.cam)
            
            # E. Control (Fly based on Perception)
            # The drone thinks it is at 'perceived_pose'. It wants to go to 'target_pose'.
            # wait, monolith logic: controller.step(current=sim.pose, target=prediction)
            # Frontnet predicts the RELATIVE target. 
            # If perceived_pose is the "Target World Pose predicted by Frontnet", 
            # then we want to fly to perceived_pose.
            
            # Let's align with monolith: "best_setpoint = prediction"
            # "prediction = controller.step(current, target=prediction)"
            
            # Logic:
            # 1. Neural Net predicts where I should be (Setpoint).
            # 2. I am at sim.pose.
            # 3. Error = Setpoint - Me.
            
            # Current State for controller needs yaw too
            current_state_full = torch.cat([sim.pose[:3], perceived_pose[3].unsqueeze(0)])
            
            # Step controller
            _, vel_cmd = sim.controller.step(current_state_full, perceived_pose, dt=dt)
            
            # Apply to Physics
            sim.update_physics(vel_cmd, dt=dt)
            
            # F. Log
            history_pose.append(sim.pose.cpu().numpy())
            
            # Save periodic debug
            if len(history_pose) % 50 == 0:
                np.save(output_dir / "all_drone_poses.npy", np.array(history_pose))

            end_time = time.time()
            print(f"Total Simulation Time: {end_time - start_time:.2f} seconds")
            time_per_steps.append((end_time - start_time))

    np.save(output_dir / "all_drone_poses.npy", np.array(history_pose))
    # print(f"Average Time per Step: {np.mean(time_per_steps):.2f} seconds")
    np.save(output_dir / "time_per_step.npy", np.array(time_per_steps))
    print("Experiment Complete.")

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