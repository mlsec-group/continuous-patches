import numpy as np
import os
from tqdm import tqdm
# from trajectory_generation import gen_target_trajectory


def check_all_drone_poses_sizes(models, modes, monitor_sizes,
                                trajectories, images, num_seeds=10):
    """
    Walks model/mode/trajectory/sizez/{random|image_X}/seed and checks if
    all_drone_poses.npy has shape (len(target_trajectory), 4).
    Prints: {model}/{mode}/{display_size}/{picture}/{seed} for mismatches.
    Writes all mismatched paths to a file named 'mismatched_paths.txt'.
    """
    mismatched_paths = []

    for model in models:
        print(f'Checking model: {model}')
        for mode in tqdm(modes):
            for trajectory in trajectories:
                # expected_len = len(gen_target_trajectory(trajectory))
                for ms in monitor_sizes:
                    for img in images:
                        picture = 'random' if img == 'random' else f'image_{img}'
                        for seed in range(num_seeds):
                            seed_dir = f'{model}/{mode}/{trajectory}/{ms}z/{picture}/{seed}'
                            poses_fp = f'{seed_dir}/all_drone_poses.npy'
                            if not os.path.exists(poses_fp):
                                # Missing file is considered incorrect size
                                mismatched_paths.append(f'{model}/{mode}/{ms}/{picture}/{seed}')
                                continue
                            try:
                                arr = np.load(poses_fp, allow_pickle=False)
                            except Exception:
                                mismatched_paths.append(f'{model}/{mode}/{ms}/{picture}/{seed}')
                                continue
                            if arr.ndim != 2 or arr.shape[1] != trajectory_shape[1] or arr.shape[0] != trajectory_shape[0]:
                                mismatched_paths.append(f'{model}/{mode}/{ms}/{picture}/{seed}')
    
    # Write all mismatched paths to a file
    with open('reruns_needed.txt', 'w') as f:
        for path in mismatched_paths:
            f.write(path + '\n')


if __name__ == '__main__':
    trajectory_shape = (20,4)
    models = ['frontnet', 'yolov5']
    modes = ['optimal/cold', 'optimal/warm', 'timeout_10Hz/cold', 
         'timeout_10Hz/warm', 'timeout_20Hz/cold', 'timeout_20Hz/warm', 
         'timeout_30Hz/cold', 'timeout_30Hz/warm', 'white', 'black', 'random', 'fap']
    monitor_sizes = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    img_idx = [505, 4847, 3059, 1860, 3205, 4861, 2613, 2309, 5431, 2847, 'random']
    trajectories = ["figure8", "square", "circle", "line_y", "line_x"]

    check_all_drone_poses_sizes(models=models, modes=modes, monitor_sizes=monitor_sizes,
                                trajectories=trajectories, images=img_idx, num_seeds=10)
    