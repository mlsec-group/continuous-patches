import numpy as np
import torch
import pathlib
import os, sys
import pandas as pd
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Project root: ", project_root)

from util import gen_target_trajectory 


def compute_dtw_distance(target_traj: torch.tensor, actual_traj: torch.tensor):
    """
    Computes the Dynamic Time Warping (DTW) distance between two 3D trajectories.
    
    Args:
        target_traj: np.array or tensor of shape (N, 3)
        actual_traj: np.array or tensor of shape (M, 3)
        
    Returns:
        float: The normalized DTW distance (average distance per point alignment)
    """
    # # Ensure inputs are numpy arrays
    # if hasattr(target_traj, 'cpu'): target_traj = target_traj.cpu().numpy()
    # if hasattr(actual_traj, 'cpu'): actual_traj = actual_traj.cpu().numpy()

    # 1. Compute the pairwise distance matrix (Euclidean)
    # Shape: (N, M)
    dists = torch.cdist(target_traj, actual_traj, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    # 2. Initialize the cumulative cost matrix
    N, M = dists.shape
    dtw_matrix = torch.zeros((N, M), device=dists.device, dtype=dists.dtype)
    dtw_matrix[0, 0] = dists[0, 0]

    # 3. Fill first row and first column
    for i in range(1, N):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + dists[i, 0]
    for j in range(1, M):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + dists[0, j]

    # 4. Fill the rest using the recurrence relation
    # Cost = current_dist + min(left, up, diagonal)
    for i in range(1, N):
        for j in range(1, M):
            min_prev = torch.min(torch.stack([
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            ]))
            dtw_matrix[i, j] = dists[i, j] + min_prev

    # 5. Return normalized distance (optional but recommended for comparison)
    # The value at dtw_matrix[-1, -1] is the total accumulated cost.
    # Normalizing by path length (N+M) makes it interpretable as "avg error in meters"
    return dtw_matrix[-1, -1] / (N + M)

# frechet distance between two 3D trajectories
def compute_frechet_distance(target_traj: torch.tensor, actual_traj: torch.tensor):
    """
    Computes the Discrete Fréchet distance between two 3D trajectories.
    Args:
        target_traj: np.array or tensor of shape (N, 3)
        actual_traj: np.array or tensor of shape (M, 3)
    Returns:
        float: The Fréchet distance
    """
    N = target_traj.shape[0]
    M = actual_traj.shape[0]

    # Create a 2D array to store the distances
    ca = -torch.ones((N, M), device=target_traj.device, dtype=target_traj.dtype)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = torch.norm(target_traj[0] - actual_traj[0])
        elif i > 0 and j == 0:
            ca[i, j] = torch.max(c(i-1, 0), torch.norm(target_traj[i] - actual_traj[0]))
        elif i == 0 and j > 0:
            ca[i, j] = torch.max(c(0, j-1), torch.norm(target_traj[0] - actual_traj[j]))
        elif i > 0 and j > 0:
            ca[i, j] = torch.max(
                torch.min(torch.stack([c(i-1, j), c(i-1, j-1), c(i, j-1)])),
                torch.norm(target_traj[i] - actual_traj[j])
            )
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return c(N-1, M-1)

# frontnet_results_fp = pathlib.Path(project_root) / "paper_results" / "frontnet"


# black_figure8_40_image_1860_0 = np.load(frontnet_results_fp / "velo/figure8/40/image_1860/0/all_drone_poses.npy")
# black_figure8_40_image_1860_0 = torch.tensor(black_figure8_40_image_1860_0, dtype=torch.float32)

# print("Len poses: ", black_figure8_40_image_1860_0.shape)

# # len_target_trajectory = max(len(black_figure8_40_image_1860_0)-5, 25)
# target_trajectory = gen_target_trajectory("figure8")

# print("Target trajectory shape: ", target_trajectory.shape)


# # compute DTW between target_trajectory and black_figure8_40_image_1860_0
# # dtw_score = compute_dtw_distance(target_trajectory[:, :3], black_figure8_40_image_1860_0[:, :3])
# # print(f"DTW score (black, figure8, 40, image_1860, seed 0): {dtw_score.item():.4f} meters")

# frechet_score = compute_frechet_distance(target_trajectory[:, :3], black_figure8_40_image_1860_0[:, :3])
# print(f"Fréchet score (black, figure8, 40, image_1860, seed 0): {frechet_score.item():.4f} meters")

base_path = f"{project_root}/paper_results"
print("Base path: ", base_path)

MODELS = ("frontnet", )
PATCH_MODES = ("random", "black", "timeout", "velo", "diffusion", "optimal", "corpus", "none")
TEMPERATURES = ("warm", "cold")
TRAJECTORIES = ("figure8", "triangle", "u", "s", "slingshot_left")
DISPLAY_SIZES = (40, 50, 60, 70, 80, 90, 100, 110, 120)
SEEDS = (0, 1, 2)
PIC_MODES = ("image", "random")
IMAGE_IDX = (1860, 4861, 5431)
# TIMEOUT_VALUES = (10, 20, 30)

found_files = []
missing_combinations = []

for model in MODELS:
    for patch_mode in PATCH_MODES:
        for temperature in TEMPERATURES:
            for trajectory in TRAJECTORIES:
                for display_size in DISPLAY_SIZES:
                    for pic_mode in PIC_MODES:
                        if pic_mode == "image":
                            for image_index in IMAGE_IDX:
                                for seed in SEEDS:
                                    file_path = os.path.join(base_path, model, f"{patch_mode}", temperature, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", str(seed), "all_drone_poses.npy")
                                    if os.path.isfile(file_path):
                                        found_files.append((model, f"{patch_mode}", temperature, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", seed))
                                    else:
                                        missing_combinations.append((model, f"{patch_mode}", temperature, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", seed))
                        else:  # random pic_mode
                            for seed in SEEDS:
                                file_path = os.path.join(base_path, model, f"{patch_mode}", temperature, trajectory, f"{display_size}", pic_mode, str(seed), "all_drone_poses.npy")
                                if os.path.isfile(file_path):
                                    found_files.append((model, f"{patch_mode}", temperature, trajectory, f"{display_size}", pic_mode, seed))
                                else:
                                    missing_combinations.append((model, f"{patch_mode}", temperature, trajectory, f"{display_size}", pic_mode, seed))
        if patch_mode == "velo":
            for temperature in TEMPERATURES:
                for trajectory in TRAJECTORIES:
                    for display_size in DISPLAY_SIZES:
                        for pic_mode in PIC_MODES:
                            if pic_mode == "image":
                                for image_index in IMAGE_IDX:
                                    for seed in SEEDS:
                                        file_path = os.path.join(base_path, model, patch_mode, temperature, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", str(seed), "all_drone_poses.npy")
                                        if os.path.isfile(file_path):
                                            found_files.append((model, patch_mode, temperature, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", seed))
                                        else:
                                            missing_combinations.append((model, patch_mode, temperature, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", seed))
                            else:  # random pic_mode
                                for seed in SEEDS:
                                    file_path = os.path.join(base_path, model, patch_mode, temperature, trajectory, f"{display_size}", pic_mode, str(seed), "all_drone_poses.npy")
                                    if os.path.isfile(file_path):
                                        found_files.append((model, patch_mode, temperature, trajectory, f"{display_size}", pic_mode, seed))
                                    else:
                                        missing_combinations.append((model, patch_mode, temperature, trajectory, f"{display_size}", pic_mode, seed))
        else:  # random and black patch modes
            for trajectory in TRAJECTORIES:
                for display_size in DISPLAY_SIZES:
                    for pic_mode in PIC_MODES:
                        if pic_mode == "image":
                            for image_index in IMAGE_IDX:
                                for seed in SEEDS:
                                    file_path = os.path.join(base_path, model, patch_mode, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", str(seed), "all_drone_poses.npy")
                                    if os.path.isfile(file_path):
                                        found_files.append((model, patch_mode, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", seed))
                                    else:
                                        missing_combinations.append((model, patch_mode, trajectory, f"{display_size}", f"{pic_mode}_{image_index}", seed))
                        else:  # random pic_mode
                            for seed in SEEDS:
                                file_path = os.path.join(base_path, model, patch_mode, trajectory, f"{display_size}", pic_mode, str(seed), "all_drone_poses.npy")
                                if os.path.isfile(file_path):
                                    found_files.append((model, patch_mode, trajectory, f"{display_size}", pic_mode, seed))
                                else:
                                    missing_combinations.append((model, patch_mode, trajectory, f"{display_size}", pic_mode, seed))



print(f"Found {len(found_files)} existing result files.")
print(f"Missing {len(missing_combinations)} result files.")




# print("found_files examples: ", found_files[:5])

# print("missing_combinations examples: ", missing_combinations[:5])


target_trajectories = {"figure8": gen_target_trajectory("figure8"),
                       "triangle": gen_target_trajectory("triangle"),
                       "u": gen_target_trajectory("u"),
                       "s": gen_target_trajectory("s"),
                       "slingshot_left": gen_target_trajectory("slingshot_left")}


# check if .csv and .pkl already exist
# if so, load rows from there to avoid recomputation
if os.path.isfile(os.path.join(base_path, "all_results_frontnet.csv")) and os.path.isfile(os.path.join(base_path, "all_results_frontnet.pkl")):
    print("Loading existing results from CSV and PKL...")
    df_existing = pd.read_csv(os.path.join(base_path, "all_results_frontnet.csv"))
    rows = df_existing.to_dict('records')
    print(f"Loaded {len(rows)} existing result rows.")

    set_existing_files = set()
    for row in rows:
        if row["temperature"] != "":
            set_existing_files.add((row["model"], row["patch_mode"], row["temperature"], row["trajectory"], str(row["display_size"]), f"{row['pic_mode']}_{row['image_index']}" if row['pic_mode']=="image" else row['pic_mode'], row["seed"]))
        else:
            set_existing_files.add((row["model"], row["patch_mode"], row["trajectory"], str(row["display_size"]), f"{row['pic_mode']}_{row['image_index']}" if row['pic_mode']=="image" else row['pic_mode'], row["seed"]))
    # Filter found_files to only those not already in existing results
    found_files = [f for f in found_files if f not in set_existing_files]
    print(f"{len(found_files)} files remain to be processed after filtering existing results.")

else:
    rows = []

for file_info in tqdm(found_files):
    # Normalize tuple shapes: with temperature -> len 7, without -> len 6
    if len(file_info) == 7:
        model, patch_mode, temperature, trajectory, display_size, pic_mode_image, seed = file_info
        temp_in_path = True
    else:
        model, patch_mode, trajectory, display_size, pic_mode_image, seed = file_info
        temperature = ""
        temp_in_path = False



    # Determine pic_mode and image_index
    if isinstance(pic_mode_image, str) and pic_mode_image.startswith("image_"):
        pic_mode = "image"
        try:
            image_index = int(pic_mode_image.split("_", 1)[1])
        except Exception:
            image_index = None
    else:
        pic_mode = pic_mode_image if pic_mode_image in ("random", "image") else "random"
        image_index = None

    # Build file_path depending on whether temperature is part of the path
    if temp_in_path:
        file_path = os.path.join(base_path, model, patch_mode, temperature, trajectory, str(display_size), pic_mode_image, str(seed), "all_drone_poses.npy")
    else:
        file_path = os.path.join(base_path, model, patch_mode, trajectory, str(display_size), pic_mode_image, str(seed), "all_drone_poses.npy")

    if not os.path.isfile(file_path):
        # Skip if file unexpectedly missing
        missing_combinations.append((model, patch_mode, temperature if temp_in_path else None, trajectory, display_size, pic_mode_image, seed))
        continue

    try:
        drone_poses = np.load(file_path)
    except Exception as e:
        print("Failed to load:", file_path, "error:", e)
        missing_combinations.append((model, patch_mode, temperature if temp_in_path else None, trajectory, display_size, pic_mode_image, seed))
        continue

    drone_poses = torch.tensor(drone_poses, dtype=torch.float32)
    target_traj = target_trajectories[trajectory]

    dtw_score = compute_dtw_distance(target_traj[:, :3], drone_poses[:, :3]).item()
    frechet_score = compute_frechet_distance(target_traj[:, :3], drone_poses[:, :3]).item()

    mean_time_per_step = np.mean(np.load(file_path.replace("all_drone_poses.npy", "time_per_step.npy")))

    # Save per-file score arrays (preserve existing behavior)
    np.save(file_path.replace("all_drone_poses.npy", "dtw_score.npy"), np.array(dtw_score))
    np.save(file_path.replace("all_drone_poses.npy", "frechet_score.npy"), np.array(frechet_score))

    # Append row
    rows.append({
        "model": model,
        "patch_mode": patch_mode,
        "temperature": temperature if temp_in_path else "cold",
        "trajectory": trajectory,
        "display_size": int(display_size) if str(display_size).isdigit() else display_size,
        "pic_mode": pic_mode,
        "image_index": image_index,
        "seed": int(seed),
        "file_path": file_path,
        "dtw": float(dtw_score),
        "frechet": float(frechet_score),
        "mean_time_per_step": float(mean_time_per_step)
    })

    # print("Saved scores for file: ", file_path)


 # save missing combinations to a csv for reference
missing_df = pd.DataFrame(missing_combinations, columns=["model", "patch_mode", "temperature", "trajectory", "display_size", "pic_mode_image", "seed"])
missing_df.to_csv(os.path.join(base_path, "missing_combinations_frontnet.csv"), index=False)
print(f"Wrote missing combinations to CSV: {os.path.join(base_path, 'missing_combinations_frontnet.csv')}")

# Build DataFrame and persist
df = pd.DataFrame(rows)
csv_fp = os.path.join(base_path, "all_results_frontnet.csv")
pkl_fp = os.path.join(base_path, "all_results_frontnet.pkl")
df.to_csv(csv_fp, index=False)
df.to_pickle(pkl_fp)

print(f"Wrote aggregated results: {csv_fp} ({len(df)} rows)")