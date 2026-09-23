import numpy as np
import torch
import pathlib
import os, sys
import pandas as pd
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Project root: ", project_root)

from src.util import gen_target_trajectory


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

MODELS = ("frontnet", "yolov5")
PATCH_MODES = ("random", "black", "timeout", "velo", "diffusion", "optimal", "corpus", "none", "interpolation", "fap")
TEMPERATURES = ("warm", "cold")
# Only these patch modes are run with a temperature directory (warm/cold) in the path
TEMPERATURE_MODES = ("velo", "timeout")
TRAJECTORIES = ("figure8", "triangle", "u", "s", "slingshot_left", "slingshot_right", "slingshot_forward", "slingshot_backward")
DISPLAY_SIZES = (40, 50, 60, 70, 80, 90, 100, 110, 120)
SEEDS = (0, 1, 2)
PIC_MODES = ("image", "random")
IMAGE_IDX = (1860, 4861, 5431)
# TIMEOUT_VALUES = (10, 20, 30)

found_files = []
missing_combinations = []

for model in MODELS:
    for patch_mode in PATCH_MODES:
        uses_temp = patch_mode in TEMPERATURE_MODES
        for temperature in (TEMPERATURES if uses_temp else ("",)):
            for trajectory in TRAJECTORIES:
                for display_size in DISPLAY_SIZES:
                    for pic_mode in PIC_MODES:
                        if pic_mode == "image":
                            for image_index in IMAGE_IDX:
                                for seed in SEEDS:
                                    pic_dir = f"{pic_mode}_{image_index}"
                                    if uses_temp:
                                        file_path = os.path.join(base_path, model, patch_mode, temperature, trajectory, f"{display_size}", pic_dir, str(seed), "all_drone_poses.npy")
                                    else:
                                        file_path = os.path.join(base_path, model, patch_mode, trajectory, f"{display_size}", pic_dir, str(seed), "all_drone_poses.npy")
                                    combo = (model, patch_mode, temperature, trajectory, f"{display_size}", pic_dir, seed)
                                    if os.path.isfile(file_path):
                                        found_files.append(combo)
                                    else:
                                        missing_combinations.append(combo)
                        else:  # random pic_mode
                            for seed in SEEDS:
                                if uses_temp:
                                    file_path = os.path.join(base_path, model, patch_mode, temperature, trajectory, f"{display_size}", pic_mode, str(seed), "all_drone_poses.npy")
                                else:
                                    file_path = os.path.join(base_path, model, patch_mode, trajectory, f"{display_size}", pic_mode, str(seed), "all_drone_poses.npy")
                                combo = (model, patch_mode, temperature, trajectory, f"{display_size}", pic_mode, seed)
                                if os.path.isfile(file_path):
                                    found_files.append(combo)
                                else:
                                    missing_combinations.append(combo)



print(f"Found {len(found_files)} existing result files.")
print(f"Missing {len(missing_combinations)} result files.")




# print("found_files examples: ", found_files[:5])

# print("missing_combinations examples: ", missing_combinations[:5])


target_trajectories = {traj: gen_target_trajectory(traj) for traj in TRAJECTORIES}


# check if .csv and .pkl already exist for each model
# if so, load rows from there to avoid recomputation
all_rows = {}
for model in MODELS:
    csv_path = os.path.join(base_path, f"all_results_{model}.csv")
    pkl_path = os.path.join(base_path, f"all_results_{model}.pkl")
    if os.path.isfile(csv_path) and os.path.isfile(pkl_path):
        print(f"Loading existing results for {model} from CSV and PKL...")
        df_existing = pd.read_csv(csv_path)
        all_rows[model] = df_existing.to_dict('records')
        print(f"Loaded {len(all_rows[model])} existing result rows for {model}.")
    else:
        all_rows[model] = []

# Process files per model
files_per_model = {model: [] for model in MODELS}
for f in found_files:
    model = f[0]
    if model in MODELS:
        # Normalize to 7-tuple format
        if len(f) == 6:
            files_per_model[model].append((f[0], f[1], "", f[2], f[3], f[4], f[5]))
        else:
            files_per_model[model].append(f)

# Filter out already processed files per model
def _canonical_temp(patch_mode, temperature):
    # Non-temperature modes use "" as the canonical key (old CSVs may store "cold")
    if patch_mode not in TEMPERATURE_MODES:
        return ""
    return "" if temperature is None or pd.isna(temperature) else temperature

def _row_key(row):
    pic_mode = row['pic_mode']
    image_index = row.get('image_index')
    if pic_mode == "image" and image_index is not None and not pd.isna(image_index):
        pic_mode_image = f"image_{int(image_index)}"
    else:
        pic_mode_image = pic_mode
    return (row["model"], row["patch_mode"], _canonical_temp(row["patch_mode"], row["temperature"]), row["trajectory"], str(int(row["display_size"])), pic_mode_image, int(row["seed"]))

for model in MODELS:
    if all_rows[model]:
        set_existing_files = set()
        for row in all_rows[model]:
            set_existing_files.add(_row_key(row))
        files_per_model[model] = [f for f in files_per_model[model] if f not in set_existing_files]
        print(f"{len(files_per_model[model])} files remain to be processed for {model} after filtering existing results.")

rows = []

for model in MODELS:
    for file_info in tqdm(files_per_model[model], desc=f"Processing {model}"):
        # All tuples are normalized to 7-tuple format
        model_name, patch_mode, temperature, trajectory, display_size, pic_mode_image, seed = file_info
        temp_in_path = temperature != ""

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
            file_path = os.path.join(base_path, model_name, patch_mode, temperature, trajectory, str(display_size), pic_mode_image, str(seed), "all_drone_poses.npy")
        else:
            file_path = os.path.join(base_path, model_name, patch_mode, trajectory, str(display_size), pic_mode_image, str(seed), "all_drone_poses.npy")

        if not os.path.isfile(file_path):
            # Skip if file unexpectedly missing
            missing_combinations.append((model_name, patch_mode, temperature if temp_in_path else None, trajectory, display_size, pic_mode_image, seed))
            continue

        try:
            drone_poses = np.load(file_path)
        except Exception as e:
            print("Failed to load:", file_path, "error:", e)
            missing_combinations.append((model_name, patch_mode, temperature if temp_in_path else None, trajectory, display_size, pic_mode_image, seed))
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
            "model": model_name,
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
missing_df.to_csv(os.path.join(base_path, "missing_combinations.csv"), index=False)
print(f"Wrote missing combinations to CSV: {os.path.join(base_path, 'missing_combinations.csv')}")

# Build DataFrame and persist per model (merge existing rows with new rows, deduplicated)
for model in MODELS:
    new_rows = [r for r in rows if r["model"] == model]
    existing_rows = all_rows.get(model, [])

    merged = list(existing_rows)
    existing_keys = {_row_key(r) for r in existing_rows}
    for row in new_rows:
        key = _row_key(row)
        if key not in existing_keys:
            merged.append(row)
            existing_keys.add(key)

    if merged:
        df = pd.DataFrame(merged)
        csv_fp = os.path.join(base_path, f"all_results_{model}.csv")
        pkl_fp = os.path.join(base_path, f"all_results_{model}.pkl")
        df.to_csv(csv_fp, index=False)
        df.to_pickle(pkl_fp)
        print(f"Wrote aggregated results for {model}: {csv_fp} ({len(df)} rows)")
    else:
        print(f"No results to write for {model}")