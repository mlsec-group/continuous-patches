import os, sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import re


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Project root: ", project_root)

from src.util import gen_target_trajectory


parser = argparse.ArgumentParser(description='Generate evaluation tables and plots for a victim model')
parser.add_argument('-m', '--model', type=str, choices=['frontnet', 'yolov5'], default='frontnet', help='Victim model to plot')
args = parser.parse_args()

csv_fp = f'paper_results/all_results_{args.model}.csv'
if not os.path.isfile(csv_fp):
    print(f"Error: {csv_fp} not found. Run evaluation/compute_metrics.py first.")
    sys.exit(1)

df = pd.read_csv(csv_fp)
# print(df.head())

# header: model,patch_mode,temperature,trajectory,display_size,pic_mode,image_index,seed,file_path,dtw,frechet,mean_time_per_step
print(df[df['patch_mode'] == 'none'].head())

# aggregate mean frechet per (model, patch_mode, temperature, trajectory, display_size)
# take mean over pic_mode, image_index, seed
# temperature is only not NaN for timeout and velo
agg = df
agg['temperature'] = agg['temperature'].fillna('cold')
print(agg.head())
# compute mean and std for frechet per grouping
agg = agg.groupby(['model', 'patch_mode', 'temperature', 'trajectory', 'display_size'])['frechet'].agg(['mean', 'std']).reset_index()
print(agg.head())

# normalize std NaNs to 0.0 for formatting, but keep NaN means as-is
agg['std'] = agg['std'].fillna(0.0)
# create formatted string column mean\pmstd for LaTeX tables; keep NaN for missing means
def _fmt(row):
    if pd.isna(row['mean']):
        return np.nan
    return f"{row['mean']:.2f}\\pm{row['std']:.2f}"

agg['fmt'] = agg.apply(_fmt, axis=1)

# get rid of temperature column, change 'velo' and 'timeout' patch_modes to include temperature
def combine_patch_mode(row):
    if row['patch_mode'] in ['velo', 'timeout']:
        return f"{row['patch_mode']}_{row['temperature']}"
    else:
        return row['patch_mode']

agg['patch_mode'] = agg.apply(combine_patch_mode, axis=1)
agg = agg.drop(columns=['temperature'])
print(agg.head())
# for patch_mode 'optimal', display_size is only 60, so for all other display sizes, add placeholder rows
# optimal_rows = agg[agg['patch_mode'] == 'optimal']
# optimal_display_sizes = set(optimal_rows['display_size'].unique())
# all_display_sizes = set(agg['display_size'].unique())
# missing_display_sizes = all_display_sizes - optimal_display_sizes
# for ds in missing_display_sizes:
#     new_row = {
#         'model': 'frontnet',
#         'patch_mode': 'optimal',
#         'temperature': 'cold',
#         'trajectory': 'all',
#         'display_size': ds,
#         'mean': np.nan,
#         'std': np.nan,
#         'fmt': np.nan,
#     }
#     agg = pd.concat([agg, pd.DataFrame([new_row])], ignore_index=True)

# none_rows = agg[agg['patch_mode'] == 'none']
# print(none_rows.head())
# none_display_sizes = set(none_rows['display_size'].unique())
# all_display_sizes = set(agg['display_size'].unique())
# missing_display_sizes = all_display_sizes - none_display_sizes
# for ds in missing_display_sizes:
#     new_row = {
#         'model': 'frontnet',
#         'patch_mode': 'none',
#         'temperature': 'cold',
#         'trajectory': '',
#         'display_size': ds,
#         'mean': np.nan,
#         'std': np.nan,
#         'fmt': np.nan,
#     }
#     agg = pd.concat([agg, pd.DataFrame([new_row])], ignore_index=True)

# create latex table with row patch_mode, columns display_size, values frechet
# save to paper_results/tables/frechet_{model}.txt
os.makedirs('paper_results/tables', exist_ok=True)
with open(f'paper_results/tables/frechet_{args.model}.txt', 'w') as f:
    for trajectory in agg['trajectory'].unique():
        traj_data = agg[agg['trajectory'] == trajectory]
        table = traj_data.pivot(index='patch_mode', columns='display_size', values='fmt')
        # replace NaN with '-' for missing cells and allow LaTeX \pm to be preserved
        table = table.fillna('-')
        f.write(f"Trajectory: {trajectory}\n")
        f.write(table.to_latex(escape=False))
        f.write("\n\n")

# --- Computation time saved table (mean_time_per_step) ---
# aggregate mean and std for mean_time_per_step per grouping (same grouping as frechet)
time_df = df.copy()
time_df['temperature'] = time_df['temperature'].fillna('cold')
time_agg = time_df.groupby(['model', 'patch_mode', 'temperature', 'trajectory', 'display_size'])['mean_time_per_step'].agg(['mean', 'std']).reset_index()

# normalize std NaNs to 0.0 for formatting, but keep NaN means as-is
time_agg['std'] = time_agg['std'].fillna(0.0)
def _fmt_time(row):
    if pd.isna(row['mean']):
        return np.nan
    return f"{row['mean']:.2f}\\pm{row['std']:.2f}"

time_agg['fmt'] = time_agg.apply(_fmt_time, axis=1)

# combine temperature into patch_mode for timeout/velo as done for frechet
time_agg['patch_mode'] = time_agg.apply(combine_patch_mode, axis=1)
time_agg = time_agg.drop(columns=['temperature'])

# # ensure 'optimal' has placeholder rows for missing display sizes
# optimal_rows_time = time_agg[time_agg['patch_mode'] == 'optimal']
# optimal_display_sizes_time = set(optimal_rows_time['display_size'].unique())
# all_display_sizes_time = set(time_agg['display_size'].unique())
# missing_display_sizes_time = all_display_sizes_time - optimal_display_sizes_time
# for ds in missing_display_sizes_time:
#     new_row = {
#         'model': 'frontnet',
#         'patch_mode': 'optimal',
#         'temperature': 'cold',
#         'trajectory': 'all',
#         'display_size': ds,
#         'mean': np.nan,
#         'std': np.nan,
#         'fmt': np.nan,
#     }
#     time_agg = pd.concat([time_agg, pd.DataFrame([new_row])], ignore_index=True)

# # ensure 'none' has placeholder rows for missing display sizes
# none_rows_time = time_agg[time_agg['patch_mode'] == 'none']
# none_display_sizes_time = set(none_rows_time['display_size'].unique())
# all_display_sizes_time = set(time_agg['display_size'].unique())
# missing_display_sizes_time = all_display_sizes_time - none_display_sizes_time
# for ds in missing_display_sizes_time:
#     new_row = {
#         'model': 'frontnet',
#         'patch_mode': 'none',
#         'temperature': 'cold',
#         'trajectory': 'all',
#         'display_size': ds,
#         'mean': np.nan,
#         'std': np.nan,
#         'fmt': np.nan,
#     }
#     time_agg = pd.concat([time_agg, pd.DataFrame([new_row])], ignore_index=True)

# write time table to paper_results/tables/computation_time_{model}.txt
with open(f'paper_results/tables/computation_time_{args.model}.txt', 'w') as f:
    for trajectory in time_agg['trajectory'].unique():
        traj_data = time_agg[time_agg['trajectory'] == trajectory]
        table = traj_data.pivot(index='patch_mode', columns='display_size', values='fmt')
        table = table.fillna('-')
        f.write(f"Trajectory: {trajectory}\n")
        f.write(table.to_latex(escape=False))
        f.write("\n\n")


target_trajectories = {"figure8": gen_target_trajectory("figure8"),
                        "triangle": gen_target_trajectory("triangle"),
                        "u": gen_target_trajectory("u"),
                        "s": gen_target_trajectory("s"),
                        "slingshot_left": gen_target_trajectory("slingshot_left"),
                        "slingshot_right": gen_target_trajectory("slingshot_right"),
                        "slingshot_forward": gen_target_trajectory("slingshot_forward"),
                        "slingshot_backward": gen_target_trajectory("slingshot_backward")}

# now plot each target trajectory as a 2D plot (ommitting z and yaw) for each patch_mode
# load all_drone_poses.npy, path stored in file_path column
# interpolate pad with last entry to max length and then take mean over seeds, pic_modes, display_sizes
# add target trajectory as dashed line


os.makedirs('paper_results/plots', exist_ok=True)

for trajectory in agg['trajectory'].unique():
    traj_df = df[df['trajectory'] == trajectory]
    plt.figure(figsize=(8, 8))
    for patch_mode in traj_df['patch_mode'].unique():
        mode_df = traj_df[traj_df['patch_mode'] == patch_mode]
        all_poses = []
        for fp in mode_df['file_path'].unique():
            try:
                poses = np.load(fp)  # (n_steps, 4)
            except Exception:
                continue
            all_poses.append(poses)
        if not all_poses:
            continue
        # find max length
        max_len = max(p.shape[0] for p in all_poses)
        # pad each to max length
        padded = []
        for p in all_poses:
            if p.shape[0] < max_len:
                last = p[-1:]
                reps = max_len - p.shape[0]
                pad = np.vstack([p, np.repeat(last, reps, axis=0)])
                padded.append(pad)
            else:
                padded.append(p[:max_len])
        stacked = np.stack(padded, axis=0)
        mean_traj = np.mean(stacked, axis=0)  # (max_len, 4)
        plt.plot(mean_traj[:, 0], mean_traj[:, 1], label=patch_mode)
    # plot target trajectory
    target_traj = target_trajectories[trajectory]
    plt.plot(target_traj[:, 0], target_traj[:, 1], 'r--', label='Target Trajectory')
    plt.title(f'Trajectory: {trajectory}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xlim([-1., 1.])
    plt.ylim([-1., 1.])
    plt.legend()
    plt.grid()
    plt.savefig(f'paper_results/plots/trajectory_{trajectory}_{args.model}.png')
    plt.close()


# create plots for each trajectory showing mean trajectory for velo/timeout warm and cold per display size
for mode in ['velo', 'timeout']:
    for temperature in ['warm', 'cold']:
        for trajectory in agg['trajectory'].unique():
            traj_df = df[(df['trajectory'] == trajectory) & (df['patch_mode'] == mode) & (df['temperature'] == temperature)]
            plt.figure(figsize=(8, 8))
            for display_size in sorted(traj_df['display_size'].unique()):
                size_df = traj_df[traj_df['display_size'] == display_size]
                all_poses = []
                for fp in size_df['file_path'].unique():
                    try:
                        poses = np.load(fp)  # (n_steps, 4)
                    except Exception:
                        continue
                    all_poses.append(poses)
                if not all_poses:
                    continue
                # find max length
                max_len = max(p.shape[0] for p in all_poses)
                # pad each to max length
                padded = []
                for p in all_poses:
                    if p.shape[0] < max_len:
                        last = p[-1:]
                        reps = max_len - p.shape[0]
                        pad = np.vstack([p, np.repeat(last, reps, axis=0)])
                        padded.append(pad)
                    else:
                        padded.append(p[:max_len])
                stacked = np.stack(padded, axis=0)
                mean_traj = np.mean(stacked, axis=0)  # (max_len, 4)
                plt.plot(mean_traj[:, 0], mean_traj[:, 1], label=f'{display_size}z')
            # plot target trajectory
            target_traj = target_trajectories[trajectory]
            plt.plot(target_traj[:, 0], target_traj[:, 1], 'r--', label='Target Trajectory')
            plt.title(f'Trajectory: {trajectory} - {mode} {temperature} by Display Size')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.xlim([-1., 1.])
            plt.ylim([-1., 1.])
            plt.legend()
            plt.grid()
            plt.savefig(f'paper_results/plots/trajectory_{trajectory}_{mode}_{temperature}_by_display_size_{args.model}.png')
            plt.close()

for mode in ['diffusion', 'corpus']:
    for trajectory in agg['trajectory'].unique():
        traj_df = df[(df['trajectory'] == trajectory) & (df['patch_mode'] == mode)]
        plt.figure(figsize=(8, 8))
        for display_size in sorted(traj_df['display_size'].unique()):
            size_df = traj_df[traj_df['display_size'] == display_size]
            all_poses = []
            for fp in size_df['file_path'].unique():
                try:
                    poses = np.load(fp)  # (n_steps, 4)
                except Exception:
                    continue
                all_poses.append(poses)
            if not all_poses:
                continue
            # find max length
            max_len = max(p.shape[0] for p in all_poses)
            # pad each to max length
            padded = []
            for p in all_poses:
                if p.shape[0] < max_len:
                    last = p[-1:]
                    reps = max_len - p.shape[0]
                    pad = np.vstack([p, np.repeat(last, reps, axis=0)])
                    padded.append(pad)
                else:
                    padded.append(p[:max_len])
            stacked = np.stack(padded, axis=0)
            mean_traj = np.mean(stacked, axis=0)  # (max_len, 4)
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], label=f'{display_size}z')
        # plot target trajectory
        target_traj = target_trajectories[trajectory]
        plt.plot(target_traj[:, 0], target_traj[:, 1], 'r--', label='Target Trajectory')
        plt.title(f'Trajectory: {trajectory} - {mode} by Display Size')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xlim([-1., 1.])
        plt.ylim([-1., 1.])
        plt.legend()
        plt.grid()
        plt.savefig(f'paper_results/plots/trajectory_{trajectory}_{mode}_by_display_size_{args.model}.png')
        plt.close()



# # replace all NaN temperatures with 'cold' for frontnet
# agg['temperature'] = agg['temperature'].fillna('cold')
# print(agg[agg['patch_mode'] == 'black'].head())

# # create table for frontnet, all 'cold' or NaN temperature per trajectory
# frontnet_agg = agg[(agg['temperature'] == 'cold')]
# print(frontnet_agg.head())

# create latex table with row patch_mode, columns display_size, values frechet
# for trajectory in agg['trajectory'].unique():
#     traj_data = agg[agg['trajectory'] == trajectory]
#     table = traj_data.pivot(index='patch_mode', columns='display_size', values='frechet')
#     print(f"Trajectory: {trajectory}")
#     print(table.to_latex(float_format="%.2f"))
#     print("\n")


# # patch structure: model/patch_mode/(temperature/)?trajectory/{display_size}z/pic_mode/seed/all_drone_poses.npy
# # MODELS=("frontnet" "yolov5")
# # PATCH_MODES=("random" "black" "timeout" "velo")
# # TEMPERATURES=("warm" "cold")
# # TRAJECTORIES=("figure8" "triangle" "c" "s")
# # DISPLAY_SIZES=(40 50 60 70 80 90 100)
# # PIC_MODES=("idx" "random")
# # TIMEOUT_VALUES=(10 20 30)
# # SEEDS=(0 1 2)
# # warm/cold only for timeout/velo
# # timeout values only for timeout
# # pic_mode is either f'image_{image_index}' or 'random'.
# # patch_mode for timeout is f'timeout_{timeout_value}Hz'

# # take mean over seeds
# # take mean over pic_modes, take mean over all 'image_{image_index}' as 'static', 'random' stays as 'random'

# # poses are saved as n steps: [x, y, z, yaw]

# # Generate the following plots:
# # 1) One plot per trajectory and (if needed) temperature
# # create 2x2 grid for each trajectory, labels should be the display sizes
# # x limits: [-1., 1.]
# # y limits: [-1., 1.]

# # 2) One plot per trajectory and (if needed) temperature
# # create 2x2 grid for each trajectory, labels should be the pic modes, take the mean over seeds and display sizes

# # 2) One plot per trajectory, mean over display sizes, pic modes, seeds for timeout and velo temperatures, labels should be velo_warm, velo_cold, timeout_10Hz, timeout_20Hz, timeout_30Hz

# def plot_trajectories(poses_dict, save_path):
#     """Plot per-trajectory grids from a structured poses dict.

#     poses_dict: dict keyed by (model, trajectory, temperature) -> {display_size: {pic_mode: poses(np.ndarray)}}
#     """
#     for key, traj_data in poses_dict.items():
#         # support keys of form (model, trajectory, temperature) or (model, patch_mode, trajectory, temperature)
#         if isinstance(key, tuple) and len(key) == 4:
#             model, patch_mode, trajectory, temperature = key
#         elif isinstance(key, tuple) and len(key) == 3:
#             model, trajectory, temperature = key
#             patch_mode = None
#         else:
#             continue
#         temp_label = temperature if temperature is not None else 'all_temps'

#         displays = sorted(traj_data.keys())
#         n = len(displays)
#         ncols = 2
#         nrows = max(1, (n + ncols - 1) // ncols)

#         fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
#         fig.suptitle(f'Trajectory: {trajectory}, Temperature: {temp_label}')

#         axs_flat = np.array(axs).ravel()

#         for i, display_size in enumerate(displays):
#             ax = axs_flat[i]
#             for pic_mode, poses in traj_data[display_size].items():
#                 ax.plot(poses[:, 0], poses[:, 1], label=pic_mode)
#             ax.set_title(f'Display Size: {display_size}z')
#             ax.set_xlim([-1., 1.])
#             ax.set_ylim([-1., 1.])
#             ax.set_xlabel('X Position')
#             ax.set_ylabel('Y Position')
#             ax.legend()

#         # hide any unused subplots
#         for j in range(i + 1, axs_flat.size):
#             axs_flat[j].axis('off')

#         plt.tight_layout()
#         output_file = pathlib.Path(save_path) / f'{model}_{patch_mode if patch_mode else "all"}_trajectory_{trajectory}_temp_{temp_label}.png'
#         plt.savefig(output_file)
#         plt.close()


# def find_all_drone_poses(base_path=None, save_txt=True, out_filename="all_drone_poses_paths.txt"):
#     """Recursively find, load and aggregate all `all_drone_poses.npy` under `base_path`.

#     Returns a tuple (structured_dict, n_matches).

#     structured_dict structure: {
#         (model, trajectory, temperature_or_None): {
#             display_size_int: {
#                 pic_mode: poses_array (n_steps x 4)  # mean over seeds
#             }
#         }
#     }
#     """
#     if base_path is None:
#         base = pathlib.Path(__file__).parent
#     else:
#         base = pathlib.Path(base_path)

#     matches = sorted(base.rglob("all_drone_poses.npy"))

#     # write flat list if requested
#     if save_txt:
#         out_path = pathlib.Path(__file__).parent / out_filename
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(out_path, "w") as f:
#             for p in matches:
#                 f.write(str(p) + "\n")

#     # aggregate
#     structured = {}
#     for p in matches:
#         try:
#             rel = p.relative_to(base).parts
#         except Exception:
#             # fallback to absolute parts
#             rel = p.parts

#         if len(rel) < 6:
#             continue

#         # Basic layout variations:
#         # model/patch_mode/trajectory/...  (random/black)
#         # model/patch_mode/temperature/trajectory/... (timeout/velo)
#         model = rel[0]
#         patch_mode = rel[1] if len(rel) > 1 else ''

#         # initialize
#         temperature = None
#         trajectory = None
#         display = None
#         pic_mode = None
#         seed = None

#         # handle timeout and velo which include temperature directory
#         if patch_mode.startswith('timeout_') or patch_mode == 'velo':
#             if len(rel) >= 7:
#                 # model, patch_mode, temperature, trajectory, display, pic_mode, seed, file
#                 temperature = rel[2] if rel[2] in ('warm', 'cold') else None
#                 trajectory = rel[3]
#                 display = rel[4]
#                 pic_mode = rel[5]
#                 seed = rel[6]
#             else:
#                 continue
#         else:
#             # model, patch_mode, trajectory, display, pic_mode, seed, file
#             if len(rel) >= 6:
#                 trajectory = rel[2]
#                 display = rel[3]
#                 pic_mode = rel[4]
#                 seed = rel[5]
#             else:
#                 continue

#         # normalize display to integer
#         display_str = str(display)
#         if display_str.endswith('z'):
#             display_num = display_str[:-1]
#         else:
#             display_num = display_str
#         try:
#             display_num = int(display_num)
#         except Exception:
#             # leave as string if not integer
#             pass

#         # load poses
#         try:
#             arr = np.load(p)
#         except Exception:
#             continue

#         key = (model, patch_mode, trajectory, temperature)
#         structured.setdefault(key, {}).setdefault(display_num, {}).setdefault(pic_mode, []).append(arr)

#     # convert lists -> mean arrays (mean over seeds)
#     # Ensure arrays are padded to the maximum length within each key (repeat last row)
#     for key in list(structured.keys()):
#         # find maximum length across all arrays under this key
#         max_len = 0
#         for display_num in structured[key].keys():
#             for pic_mode in structured[key][display_num].keys():
#                 for a in structured[key][display_num][pic_mode]:
#                     if a is None:
#                         continue
#                     max_len = max(max_len, a.shape[0])

#         if max_len == 0:
#             # nothing to do for this key
#             for display_num in list(structured[key].keys()):
#                 for pic_mode in list(structured[key][display_num].keys()):
#                     structured[key][display_num][pic_mode] = np.empty((0, 4))
#             continue

#         # pad each array to max_len by repeating last row, then average
#         for display_num in list(structured[key].keys()):
#             for pic_mode in list(structured[key][display_num].keys()):
#                 arrs = structured[key][display_num][pic_mode]
#                 if not arrs:
#                     structured[key][display_num][pic_mode] = np.empty((0, 4))
#                     continue
#                 padded = []
#                 for a in arrs:
#                     if a.shape[0] < max_len:
#                         if a.shape[0] == 0:
#                             # pad with zeros if empty
#                             pad = np.zeros((max_len, a.shape[1] if a.ndim > 1 else 1))
#                         else:
#                             last = a[-1:]
#                             reps = max_len - a.shape[0]
#                             pad = np.vstack([a, np.repeat(last, reps, axis=0)])
#                         padded.append(pad)
#                     else:
#                         padded.append(a[:max_len])
#                 stacked = np.stack(padded, axis=0)
#                 mean_arr = np.mean(stacked, axis=0)
#                 structured[key][display_num][pic_mode] = mean_arr

#     return structured, len(matches)


# def save_structured_trajectories(structured, out_dir=None):
#     """Save aggregated trajectories per (model,trajectory,temperature) as compressed .npz files.

#     Each file will be written into `out_dir` (defaults to this script's folder) with name
#     `<model>_trajectory_<trajectory>_temp_<temperature>.npz` and arrays named
#     `display_<display>_pic_<pic_mode>`.
#     """
#     if out_dir is None:
#         out_dir = pathlib.Path(__file__).parent
#     else:
#         out_dir = pathlib.Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     def _safe(s):
#         return re.sub(r'[^A-Za-z0-9]+', '_', str(s))

#     for (model, patch_mode, trajectory, temperature), traj_data in structured.items():
#         temp_label = temperature if temperature is not None else 'all_temps'
#         fname = f"{model}_{patch_mode}_trajectory_{trajectory}_temp_{temp_label}.npz"
#         out_path = out_dir / fname
#         tosave = {}
#         for display_num, display_data in traj_data.items():
#             for pic_mode, arr in display_data.items():
#                 keyname = f"display_{_safe(display_num)}_pic_{_safe(pic_mode)}"
#                 tosave[keyname] = arr
#         try:
#             np.savez_compressed(out_path, **tosave)
#         except Exception:
#             # best-effort: skip on failure
#             continue


# def _mean_traj_from_nested(traj_data):
#     """Return mean trajectory (n_steps x 4) averaging across displays and pic_modes.

#     Combines all arrays found under `traj_data` (a dict display->pic_mode->array), truncates
#     to shortest length and averages.
#     """
#     arrs = []
#     for display_data in traj_data.values():
#         for pic_mode, arr in display_data.items():
#             arrs.append(arr)
#     if not arrs:
#         return None
#     lengths = [a.shape[0] for a in arrs]
#     min_len = min(lengths)
#     stacked = np.stack([a[:min_len] for a in arrs], axis=0)
#     mean_arr = np.mean(stacked, axis=0)
#     return mean_arr


# def plot_per_model(structured, model, modes, save_path, display_sizes=None, include_random=True, include_static=True):
#     """Create one figure for `model` with a subplot per trajectory.

#     - structured: dict keyed by (model, patch_mode, trajectory, temperature)
#     - model: string model name (e.g., 'frontnet')
#     - modes: list of (patch_mode_prefix, temperature) tuples to plot (prefix match, case-insensitive)
#       e.g. [('velo','warm'), ('velo','cold'), ('timeout_10hz','warm'), ('timeout_10hz','cold')]
#     """
#     # collect trajectories for this model
#     trajs = sorted({k[2] for k in structured.keys() if k[0] == model})
#     if not trajs:
#         print(f"No data for model {model}")
#         return

#     n = len(trajs)
#     ncols = 2
#     nrows = max(1, (n + ncols - 1) // ncols)
#     fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
#     axs_flat = np.array(axs).ravel()

#     for i, traj in enumerate(trajs):
#         ax = axs_flat[i]
#         for patch_prefix, temp in modes:
#             # find matching keys
#             matches = [k for k in structured.keys() if k[0] == model and k[2] == traj and ((k[1].lower().startswith(patch_prefix.lower())) or (k[1].lower() == patch_prefix.lower())) and (k[3] == temp)]
#             if not matches:
#                 continue
#             # if multiple patch variants match (unlikely), average them
#             mean_list = []
#             for k in matches:
#                 # optionally filter displays
#                 if display_sizes is not None:
#                     candidate = {d: structured[k][d] for d in structured[k].keys() if d in display_sizes}
#                 else:
#                     candidate = dict(structured[k])

#                 # filter pic_modes inside each display according to include_random/include_static
#                 filtered = {}
#                 for d, display_data in candidate.items():
#                     fm = {}
#                     for pm, arr in display_data.items():
#                         pm_str = str(pm)
#                         is_random = pm_str == 'random'
#                         is_static = pm_str.startswith('image_')
#                         if is_random and not include_random:
#                             continue
#                         if is_static and not include_static:
#                             continue
#                         fm[pm] = arr
#                     if fm:
#                         filtered[d] = fm

#                 mean_arr = _mean_traj_from_nested(filtered)
#                 if mean_arr is not None:
#                     mean_list.append(mean_arr)
#             if not mean_list:
#                 continue
#             # align lengths and average
#             min_len = min(a.shape[0] for a in mean_list)
#             stacked = np.stack([a[:min_len] for a in mean_list], axis=0)
#             mean_final = np.mean(stacked, axis=0)
#             if temp is None:
#                 label = f"{patch_prefix}"
#             else:
#                 label = f"{patch_prefix}_{temp}"
#             ax.plot(mean_final[:, 0], mean_final[:, 1], label=label)

#         ax.set_title(traj)
#         ax.set_xlim([-1., 1.])
#         ax.set_ylim([-1., 1.])
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.legend()

#     for j in range(i + 1, axs_flat.size):
#         axs_flat[j].axis('off')

#     plt.tight_layout()
#     out_file = pathlib.Path(save_path) / f'{model}_trajectories_modes.png'
#     plt.savefig(out_file)
#     plt.close()

# def plot_overall(poses_dict, save_path):
#     """Plot overall (mean across display sizes) trajectories per pic_mode.

#     Expects the same structured dict as `plot_trajectories`.
#     """
#     for key, traj_data in poses_dict.items():
#         # support keys with/without patch_mode
#         if isinstance(key, tuple) and len(key) == 4:
#             model, patch_mode, trajectory, temperature = key
#         elif isinstance(key, tuple) and len(key) == 3:
#             model, trajectory, temperature = key
#             patch_mode = None
#         else:
#             continue
#         temp_label = temperature if temperature is not None else 'all_temps'

#         # aggregate per pic_mode across display sizes
#         mode_acc = {}
#         for display_data in traj_data.values():
#             for pic_mode, poses in display_data.items():
#                 mode_acc.setdefault(pic_mode, []).append(poses)

#         # compute mean across displays for each pic_mode
#         mode_mean = {}
#         for pic_mode, arrs in mode_acc.items():
#             # ensure same length, truncate to shortest if necessary
#             lengths = [a.shape[0] for a in arrs]
#             min_len = min(lengths)
#             stacked = np.stack([a[:min_len] for a in arrs], axis=0)
#             mean_arr = np.mean(stacked, axis=0)
#             mode_mean[pic_mode] = mean_arr

#         fig, ax = plt.subplots(figsize=(8, 8))
#         fig.suptitle(f'Overall Trajectory: {trajectory}, Temperature: {temp_label}')

#         for mode_key, poses in mode_mean.items():
#             ax.plot(poses[:, 0], poses[:, 1], label=mode_key)

#         ax.set_xlim([-1., 1.])
#         ax.set_ylim([-1., 1.])
#         ax.set_xlabel('X Position')
#         ax.set_ylabel('Y Position')
#         ax.legend()

#         plt.tight_layout()
#         output_file = pathlib.Path(save_path) / f'{model}_overall_trajectory_{trajectory}_temp_{temp_label}.png'
#         plt.savefig(output_file)
#         plt.close()

# if __name__ == "__main__":
#     # Example usage
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--display-sizes', help='Comma-separated display sizes to include, e.g. 40,50,60', default=None)
#     # allow explicit enable/disable of random/static pic modes
#     if hasattr(argparse, 'BooleanOptionalAction'):
#         parser.add_argument('--include-random', action=argparse.BooleanOptionalAction, default=True, help='Include random pic_mode')
#         parser.add_argument('--include-static', action=argparse.BooleanOptionalAction, default=True, help='Include static image pic_modes')
#     else:
#         parser.add_argument('--include-random', action='store_true', default=True, help='Include random pic_mode')
#         parser.add_argument('--include-static', action='store_true', default=True, help='Include static image pic_modes')
#     args = parser.parse_args()

#     structured, n_matches = find_all_drone_poses()
#     print(f"Found {n_matches} all_drone_poses.npy files. Paths written to 'all_drone_poses_paths.txt' in paper_results/.")

#     save_path = 'plots_output'
#     os.makedirs(save_path, exist_ok=True)

#     # save aggregated per-(model,patch_mode,trajectory,temperature) .npz files
#     save_structured_trajectories(structured, out_dir=pathlib.Path(__file__).parent)

#     # Plot for frontnet: mean trajectories for velo_warm, velo_cold, timeout_10Hz warm/cold
#     modes = [
#         ('velo', 'warm'),
#         ('velo', 'cold'),
#         ('timeout_10hz', 'warm'),
#         ('timeout_10hz', 'cold'),
#         ('random', None),
#         ('black', None),
#     ]
#     # parse display sizes CLI arg
#     if args.display_sizes:
#         try:
#             display_sizes = [int(x.strip()) for x in args.display_sizes.split(',') if x.strip()]
#         except Exception:
#             display_sizes = None
#     else:
#         display_sizes = None

#     plot_per_model(
#         structured,
#         'frontnet',
#         modes,
#         save_path,
#         display_sizes=display_sizes,
#         include_random=args.include_random,
#         include_static=args.include_static,
#     )
#     # Collect all all_drone_poses.npy files (and save paths to paper_results/all_drone_poses_paths.txt)
