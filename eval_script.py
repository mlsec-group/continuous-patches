import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from attack_minimal_single import gen_target_trajectory, normalize_yaw

MODES = ['optimal/cold', 'optimal/warm', 'timeout_10Hz/cold', 
         'timeout_10Hz/warm', 'timeout_20Hz/cold', 'timeout_20Hz/warm', 
         'timeout_30Hz/cold', 'timeout_30Hz/warm', 'white', 'black', 'random']
MONITOR_SIZES = [30, 60, 90, 120]
IMG_IDX = [505, 4847, 3059, 1860, 3205, 4861, 2613, 2309, 5431, 2847, 'random']
TRAJECTORIES = ["figure8", "square", "circle", "line_y", "line_x"]


def euclidean_distance(a, b):
    distance = np.linalg.norm(a[:3] - b[:3])
    return distance

def angular_error(a, b):
    angular_loss = 1 - np.cos(normalize_yaw(a[3])) - normalize_yaw((b[3]))
    return angular_loss


def gen_data(mode):
    for trajectory in tqdm(TRAJECTORIES):
        target_trajectory = gen_target_trajectory(trajectory).detach().cpu().numpy()

        for monitor_size in MONITOR_SIZES:
            for img_idx in IMG_IDX:
                if img_idx == 'random':
                    path = f'{mode}/{trajectory}/{monitor_size}z/random'
                else:
                    path = f'{mode}/{trajectory}/{monitor_size}z/image_{img_idx}'

                # ensure directory exists for outputs
                os.makedirs(path, exist_ok=True)

                all_drone_poses_p_seed = [np.load(f'{path}/{i}/all_drone_poses.npy') for i in range(10)]
                all_drone_poses_p_seed = np.array(all_drone_poses_p_seed)

                mean_distance_per_seed = []
                std_distance_per_seed = []
                for i in range(10):
                    distances = [euclidean_distance(all_drone_poses_p_seed[i][j], target_trajectory[j]) for j in range(len(target_trajectory))]
                    mean_distance_per_seed.append(np.mean(distances))
                    std_distance_per_seed.append(np.std(distances))
                    np.save(f'{path}/{i}/distances.npy', np.array(distances))
                    np.save(f'{path}/{i}/mean_distance.npy', np.array(mean_distance_per_seed[-1]))
                    np.save(f'{path}/{i}/std_distance.npy', np.array(std_distance_per_seed[-1]))

                mean_for_idx = np.mean(mean_distance_per_seed)
                std_for_idx = np.std(mean_distance_per_seed)
                np.save(f'{path}/mean_distance_over_seeds.npy', np.array(mean_for_idx))
                np.save(f'{path}/std_distance_over_seeds.npy', np.array(std_for_idx))


def gen_plot_per_image(mode, trajectory, monitor_size, img_idx):
    if img_idx == 'random':
        path = f'{mode}/{trajectory}/{monitor_size}z/random'
    else:
        path = f'{mode}/{trajectory}/{monitor_size}z/image_{img_idx}'
    os.makedirs(path, exist_ok=True)

    all_distances = [np.load(f'{path}/{i}/distances.npy') for i in range(10)]
    all_distances = np.array(all_distances)  # [seed, timestep]

    # Save per-timestep mean/std across seeds
    mean_over_seeds_per_timestep = all_distances.mean(axis=0)
    std_over_seeds_per_timestep = all_distances.std(axis=0)
    np.save(f'{path}/mean_over_seeds_per_timestep.npy', mean_over_seeds_per_timestep)
    np.save(f'{path}/std_over_seeds_per_timestep.npy', std_over_seeds_per_timestep)

    # Save per-seed mean/std over time (optional, but consistent)
    per_seed_mean_over_time = all_distances.mean(axis=1)
    per_seed_std_over_time = all_distances.std(axis=1)
    np.save(f'{path}/per_seed_mean_over_time.npy', per_seed_mean_over_time)
    np.save(f'{path}/per_seed_std_over_time.npy', per_seed_std_over_time)

    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(all_distances[i], label=f'Seed {i}')
    plt.xlabel('Time Step')
    plt.ylabel('Euclidean Distance to Target')
    plt.title('Distance to Target Trajectory Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(f'{path}/distance_plot.png')
    plt.close()

    return all_distances


def gen_plot_per_monitor_size(mode, trajectory, monitor_size):
    distances_per_size = []
    for img_idx in IMG_IDX:
        distances = gen_plot_per_image(mode, trajectory, monitor_size, img_idx)  # [seed, timestep]
        # mean over seeds per timestep -> a curve per image
        distances_per_size.append(np.mean(distances, axis=0))

    distances_per_size = np.array(distances_per_size)  # [image, timestep]

    out_dir = f'{mode}/{trajectory}/{monitor_size}z'
    os.makedirs(out_dir, exist_ok=True)

    # Save per-timestep mean/std across images
    per_timestep_mean_over_images = distances_per_size.mean(axis=0)
    per_timestep_std_over_images = distances_per_size.std(axis=0)
    np.save(f'{out_dir}/per_timestep_mean_over_images.npy', per_timestep_mean_over_images)
    np.save(f'{out_dir}/per_timestep_std_over_images.npy', per_timestep_std_over_images)

    # Save per-image mean/std over time
    per_image_mean_over_time = distances_per_size.mean(axis=1)
    per_image_std_over_time = distances_per_size.std(axis=1)
    np.save(f'{out_dir}/per_image_mean_over_time.npy', per_image_mean_over_time)
    np.save(f'{out_dir}/per_image_std_over_time.npy', per_image_std_over_time)

    # Also save overall mean/std for convenience
    np.save(f'{out_dir}/overall_mean.npy', np.array(distances_per_size.mean()))
    np.save(f'{out_dir}/overall_std.npy', np.array(distances_per_size.std()))

    # Violin plot of the per-image curves
    plt.figure(figsize=(10, 7))
    plt.violinplot(distances_per_size.T, showmeans=True)
    plt.xlabel('Image Index')
    plt.xticks(ticks=range(1, len(IMG_IDX) + 1), labels=IMG_IDX, rotation=45)
    plt.ylabel('Mean Euclidean Distance to Target')
    plt.title(f'Mean Distance to Target Trajectory for Monitor Size {monitor_size}z')
    plt.grid()
    plt.savefig(f'{out_dir}/mean_distance_violin_plot.png')
    plt.close()
    return distances_per_size


def plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory):
    """
    distances_per_trajectory: [traj, monitor_size, image, timestep]
    Saves mean and std (over images, timesteps) per monitor size and plots bars with error bars.
    """
    means_per_ms = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
    stds_per_ms = np.std(distances_per_trajectory, axis=(2, 3))    # [traj, ms]

    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]
        stds = stds_per_ms[i]
        traj_dir = f'{mode}/{trajectory}'
        os.makedirs(traj_dir, exist_ok=True)

        np.save(f'{traj_dir}/mean_distance_per_monitor_size_values.npy', means)
        np.save(f'{traj_dir}/std_distance_per_monitor_size_values.npy', stds)

        plt.figure(figsize=(6, 4))
        plt.bar([f'{ms}z' for ms in MONITOR_SIZES], means, yerr=stds, capsize=4,
                color='skyblue', edgecolor='black')
        plt.xlabel('Monitor Size')
        plt.ylabel('Mean Euclidean Distance to Target')
        plt.title(f'Mean Distance per Monitor Size\nTrajectory: {trajectory}')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{traj_dir}/mean_distance_per_monitor_size_bar.png')
        plt.close()


def plot_mean_per_trajectory(mode, distances_per_trajectory):
    """
    distances_per_trajectory shape: [num_trajectories, num_monitor_sizes, num_images, num_timesteps]
    Creates a violin plot per trajectory; saves mean and std per trajectory.
    """
    os.makedirs(mode, exist_ok=True)
    per_image_means = np.mean(distances_per_trajectory, axis=3)  # [traj, ms, img]
    data = [per_image_means[i].reshape(-1) for i in range(per_image_means.shape[0])]

    means_per_traj = np.array([d.mean() for d in data])
    stds_per_traj = np.array([d.std() for d in data])
    np.save(f'{mode}/per_image_mean_by_traj_ms_img.npy', per_image_means)
    np.save(f'{mode}/mean_distance_per_trajectory.npy', means_per_traj)
    np.save(f'{mode}/std_distance_per_trajectory.npy', stds_per_traj)

    plt.figure(figsize=(10, 5))
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks(ticks=range(1, len(TRAJECTORIES) + 1), labels=TRAJECTORIES)
    plt.ylabel('Mean Euclidean Distance to Target')
    plt.title('Mean Distance per Trajectory (distribution across images and monitor sizes)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{mode}/mean_distance_per_trajectory_violin.png')
    plt.close()


def write_latex_table(mode, distances_per_trajectory, decimals=3):
    """
    distances_per_trajectory: [traj, monitor_size, image, timestep]
    Writes mean ± std over images for each (trajectory, monitor size) and saves both matrices.
    """
    os.makedirs(mode, exist_ok=True)
    means = np.mean(distances_per_trajectory, axis=(2, 3))
    stds = np.std(distances_per_trajectory, axis=(2, 3))

    np.save(f'{mode}/mean_distance_matrix_traj_by_monitor.npy', means)
    np.save(f'{mode}/std_distance_matrix_traj_by_monitor.npy', stds)

    header_cols = ' & '.join(f'{ms}z' for ms in MONITOR_SIZES)
    safe_label = mode.replace('/', '_').replace(' ', '_')

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(fr'\caption{{Mean $\pm$ std of distance over images per monitor size and trajectory ({mode}).}}')
    lines.append(fr'\label{{tab:mean_std_distance_{safe_label}}}')
    alignment = 'l' + 'c' * len(MONITOR_SIZES)  # Adjust alignment based on the number of columns
    lines.append(r'\begin{tabular}{' + alignment + '}')
    lines.append(r'\hline')
    lines.append(r'Trajectory & ' + header_cols + r' \\')
    lines.append(r'\hline')

    for i, traj in enumerate(TRAJECTORIES):
        row_cells = [f'{means[i, j]:.{decimals}f} \\pm {stds[i, j]:.{decimals}f}'
                     for j in range(len(MONITOR_SIZES))]
        lines.append(f'{traj} & ' + ' & '.join(row_cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines)

    with open(f'{mode}/mean_std_distance_table.tex', 'w') as f:
        f.write(tex)


def write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_distance_all_modes_table.tex'):
    """
    all_mode_means/stds: dict mode -> np.ndarray [traj, monitor_size]
    Writes a single LaTeX table:
      - Rows: modes
      - Column groups: trajectories
      - Subcolumns per trajectory: display sizes (MONITOR_SIZES)
      - Cells: mean ± std
    """
    modes_in_order = [m for m in MODES if m in all_mode_means]
    if not modes_in_order:
        return

    # Build tabular alignment: 1 for Mode + one 'c' per trajectory*monitor size
    num_value_cols = len(TRAJECTORIES) * len(MONITOR_SIZES)
    alignment = 'l' + 'c' * num_value_cols

    # First header row: Mode + trajectory multicolumn headers
    header_row_1 = ['Mode']
    for traj in TRAJECTORIES:
        header_row_1.append(fr'\multicolumn{{{len(MONITOR_SIZES)}}}{{c}}{{{traj}}}')
    header_row_1 = ' & '.join(header_row_1) + r' \\'

    # Second header row: the display sizes repeated for each trajectory
    header_row_2 = ['']
    header_row_2.extend([f'{ms}z' for _ in TRAJECTORIES for ms in MONITOR_SIZES])
    header_row_2 = ' & '.join(header_row_2) + r' \\'

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Mean $\pm$ std of distance over images per monitor size, grouped by trajectory, for all modes.}')
    lines.append(r'\label{tab:mean_std_distance_all_modes}')
    lines.append(r'\begin{tabular}{' + alignment + '}')
    lines.append(r'\hline')
    lines.append(header_row_1)
    lines.append(header_row_2)
    lines.append(r'\hline')

    for mode in modes_in_order:
        means = all_mode_means[mode]  # [traj, ms]
        stds = all_mode_stds[mode]    # [traj, ms]
        row_cells = [mode]
        for ti in range(len(TRAJECTORIES)):
            for mj in range(len(MONITOR_SIZES)):
                row_cells.append(f'{means[ti, mj]:.{decimals}f} \\pm {stds[ti, mj]:.{decimals}f}')
        lines.append(' & '.join(row_cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines)

    with open(outfile, 'w') as f:
        f.write(tex)


if __name__ == "__main__":

    # gen_data('random')
   
    all_mode_means = {}
    all_mode_stds = {}

    for mode in tqdm(MODES):

        # try:
        #     gen_data(mode)
        # except Exception as e:
        #     print(f"Error generating data for mode {mode}: {e}")
        #     continue

        distances_per_trajectory = []
        for trajectory in TRAJECTORIES:
            mean_per_monitor_size = []
            for monitor_size in MONITOR_SIZES:
                # print("Processing trajectory:", trajectory, "Monitor size:", monitor_size)
                distance_per_size = gen_plot_per_monitor_size(mode, trajectory, monitor_size)
                mean_per_monitor_size.append(distance_per_size)
            mean_per_monitor_size = np.array(mean_per_monitor_size)
            traj_dir = f'{mode}/{trajectory}'
            os.makedirs(traj_dir, exist_ok=True)
            np.save(f'{traj_dir}/mean_distance_per_monitor_size.npy', mean_per_monitor_size)
            distances_per_trajectory.append(mean_per_monitor_size)

        distances_per_trajectory = np.array(distances_per_trajectory) # [traj, ms, img, timestep]

        # Per-mode summary matrices for the mega table
        mode_means = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
        mode_stds  = np.std(distances_per_trajectory, axis=(2, 3))   # [traj, ms]
        all_mode_means[mode] = mode_means
        all_mode_stds[mode] = mode_stds

        # Create the violin plot for each trajectory (under mode root)
        os.makedirs(mode, exist_ok=True)
        plt.figure(figsize=(15, 10))
        y_min, y_max = float('inf'), float('-inf')
        for i, trajectory in enumerate(TRAJECTORIES):
            data = np.mean(distances_per_trajectory[i], axis=1).T
            y_min = min(y_min, data.min())
            y_max = max(y_max, data.max())
        for i, trajectory in enumerate(TRAJECTORIES):
            plt.subplot(2, 3, i+1)
            plt.violinplot(np.mean(distances_per_trajectory[i], axis=1).T, showmeans=True)
            plt.xlabel('Monitor Size')
            plt.xticks(ticks=range(1, len(MONITOR_SIZES) + 1), 
                    labels=MONITOR_SIZES)
            plt.ylabel('Mean Euclidean Distance to Target')
            plt.title(f'{trajectory} Trajectory')
            plt.ylim(y_min, y_max+1.)
            plt.grid()
        plt.tight_layout()
        plt.savefig(f'{mode}/all_trajectories_mean_distance_violin_plot.png')
        plt.close()

        # New aggregated plots under mode root:
        plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory)
        plot_mean_per_trajectory(mode, distances_per_trajectory)

        # LaTeX table per mode
        write_latex_table(mode, distances_per_trajectory)

    # Write the mega table across all modes at the repo root
    write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_distance_all_modes_table.tex')


