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

                os.makedirs(path, exist_ok=True)

                mean_distance_per_seed = []
                std_distance_per_seed = []

                for i in range(10):
                    seed_dir = f'{path}/{i}'
                    os.makedirs(seed_dir, exist_ok=True)

                    distances_fp = f'{seed_dir}/distances.npy'
                    mean_fp = f'{seed_dir}/mean_distance.npy'
                    std_fp = f'{seed_dir}/std_distance.npy'
                    poses_fp = f'{seed_dir}/all_drone_poses.npy'

                    # Load distances if present, otherwise compute them (if poses exist)
                    if os.path.exists(distances_fp):
                        distances = np.load(distances_fp)
                    else:
                        if not os.path.exists(poses_fp):
                            print(f"[gen_data] Missing poses for {poses_fp}, skipping seed {i}.")
                            continue
                        all_drone_poses = np.load(poses_fp)
                        distances = np.array([
                            euclidean_distance(all_drone_poses[j], target_trajectory[j])
                            for j in range(len(target_trajectory))
                        ])
                        np.save(distances_fp, distances)

                    # Per-seed stats: load if exist, else compute and save
                    if os.path.exists(mean_fp):
                        mean_d = float(np.load(mean_fp))
                    else:
                        mean_d = float(np.mean(distances))
                        np.save(mean_fp, np.array(mean_d))

                    if os.path.exists(std_fp):
                        std_d = float(np.load(std_fp))
                    else:
                        std_d = float(np.std(distances))
                        np.save(std_fp, np.array(std_d))

                    mean_distance_per_seed.append(mean_d)
                    std_distance_per_seed.append(std_d)

                # Aggregate across available seeds
                if len(mean_distance_per_seed) == 0:
                    print(f"[gen_data] No seeds available for {path}, skipping aggregation.")
                    continue

                agg_mean_fp = f'{path}/mean_distance_over_seeds.npy'
                agg_std_fp = f'{path}/std_distance_over_seeds.npy'
                if not os.path.exists(agg_mean_fp):
                    np.save(agg_mean_fp, np.array(np.mean(mean_distance_per_seed)))
                if not os.path.exists(agg_std_fp):
                    np.save(agg_std_fp, np.array(np.std(mean_distance_per_seed)))


def gen_plot_per_image(mode, trajectory, monitor_size, img_idx):
    if img_idx == 'random':
        path = f'{mode}/{trajectory}/{monitor_size}z/random'
    else:
        path = f'{mode}/{trajectory}/{monitor_size}z/image_{img_idx}'
    os.makedirs(path, exist_ok=True)

    # Load available seeds
    seed_files = [f'{path}/{i}/distances.npy' for i in range(10)]
    existing = [fp for fp in seed_files if os.path.exists(fp)]
    if not existing:
        print(f"[gen_plot_per_image] No distances found at {path}.")
        return np.array([])

    all_distances = np.array([np.load(fp) for fp in existing])  # [seed, timestep]

    # Save per-timestep mean/std across seeds if missing
    mean_over_seeds_per_timestep_fp = f'{path}/mean_over_seeds_per_timestep.npy'
    std_over_seeds_per_timestep_fp = f'{path}/std_over_seeds_per_timestep.npy'
    if not os.path.exists(mean_over_seeds_per_timestep_fp) or not os.path.exists(std_over_seeds_per_timestep_fp):
        mean_over_seeds_per_timestep = all_distances.mean(axis=0)
        std_over_seeds_per_timestep = all_distances.std(axis=0)
        np.save(mean_over_seeds_per_timestep_fp, mean_over_seeds_per_timestep)
        np.save(std_over_seeds_per_timestep_fp, std_over_seeds_per_timestep)

    # Save per-seed mean/std over time if missing
    per_seed_mean_over_time_fp = f'{path}/per_seed_mean_over_time.npy'
    per_seed_std_over_time_fp = f'{path}/per_seed_std_over_time.npy'
    if not os.path.exists(per_seed_mean_over_time_fp) or not os.path.exists(per_seed_std_over_time_fp):
        per_seed_mean_over_time = all_distances.mean(axis=1)
        per_seed_std_over_time = all_distances.std(axis=1)
        np.save(per_seed_mean_over_time_fp, per_seed_mean_over_time)
        np.save(per_seed_std_over_time_fp, per_seed_std_over_time)

    # Plot only if missing
    plot_fp = f'{path}/distance_plot.png'
    if not os.path.exists(plot_fp):
        plt.figure(figsize=(10, 6))
        for i in range(all_distances.shape[0]):
            plt.plot(all_distances[i], label=f'Seed {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Euclidean Distance to Target')
        plt.title('Distance to Target Trajectory Over Time')
        plt.legend()
        plt.grid()
        plt.savefig(plot_fp)
        plt.close()

    return all_distances


def gen_plot_per_monitor_size(mode, trajectory, monitor_size):
    distances_per_size = []
    img_labels = []
    for img_idx in IMG_IDX:
        distances = gen_plot_per_image(mode, trajectory, monitor_size, img_idx)  # [seed, timestep]
        if distances.size == 0:
            continue
        distances_per_size.append(np.mean(distances, axis=0))  # mean over seeds -> curve per image
        img_labels.append(img_idx)

    distances_per_size = np.array(distances_per_size)  # [image, timestep]

    out_dir = f'{mode}/{trajectory}/{monitor_size}z'
    os.makedirs(out_dir, exist_ok=True)

    # Save per-timestep mean/std across images if missing
    per_timestep_mean_fp = f'{out_dir}/per_timestep_mean_over_images.npy'
    per_timestep_std_fp = f'{out_dir}/per_timestep_std_over_images.npy'
    if not os.path.exists(per_timestep_mean_fp) or not os.path.exists(per_timestep_std_fp):
        if distances_per_size.size > 0:
            np.save(per_timestep_mean_fp, distances_per_size.mean(axis=0))
            np.save(per_timestep_std_fp, distances_per_size.std(axis=0))

    # Save per-image mean/std over time if missing
    per_image_mean_fp = f'{out_dir}/per_image_mean_over_time.npy'
    per_image_std_fp = f'{out_dir}/per_image_std_over_time.npy'
    if not os.path.exists(per_image_mean_fp) or not os.path.exists(per_image_std_fp):
        if distances_per_size.size > 0:
            np.save(per_image_mean_fp, distances_per_size.mean(axis=1))
            np.save(per_image_std_fp, distances_per_size.std(axis=1))

    # Save overall mean/std if missing
    overall_mean_fp = f'{out_dir}/overall_mean.npy'
    overall_std_fp = f'{out_dir}/overall_std.npy'
    if not os.path.exists(overall_mean_fp) or not os.path.exists(overall_std_fp):
        if distances_per_size.size > 0:
            np.save(overall_mean_fp, np.array(distances_per_size.mean()))
            np.save(overall_std_fp, np.array(distances_per_size.std()))

    # Violin plot of the per-image curves only if missing
    violin_fp = f'{out_dir}/mean_distance_violin_plot.png'
    if not os.path.exists(violin_fp) and distances_per_size.size > 0:
        plt.figure(figsize=(10, 7))
        plt.violinplot(distances_per_size.T, showmeans=True)
        plt.xlabel('Image Index')
        plt.xticks(ticks=range(1, len(img_labels) + 1), labels=img_labels, rotation=45)
        plt.ylabel('Mean Euclidean Distance to Target')
        plt.title(f'Mean Distance to Target Trajectory for Monitor Size {monitor_size}z')
        plt.grid()
        plt.tight_layout()
        plt.savefig(violin_fp)
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

        means_fp = f'{traj_dir}/mean_distance_per_monitor_size_values.npy'
        stds_fp = f'{traj_dir}/std_distance_per_monitor_size_values.npy'
        if not os.path.exists(means_fp):
            np.save(means_fp, means)
        if not os.path.exists(stds_fp):
            np.save(stds_fp, stds)

        plot_fp = f'{traj_dir}/mean_distance_per_monitor_size_bar.png'
        if not os.path.exists(plot_fp):
            plt.figure(figsize=(6, 4))
            plt.bar([f'{ms}z' for ms in MONITOR_SIZES], means, yerr=stds, capsize=4,
                    color='skyblue', edgecolor='black')
            plt.xlabel('Monitor Size')
            plt.ylabel('Mean Euclidean Distance to Target')
            plt.title(f'Mean Distance per Monitor Size\nTrajectory: {trajectory}')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_fp)
            plt.close()


def plot_mean_per_trajectory(mode, distances_per_trajectory):
    """
    distances_per_trajectory shape: [num_trajectories, num_monitor_sizes, num_images, num_timesteps]
    Creates a violin plot per trajectory; saves mean and std per trajectory.
    """
    os.makedirs(mode, exist_ok=True)
    per_image_means = np.mean(distances_per_trajectory, axis=3)  # [traj, ms, img]
    per_image_means_fp = f'{mode}/per_image_mean_by_traj_ms_img.npy'
    mean_traj_fp = f'{mode}/mean_distance_per_trajectory.npy'
    std_traj_fp = f'{mode}/std_distance_per_trajectory.npy'
    violin_fp = f'{mode}/mean_distance_per_trajectory_violin.png'

    if not os.path.exists(per_image_means_fp):
        np.save(per_image_means_fp, per_image_means)

    data = [per_image_means[i].reshape(-1) for i in range(per_image_means.shape[0])]
    means_per_traj = np.array([d.mean() for d in data])
    stds_per_traj = np.array([d.std() for d in data])

    if not os.path.exists(mean_traj_fp):
        np.save(mean_traj_fp, means_per_traj)
    if not os.path.exists(std_traj_fp):
        np.save(std_traj_fp, stds_per_traj)

    if not os.path.exists(violin_fp):
        plt.figure(figsize=(10, 5))
        plt.violinplot(data, showmeans=True, showextrema=True)
        plt.xticks(ticks=range(1, len(TRAJECTORIES) + 1), labels=TRAJECTORIES)
        plt.ylabel('Mean Euclidean Distance to Target')
        plt.title('Mean Distance per Trajectory (distribution across images and monitor sizes)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(violin_fp)
        plt.close()


def write_latex_table(mode, distances_per_trajectory, decimals=3):
    """
    distances_per_trajectory: [traj, monitor_size, image, timestep]
    Writes mean ± std over images for each (trajectory, monitor size) and saves both matrices.
    """
    os.makedirs(mode, exist_ok=True)
    mean_mat_fp = f'{mode}/mean_distance_matrix_traj_by_monitor.npy'
    std_mat_fp = f'{mode}/std_distance_matrix_traj_by_monitor.npy'
    table_fp = f'{mode}/mean_std_distance_table.tex'

    # Load if present, otherwise compute and save
    if os.path.exists(mean_mat_fp) and os.path.exists(std_mat_fp):
        means = np.load(mean_mat_fp)
        stds = np.load(std_mat_fp)
    else:
        means = np.mean(distances_per_trajectory, axis=(2, 3))
        stds = np.std(distances_per_trajectory, axis=(2, 3))
        np.save(mean_mat_fp, means)
        np.save(std_mat_fp, stds)

    if not os.path.exists(table_fp):
        header_cols = ' & '.join(f'{ms}z' for ms in MONITOR_SIZES)
        safe_label = mode.replace('/', '_').replace(' ', '_')

        lines = []
        lines.append(r'\begin{table}[ht]')
        lines.append(r'\centering')
        lines.append(fr'\caption{{Mean $\pm$ std of distance over images per monitor size and trajectory ({mode}).}}')
        lines.append(fr'\label{{tab:mean_std_distance_{safe_label}}}')
        alignment = 'l' + 'c' * len(MONITOR_SIZES)
        lines.append(r'\begin{tabular}{' + alignment + '}')
        lines.append(r'\hline')
        lines.append(r'Trajectory & ' + header_cols + r' \\')
        lines.append(r'\hline')

        for i, traj in enumerate(TRAJECTORIES):
            row_cells = [f'${means[i, j]:.{decimals}f} \\pm {stds[i, j]:.{decimals}f}$'
                         for j in range(len(MONITOR_SIZES))]
            lines.append(f'{traj} & ' + ' & '.join(row_cells) + r' \\')

        lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        tex = '\n'.join(lines)

        with open(table_fp, 'w') as f:
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
                row_cells.append(f'${means[ti, mj]:.{decimals}f} \\pm {stds[ti, mj]:.{decimals}f}$')
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

        try:
            # Compute missing distances/stats only; load existing where present
            gen_data(mode)
        except Exception as e:
            print(f"Error generating data for mode {mode}: {e}")
            continue

        distances_per_trajectory = []
        for trajectory in TRAJECTORIES:
            traj_dir = f'{mode}/{trajectory}'
            os.makedirs(traj_dir, exist_ok=True)

            cached_ms_fp = f'{traj_dir}/mean_distance_per_monitor_size.npy'
            if os.path.exists(cached_ms_fp):
                mean_per_monitor_size = np.load(cached_ms_fp, allow_pickle=False)
            else:
                mean_per_monitor_size = []
                for monitor_size in MONITOR_SIZES:
                    distance_per_size = gen_plot_per_monitor_size(mode, trajectory, monitor_size)
                    mean_per_monitor_size.append(distance_per_size)
                mean_per_monitor_size = np.array(mean_per_monitor_size)
                np.save(cached_ms_fp, mean_per_monitor_size)

            distances_per_trajectory.append(mean_per_monitor_size)

        distances_per_trajectory = np.array(distances_per_trajectory) # [traj, ms, img, timestep]

        # Per-mode summary matrices for the mega table
        mode_means = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
        mode_stds  = np.std(distances_per_trajectory, axis=(2, 3))   # [traj, ms]
        all_mode_means[mode] = mode_means
        all_mode_stds[mode] = mode_stds

        # Create the violin plot for each trajectory (under mode root) only if missing
        mode_violin_fp = f'{mode}/all_trajectories_mean_distance_violin_plot.png'
        if not os.path.exists(mode_violin_fp):
            os.makedirs(mode, exist_ok=True)
            plt.figure(figsize=(15, 10))
            y_min, y_max = float('inf'), float('-inf')
            for i, trajectory in enumerate(TRAJECTORIES):
                data = np.mean(distances_per_trajectory[i], axis=1).T
                if data.size > 0:
                    y_min = min(y_min, data.min())
                    y_max = max(y_max, data.max())
            for i, trajectory in enumerate(TRAJECTORIES):
                plt.subplot(2, 3, i+1)
                data = np.mean(distances_per_trajectory[i], axis=1).T
                if data.size > 0:
                    plt.violinplot(data, showmeans=True)
                plt.xlabel('Monitor Size')
                plt.xticks(ticks=range(1, len(MONITOR_SIZES) + 1), 
                        labels=MONITOR_SIZES)
                plt.ylabel('Mean Euclidean Distance to Target')
                plt.title(f'{trajectory} Trajectory')
                if y_min < y_max:
                    plt.ylim(y_min, y_max+1.)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_violin_fp)
            plt.close()

        # Aggregated plots and tables (each function internally skips if outputs exist)
        plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory)
        plot_mean_per_trajectory(mode, distances_per_trajectory)
        write_latex_table(mode, distances_per_trajectory)

    # Write the mega table across all modes at the repo root
    write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_distance_all_modes_table.tex')


