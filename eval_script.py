import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from attack_minimal_single import gen_target_trajectory, normalize_yaw
import argparse
MODELS = ("frontnet", )
MODES = ['optimal/cold', 'optimal/warm', 'timeout_10Hz/cold', 
         'timeout_10Hz/warm', 'timeout_20Hz/cold', 'timeout_20Hz/warm', 
         'timeout_30Hz/cold', 'timeout_30Hz/warm', 'white', 'black', 'random', 'fap', 
         'diffusion/1000', 'interpolation/1000', 'corpus/1000']
MONITOR_SIZES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
IMG_IDX = [505, 4847, 3059, 1860, 3205, 4861, 2613, 2309, 5431, 2847, 'random']
TRAJECTORIES = ["figure8", "square", "circle", "line_y", "line_x"]


def euclidean_distance(a, b):
    distance = np.linalg.norm(a[:3] - b[:3])
    return distance

def angular_error(a, b):
    # Correct angular error: absolute normalized yaw difference
    delta = normalize_yaw(a[3] - b[3])
    return abs(delta)


def gen_data(model, mode, recalculate=False):
    for trajectory in tqdm(TRAJECTORIES):
        target_trajectory = gen_target_trajectory(trajectory).detach().cpu().numpy()
        for monitor_size in MONITOR_SIZES:
            for img_idx in IMG_IDX:
                if img_idx == 'random':
                    path = f'{model}/{mode}/{trajectory}/{monitor_size}z/random'
                else:
                    path = f'{model}/{mode}/{trajectory}/{monitor_size}z/image_{img_idx}'
                os.makedirs(path, exist_ok=True)

                mean_distance_per_seed = []
                std_distance_per_seed = []
                mean_angular_per_seed = []
                std_angular_per_seed = []

                for i in range(10):
                    seed_dir = f'{path}/{i}'
                    os.makedirs(seed_dir, exist_ok=True)

                    poses_fp = f'{seed_dir}/all_drone_poses.npy'
                    distances_fp = f'{seed_dir}/distances.npy'
                    mean_fp = f'{seed_dir}/mean_distance.npy'
                    std_fp = f'{seed_dir}/std_distance.npy'

                    angular_fp = f'{seed_dir}/angular_errors.npy'
                    mean_ang_fp = f'{seed_dir}/mean_angular_error.npy'
                    std_ang_fp  = f'{seed_dir}/std_angular_error.npy'

                    if not recalculate and os.path.exists(distances_fp) and os.path.exists(angular_fp):
                        distances = np.load(distances_fp)
                        angular_errors = np.load(angular_fp)
                    else:
                        if not os.path.exists(poses_fp):
                            continue
                        all_drone_poses = np.load(poses_fp)
                        distances = np.array([
                            euclidean_distance(all_drone_poses[j], target_trajectory[j])
                            for j in range(len(target_trajectory))
                        ])
                        angular_errors = np.array([
                            angular_error(all_drone_poses[j], target_trajectory[j])
                            for j in range(len(target_trajectory))
                        ])
                        np.save(distances_fp, distances)
                        np.save(angular_fp, angular_errors)

                    if not recalculate and os.path.exists(mean_fp):
                        mean_d = float(np.load(mean_fp))
                    else:
                        mean_d = float(np.mean(distances))
                        np.save(mean_fp, np.array(mean_d))
                    if not recalculate and os.path.exists(std_fp):
                        std_d = float(np.load(std_fp))
                    else:
                        std_d = float(np.std(distances))
                        np.save(std_fp, np.array(std_d))

                    if not recalculate and os.path.exists(mean_ang_fp):
                        mean_a = float(np.load(mean_ang_fp))
                    else:
                        mean_a = float(np.mean(angular_errors))
                        np.save(mean_ang_fp, np.array(mean_a))
                    if not recalculate and os.path.exists(std_ang_fp):
                        std_a = float(np.load(std_ang_fp))
                    else:
                        std_a = float(np.std(angular_errors))
                        np.save(std_ang_fp, np.array(std_a))

                    mean_distance_per_seed.append(mean_d)
                    std_distance_per_seed.append(std_d)
                    mean_angular_per_seed.append(mean_a)
                    std_angular_per_seed.append(std_a)

                if len(mean_distance_per_seed) == 0:
                    continue

                # Distance aggregates
                agg_mean_fp = f'{path}/mean_distance_over_seeds.npy'
                agg_std_fp = f'{path}/std_distance_over_seeds.npy'
                if recalculate or not os.path.exists(agg_mean_fp):
                    np.save(agg_mean_fp, np.array(np.mean(mean_distance_per_seed)))
                if recalculate or not os.path.exists(agg_std_fp):
                    np.save(agg_std_fp, np.array(np.std(mean_distance_per_seed)))

                # Angular aggregates
                agg_mean_ang_fp = f'{path}/mean_angular_error_over_seeds.npy'
                agg_std_ang_fp = f'{path}/std_angular_error_over_seeds.npy'
                if recalculate or not os.path.exists(agg_mean_ang_fp):
                    np.save(agg_mean_ang_fp, np.array(np.mean(mean_angular_per_seed)))
                if recalculate or not os.path.exists(agg_std_ang_fp):
                    np.save(agg_std_ang_fp, np.array(np.std(mean_angular_per_seed)))


def gen_plot_per_image(model, mode, trajectory, monitor_size, img_idx, recalculate=False):
    if img_idx == 'random':
        path = f'{model}/{mode}/{trajectory}/{monitor_size}z/random'
    else:
        path = f'{model}/{mode}/{trajectory}/{monitor_size}z/image_{img_idx}'
    os.makedirs(path, exist_ok=True)

    dist_seed_files = [f'{path}/{i}/distances.npy' for i in range(10)]
    ang_seed_files  = [f'{path}/{i}/angular_errors.npy' for i in range(10)]
    existing = [(d,a) for d,a in zip(dist_seed_files, ang_seed_files) if os.path.exists(d) and os.path.exists(a)]
    if not existing:
        return np.array([]), np.array([])

    all_distances = np.array([np.load(d) for d,_ in existing])
    all_angular   = np.array([np.load(a) for _,a in existing])

    # Per-timestep aggregates
    dist_mean_t_fp = f'{path}/mean_over_seeds_per_timestep.npy'
    dist_std_t_fp  = f'{path}/std_over_seeds_per_timestep.npy'
    ang_mean_t_fp  = f'{path}/angular_mean_over_seeds_per_timestep.npy'
    ang_std_t_fp   = f'{path}/angular_std_over_seeds_per_timestep.npy'
    if recalculate or not (os.path.exists(dist_mean_t_fp) and os.path.exists(dist_std_t_fp)):
        np.save(dist_mean_t_fp, all_distances.mean(axis=0))
        np.save(dist_std_t_fp, all_distances.std(axis=0))
    if recalculate or not (os.path.exists(ang_mean_t_fp) and os.path.exists(ang_std_t_fp)):
        np.save(ang_mean_t_fp, all_angular.mean(axis=0))
        np.save(ang_std_t_fp, all_angular.std(axis=0))

    # Per-seed aggregates
    dist_seed_mean_fp = f'{path}/per_seed_mean_over_time.npy'
    dist_seed_std_fp  = f'{path}/per_seed_std_over_time.npy'
    ang_seed_mean_fp  = f'{path}/angular_per_seed_mean_over_time.npy'
    ang_seed_std_fp   = f'{path}/angular_per_seed_std_over_time.npy'
    if recalculate or not (os.path.exists(dist_seed_mean_fp) and os.path.exists(dist_seed_std_fp)):
        np.save(dist_seed_mean_fp, all_distances.mean(axis=1))
        np.save(dist_seed_std_fp, all_distances.std(axis=1))
    if recalculate or not (os.path.exists(ang_seed_mean_fp) and os.path.exists(ang_seed_std_fp)):
        np.save(ang_seed_mean_fp, all_angular.mean(axis=1))
        np.save(ang_seed_std_fp, all_angular.std(axis=1))

    # Distance plot
    dist_plot_fp = f'{path}/distance_plot.png'
    if recalculate or not os.path.exists(dist_plot_fp):
        plt.figure(figsize=(10,6))
        for i in range(all_distances.shape[0]):
            plt.plot(all_distances[i], label=f'Seed {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Euclidean Distance')
        plt.title('Distance to Target Trajectory Over Time')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(dist_plot_fp)
        plt.close()

    # Angular plot
    ang_plot_fp = f'{path}/angular_error_plot.png'
    if recalculate or not os.path.exists(ang_plot_fp):
        plt.figure(figsize=(10,6))
        for i in range(all_angular.shape[0]):
            plt.plot(all_angular[i], label=f'Seed {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Error (abs yaw diff)')
        plt.title('Angular Error Over Time')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(ang_plot_fp)
        plt.close()

    return all_distances, all_angular


def gen_plot_per_monitor_size(model, mode, trajectory, monitor_size, recalculate=False):
    distances_per_size = []
    angular_per_size = []
    img_labels = []
    for img_idx in IMG_IDX:
        dists, angs = gen_plot_per_image(model, mode, trajectory, monitor_size, img_idx, recalculate=recalculate)
        if dists.size == 0:
            continue
        distances_per_size.append(dists.mean(axis=0))  # mean over seeds
        angular_per_size.append(angs.mean(axis=0))     # mean over seeds
        img_labels.append(img_idx)

    distances_per_size = np.array(distances_per_size)   # [image, timestep]
    angular_per_size   = np.array(angular_per_size)     # [image, timestep]

    out_dir = f'{model}/{mode}/{trajectory}/{monitor_size}z'
    os.makedirs(out_dir, exist_ok=True)

    # Distance summary (existing behavior)
    per_timestep_mean_fp = f'{out_dir}/per_timestep_mean_over_images.npy'
    per_timestep_std_fp = f'{out_dir}/per_timestep_std_over_images.npy'
    if recalculate or not (os.path.exists(per_timestep_mean_fp) and os.path.exists(per_timestep_std_fp)):
        if distances_per_size.size:
            np.save(per_timestep_mean_fp, distances_per_size.mean(axis=0))
            np.save(per_timestep_std_fp, distances_per_size.std(axis=0))

    per_image_mean_fp = f'{out_dir}/per_image_mean_over_time.npy'
    per_image_std_fp = f'{out_dir}/per_image_std_over_time.npy'
    if recalculate or not (os.path.exists(per_image_mean_fp) and os.path.exists(per_image_std_fp)):
        if distances_per_size.size:
            np.save(per_image_mean_fp, distances_per_size.mean(axis=1))
            np.save(per_image_std_fp, distances_per_size.std(axis=1))

    overall_mean_fp = f'{out_dir}/overall_mean.npy'
    overall_std_fp = f'{out_dir}/overall_std.npy'
    if recalculate or not (os.path.exists(overall_mean_fp) and os.path.exists(overall_std_fp)):
        if distances_per_size.size:
            np.save(overall_mean_fp, np.array(distances_per_size.mean()))
            np.save(overall_std_fp, np.array(distances_per_size.std()))

    violin_fp = f'{out_dir}/mean_distance_violin_plot.png'
    if (recalculate or not os.path.exists(violin_fp)) and distances_per_size.size:
        plt.figure(figsize=(10,7))
        plt.violinplot(distances_per_size.T, showmeans=True)
        plt.xlabel('Image Index')
        plt.xticks(ticks=range(1, len(img_labels)+1), labels=img_labels, rotation=45)
        plt.ylabel('Mean Euclidean Distance')
        plt.title(f'Mean Distance per Image ({monitor_size}z)')
        plt.grid()
        plt.tight_layout()
        plt.savefig(violin_fp)
        plt.close()

    # Angular summaries
    ang_per_timestep_mean_fp = f'{out_dir}/angular_per_timestep_mean_over_images.npy'
    ang_per_timestep_std_fp  = f'{out_dir}/angular_per_timestep_std_over_images.npy'
    if recalculate or not (os.path.exists(ang_per_timestep_mean_fp) and os.path.exists(ang_per_timestep_std_fp)):
        if angular_per_size.size:
            np.save(ang_per_timestep_mean_fp, angular_per_size.mean(axis=0))
            np.save(ang_per_timestep_std_fp, angular_per_size.std(axis=0))

    ang_per_image_mean_fp = f'{out_dir}/angular_per_image_mean_over_time.npy'
    ang_per_image_std_fp  = f'{out_dir}/angular_per_image_std_over_time.npy'
    if recalculate or not (os.path.exists(ang_per_image_mean_fp) and os.path.exists(ang_per_image_std_fp)):
        if angular_per_size.size:
            np.save(ang_per_image_mean_fp, angular_per_size.mean(axis=1))
            np.save(ang_per_image_std_fp, angular_per_size.std(axis=1))

    ang_overall_mean_fp = f'{out_dir}/angular_overall_mean.npy'
    ang_overall_std_fp  = f'{out_dir}/angular_overall_std.npy'
    if recalculate or not (os.path.exists(ang_overall_mean_fp) and os.path.exists(ang_overall_std_fp)):
        if angular_per_size.size:
            np.save(ang_overall_mean_fp, np.array(angular_per_size.mean()))
            np.save(ang_overall_std_fp, np.array(angular_per_size.std()))

    ang_violin_fp = f'{out_dir}/mean_angular_error_violin_plot.png'
    if (recalculate or not os.path.exists(ang_violin_fp)) and angular_per_size.size:
        plt.figure(figsize=(10,7))
        plt.violinplot(angular_per_size.T, showmeans=True)
        plt.xlabel('Image Index')
        plt.xticks(ticks=range(1, len(img_labels)+1), labels=img_labels, rotation=45)
        plt.ylabel('Mean Angular Error')
        plt.title(f'Mean Angular Error per Image ({monitor_size}z)')
        plt.grid()
        plt.tight_layout()
        plt.savefig(ang_violin_fp)
        plt.close()

    return distances_per_size, angular_per_size


def plot_mean_per_monitor_size_per_trajectory(model, mode, distances_per_trajectory):
    """
    distances_per_trajectory: [traj, monitor_size, image, timestep]
    Saves mean and std (over images, timesteps) per monitor size and plots bars with error bars.
    """
    means_per_ms = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
    stds_per_ms = np.std(distances_per_trajectory, axis=(2, 3))    # [traj, ms]

    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]
        stds = stds_per_ms[i]
        traj_dir = f'{model}/{mode}/{trajectory}'
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


def plot_mean_per_trajectory(model, mode, distances_per_trajectory):
    """
    distances_per_trajectory shape: [num_trajectories, num_monitor_sizes, num_images, num_timesteps]
    Creates a violin plot per trajectory; saves mean and std per trajectory.
    """
    os.makedirs(f'{model}/{mode}', exist_ok=True)
    per_image_means = np.mean(distances_per_trajectory, axis=3)  # [traj, ms, img]
    per_image_means_fp = f'{model}/{mode}/per_image_mean_by_traj_ms_img.npy'
    mean_traj_fp = f'{model}/{mode}/mean_distance_per_trajectory.npy'
    std_traj_fp = f'{model}/{mode}/std_distance_per_trajectory.npy'
    violin_fp = f'{model}/{mode}/mean_distance_per_trajectory_violin.png'

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


def write_latex_table(model, mode, distances_per_trajectory, decimals=3):
    """
    distances_per_trajectory: [traj, monitor_size, image, timestep]
    Writes mean ± std over images for each (trajectory, monitor size) and saves both matrices.
    """
    os.makedirs(f'{model}/{mode}', exist_ok=True)
    mean_mat_fp = f'{model}/{mode}/mean_distance_matrix_traj_by_monitor.npy'
    std_mat_fp = f'{model}/{mode}/std_distance_matrix_traj_by_monitor.npy'
    table_fp = f'{model}/{mode}/mean_std_distance_table.tex'

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

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(outfile, 'w') as f:
        f.write(tex)


def gen_angular_errors(all_drone_poses, target_trajectory):
    return np.array([
        angular_error(all_drone_poses[j], target_trajectory[j])
        for j in range(len(target_trajectory))
    ])

def process_mode(model, mode, recalculate):
    try:
        gen_data(model, mode, recalculate=recalculate)

        distances_per_trajectory = []
        angular_errors_per_trajectory = []
        for trajectory in TRAJECTORIES:
            traj_dir = f'{model}/{mode}/{trajectory}'
            os.makedirs(traj_dir, exist_ok=True)
            cached_dist_fp = f'{traj_dir}/mean_distance_per_monitor_size.npy'
            cached_ang_fp  = f'{traj_dir}/angular_errors_per_monitor_size.npy'

            if (not recalculate and os.path.exists(cached_dist_fp) and os.path.exists(cached_ang_fp)):
                mean_per_monitor_size = np.load(cached_dist_fp, allow_pickle=False)
                angular_per_monitor_size = np.load(cached_ang_fp, allow_pickle=False)
            else:
                mean_per_monitor_size = []
                angular_per_monitor_size = []
                for monitor_size in MONITOR_SIZES:
                    dist_curves, ang_curves = gen_plot_per_monitor_size(
                        model, mode, trajectory, monitor_size, recalculate=recalculate
                    )
                    mean_per_monitor_size.append(dist_curves)
                    angular_per_monitor_size.append(ang_curves)
                mean_per_monitor_size = np.array(mean_per_monitor_size)       # [ms, img, timestep]
                angular_per_monitor_size = np.array(angular_per_monitor_size) # [ms, img, timestep]
                np.save(cached_dist_fp, mean_per_monitor_size)
                np.save(cached_ang_fp, angular_per_monitor_size)

            distances_per_trajectory.append(mean_per_monitor_size)
            angular_errors_per_trajectory.append(angular_per_monitor_size)

        distances_per_trajectory = np.array(distances_per_trajectory)        # [traj, ms, img, timestep]
        angular_errors_per_trajectory = np.array(angular_errors_per_trajectory)

        if distances_per_trajectory.size == 0:
            return model, mode, None, None, None, None

        mode_means = np.mean(distances_per_trajectory, axis=(2,3))
        mode_stds  = np.std(distances_per_trajectory, axis=(2,3))
        mode_angular_means = np.mean(angular_errors_per_trajectory, axis=(2,3))
        mode_angular_stds  = np.std(angular_errors_per_trajectory, axis=(2,3))

        # Distance violin
        mode_violin_fp = f'{model}/{mode}/all_trajectories_mean_distance_violin_plot.png'
        if not os.path.exists(mode_violin_fp):
            plt.figure(figsize=(15,10))
            y_min, y_max = np.inf, -np.inf
            for i in range(len(TRAJECTORIES)):
                data = np.mean(distances_per_trajectory[i], axis=1).T
                if data.size:
                    y_min = min(y_min, np.nanmin(data))
                    y_max = max(y_max, np.nanmax(data))
            for i, traj in enumerate(TRAJECTORIES):
                plt.subplot(2,3,i+1)
                data = np.mean(distances_per_trajectory[i], axis=1).T
                if data.size:
                    plt.violinplot(data, showmeans=True)
                plt.xlabel('Monitor Size')
                plt.xticks(ticks=range(1,len(MONITOR_SIZES)+1), labels=MONITOR_SIZES)
                plt.ylabel('Mean Distance')
                plt.title(traj)
                if y_min < y_max:
                    plt.ylim(y_min, y_max + 1.)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_violin_fp)
            plt.close()

        # Angular violin
        mode_ang_violin_fp = f'{model}/{mode}/all_trajectories_mean_angular_errors_violin_plot.png'
        if not os.path.exists(mode_ang_violin_fp):
            plt.figure(figsize=(15,10))
            a_min, a_max = np.inf, -np.inf
            for i in range(len(TRAJECTORIES)):
                ang = np.mean(angular_errors_per_trajectory[i], axis=1).T
                if ang.size:
                    a_min = min(a_min, np.nanmin(ang))
                    a_max = max(a_max, np.nanmax(ang))
            for i, traj in enumerate(TRAJECTORIES):
                plt.subplot(2,3,i+1)
                ang = np.mean(angular_errors_per_trajectory[i], axis=1).T
                if ang.size:
                    plt.violinplot(ang, showmeans=True)
                plt.xlabel('Monitor Size')
                plt.xticks(ticks=range(1,len(MONITOR_SIZES)+1), labels=MONITOR_SIZES)
                plt.ylabel('Mean Angular Error')
                plt.title(traj)
                if a_min < a_max:
                    plt.ylim(a_min, a_max + 1.)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_ang_violin_fp)
            plt.close()

        # Existing distance-only summaries
        plot_mean_per_monitor_size_per_trajectory(model, mode, distances_per_trajectory)
        plot_mean_per_trajectory(model, mode, distances_per_trajectory)
        write_latex_table(model, mode, distances_per_trajectory)

        return model, mode, mode_means, mode_stds, mode_angular_means, mode_angular_stds
    except Exception as e:
        print(f"Error processing model {model} mode {mode}: {e}")
        return model, mode, None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and generate plots for drone trajectories.")
    parser.add_argument("--recalculate", action="store_true", help="Force recalculation of all data and plots.")
    args = parser.parse_args()

    recalculate = args.recalculate

    for model in MODELS:

        all_mode_means = {}
        all_mode_stds = {}
        all_mode_angular_means = {}
        all_mode_angular_stds = {}

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_mode, model, mode, recalculate): mode for mode in MODES}
            for future in tqdm(as_completed(futures), total=len(futures)):
                mode = futures[future]
                try:
                    model, mode, mode_means, mode_stds, mode_angular_means, mode_angular_stds = future.result()
                    if mode_means is not None and mode_stds is not None:
                        all_mode_means[mode] = mode_means
                        all_mode_stds[mode] = mode_stds
                    if mode_angular_means is not None and mode_angular_stds is not None:
                        all_mode_angular_means[mode] = mode_angular_means
                        all_mode_angular_stds[mode] = mode_angular_stds
                except Exception as e:
                    print(f"Error in parallel processing for mode {mode}: {e}")

        write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile=f'{model}/mean_std_distance_all_modes_table.tex')
        write_all_modes_latex_table(all_mode_angular_means, all_mode_angular_stds, decimals=3, outfile=f'{model}/mean_std_angular_error_all_modes_table.tex')


