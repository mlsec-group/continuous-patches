import numpy as np
from tqdm import tqdm
import os
import concurrent.futures as cf
import multiprocessing as mp
import argparse

# Configure matplotlib for headless, worker-safe usage before pyplot import
os.environ.setdefault('MPLCONFIGDIR', '/home/piha/continous-patches/.mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from attack_minimal_single import gen_target_trajectory, normalize_yaw
import argparse

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


def gen_data(mode, recalculate=False):
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
                    if not recalculate and os.path.exists(distances_fp):
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

                    mean_distance_per_seed.append(mean_d)
                    std_distance_per_seed.append(std_d)

                # Aggregate across available seeds
                if len(mean_distance_per_seed) == 0:
                    print(f"[gen_data] No seeds available for {path}, skipping aggregation.")
                    continue

                agg_mean_fp = f'{path}/mean_distance_over_seeds.npy'
                agg_std_fp = f'{path}/std_distance_over_seeds.npy'
                if recalculate or not os.path.exists(agg_mean_fp):
                    np.save(agg_mean_fp, np.array(np.mean(mean_distance_per_seed)))
                if recalculate or not os.path.exists(agg_std_fp):
                    np.save(agg_std_fp, np.array(np.std(mean_distance_per_seed)))


def gen_plot_per_image(mode, trajectory, monitor_size, img_idx, recalculate=False):
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
    if recalculate or not os.path.exists(mean_over_seeds_per_timestep_fp) or not os.path.exists(std_over_seeds_per_timestep_fp):
        mean_over_seeds_per_timestep = all_distances.mean(axis=0)
        std_over_seeds_per_timestep = all_distances.std(axis=0)
        np.save(mean_over_seeds_per_timestep_fp, mean_over_seeds_per_timestep)
        np.save(std_over_seeds_per_timestep_fp, std_over_seeds_per_timestep)

    # Save per-seed mean/std over time if missing
    per_seed_mean_over_time_fp = f'{path}/per_seed_mean_over_time.npy'
    per_seed_std_over_time_fp = f'{path}/per_seed_std_over_time.npy'
    if recalculate or not os.path.exists(per_seed_mean_over_time_fp) or not os.path.exists(per_seed_std_over_time_fp):
        per_seed_mean_over_time = all_distances.mean(axis=1)
        per_seed_std_over_time = all_distances.std(axis=1)
        np.save(per_seed_mean_over_time_fp, per_seed_mean_over_time)
        np.save(per_seed_std_over_time_fp, per_seed_std_over_time)

    # Plot only if missing
    plot_fp = f'{path}/distance_plot.png'
    if recalculate or not os.path.exists(plot_fp):
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


def gen_plot_per_monitor_size(mode, trajectory, monitor_size, recalculate=False):
    distances_per_size = []
    img_labels = []
    for img_idx in IMG_IDX:
        distances = gen_plot_per_image(mode, trajectory, monitor_size, img_idx, recalculate=recalculate)  # [seed, timestep]
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
    if recalculate or not os.path.exists(per_timestep_mean_fp) or not os.path.exists(per_timestep_std_fp):
        if distances_per_size.size > 0:
            np.save(per_timestep_mean_fp, distances_per_size.mean(axis=0))
            np.save(per_timestep_std_fp, distances_per_size.std(axis=0))

    # Save per-image mean/std over time if missing
    per_image_mean_fp = f'{out_dir}/per_image_mean_over_time.npy'
    per_image_std_fp = f'{out_dir}/per_image_std_over_time.npy'
    if recalculate or not os.path.exists(per_image_mean_fp) or not os.path.exists(per_image_std_fp):
        if distances_per_size.size > 0:
            np.save(per_image_mean_fp, distances_per_size.mean(axis=1))
            np.save(per_image_std_fp, distances_per_size.std(axis=1))

    # Save overall mean/std if missing
    overall_mean_fp = f'{out_dir}/overall_mean.npy'
    overall_std_fp = f'{out_dir}/overall_std.npy'
    if recalculate or not os.path.exists(overall_mean_fp) or not os.path.exists(overall_std_fp):
        if distances_per_size.size > 0:
            np.save(overall_mean_fp, np.array(distances_per_size.mean()))
            np.save(overall_std_fp, np.array(distances_per_size.std()))

    # Violin plot of the per-image curves only if missing
    violin_fp = f'{out_dir}/mean_distance_violin_plot.png'
    if recalculate or not os.path.exists(violin_fp) and distances_per_size.size > 0:
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

        return dps, aps


class TrajectoryProcessor:
    def __init__(self, mode, trajectory):
        self.mode = mode
        self.trajectory = trajectory
        self.traj_dir = f'{mode}/{trajectory}'
        ensure_dir(self.traj_dir)

    def run(self, recalculate=False):
        target_trajectory = gen_target_trajectory(self.trajectory).detach().cpu().numpy()

        # Cached per-monitor-size arrays for this trajectory
        cached_dist_fp = f'{self.traj_dir}/mean_distance_per_monitor_size.npy'
        cached_ang_fp  = f'{self.traj_dir}/mean_angular_error_per_monitor_size.npy'

        need_compute = (not os.path.exists(cached_dist_fp)) or (not os.path.exists(cached_ang_fp))
        if not recalculate and not need_compute:
            dpt = np.load(cached_dist_fp, allow_pickle=False)
            apt = np.load(cached_ang_fp,  allow_pickle=False)
            return dpt, apt

        dpt_list, apt_list = [], []
        for ms in MONITOR_SIZES:
            msp = MonitorSizeProcessor(self.mode, self.trajectory, ms)
            dps, aps = msp.run(target_trajectory, recalculate=recalculate)  # [img, timestep] each
            dpt_list.append(dps)
            apt_list.append(aps)

        dpt = np.array(dpt_list)  # [ms, img, timestep]
        apt = np.array(apt_list)  # [ms, img, timestep]
        save_if_missing(cached_dist_fp, dpt, force=recalculate)
        save_if_missing(cached_ang_fp,  apt, force=recalculate)
        return dpt, apt


class ModeRunner:
    def __init__(self, mode, recalculate=False):
        self.mode = mode
        self.recalculate = recalculate

    def run(self):
        distances_per_trajectory = []
        angular_per_trajectory = []

        for trajectory in TRAJECTORIES:
            tp = TrajectoryProcessor(self.mode, trajectory)
            dpt, apt = tp.run(recalculate=self.recalculate)  # [ms, img, timestep] each
            distances_per_trajectory.append(dpt)
            angular_per_trajectory.append(apt)

        distances_per_trajectory = np.array(distances_per_trajectory)  # [traj, ms, img, timestep]
        angular_per_trajectory   = np.array(angular_per_trajectory)    # [traj, ms, img, timestep]

        # Per-mode summary matrices for tables
        mode_means = np.mean(distances_per_trajectory, axis=(2, 3))
        mode_stds  = np.std(distances_per_trajectory,  axis=(2, 3))
        ang_means  = np.mean(angular_per_trajectory,   axis=(2, 3))
        ang_stds   = np.std(angular_per_trajectory,    axis=(2, 3))

        # Per-mode, per-trajectory violins (distance)
        mode_violin_fp = f'{self.mode}/all_trajectories_mean_distance_violin_plot.png'
        if self.recalculate or not os.path.exists(mode_violin_fp):
            ensure_dir(self.mode)
            plt.figure(figsize=(15, 10))
            y_min, y_max = float('inf'), float('-inf')
            for i, _ in enumerate(TRAJECTORIES):
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
                plt.xticks(ticks=range(1, len(MONITOR_SIZES) + 1), labels=MONITOR_SIZES)
                plt.ylabel('Mean Euclidean Distance to Target')
                plt.title(f'{trajectory} Trajectory')
                if y_min < y_max:
                    plt.ylim(y_min, y_max + 1.0)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_violin_fp)
            plt.close()

        # Per-mode, per-trajectory violins (angular)
        mode_violin_ang_fp = f'{self.mode}/all_trajectories_mean_angular_error_violin_plot.png'
        if self.recalculate or not os.path.exists(mode_violin_ang_fp):
            ensure_dir(self.mode)
            plt.figure(figsize=(15, 10))
            y_min, y_max = float('inf'), float('-inf')
            for i, _ in enumerate(TRAJECTORIES):
                data = np.mean(angular_per_trajectory[i], axis=1).T
                if data.size > 0:
                    y_min = min(y_min, data.min())
                    y_max = max(y_max, data.max())
            for i, trajectory in enumerate(TRAJECTORIES):
                plt.subplot(2, 3, i+1)
                data = np.mean(angular_per_trajectory[i], axis=1).T
                if data.size > 0:
                    plt.violinplot(data, showmeans=True)
                plt.xlabel('Monitor Size')
                plt.xticks(ticks=range(1, len(MONITOR_SIZES) + 1), labels=MONITOR_SIZES)
                plt.ylabel('Angular Error (rad)')
                plt.title(f'{trajectory} Trajectory')
                if y_min < y_max:
                    plt.ylim(y_min, y_max + 0.1)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_violin_ang_fp)
            plt.close()

        # Per-trajectory bar plots and tables
        plot_mean_per_monitor_size_per_trajectory(self.mode, distances_per_trajectory, recalculate=self.recalculate)
        plot_mean_per_trajectory(self.mode, distances_per_trajectory, recalculate=self.recalculate)
        write_latex_table(self.mode, distances_per_trajectory, recalculate=self.recalculate)

        plot_mean_per_monitor_size_per_trajectory_angular(self.mode, angular_per_trajectory, recalculate=self.recalculate)
        plot_mean_per_trajectory_angular(self.mode, angular_per_trajectory, recalculate=self.recalculate)
        write_latex_table_angular(self.mode, angular_per_trajectory, recalculate=self.recalculate)

        return mode_means, mode_stds, ang_means, ang_stds


# -------------------- Summary plotting/table writers (reused) --------------------

def plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory, recalculate=False):
    means_per_ms = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
    stds_per_ms = np.std(distances_per_trajectory, axis=(2, 3))    # [traj, ms]

    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]
        stds = stds_per_ms[i]
        traj_dir = f'{mode}/{trajectory}'
        ensure_dir(traj_dir)

        means_fp = f'{traj_dir}/mean_distance_per_monitor_size_values.npy'
        stds_fp = f'{traj_dir}/std_distance_per_monitor_size_values.npy'
        save_if_missing(means_fp, means, force=recalculate)
        save_if_missing(stds_fp, stds, force=recalculate)

        plot_fp = f'{traj_dir}/mean_distance_per_monitor_size_bar.png'
        if recalculate or not os.path.exists(plot_fp):
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


def plot_mean_per_monitor_size_per_trajectory_angular(mode, angular_per_trajectory, recalculate=False):
    means_per_ms = np.mean(angular_per_trajectory, axis=(2, 3))
    stds_per_ms = np.std(angular_per_trajectory, axis=(2, 3))

    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]
        stds = stds_per_ms[i]
        traj_dir = f'{mode}/{trajectory}'
        ensure_dir(traj_dir)

        means_fp = f'{traj_dir}/mean_angular_error_per_monitor_size_values.npy'
        stds_fp = f'{traj_dir}/std_angular_error_per_monitor_size_values.npy'
        save_if_missing(means_fp, means, force=recalculate)
        save_if_missing(stds_fp, stds, force=recalculate)

        plot_fp = f'{traj_dir}/mean_angular_error_per_monitor_size_bar.png'
        if recalculate or not os.path.exists(plot_fp):
            plt.figure(figsize=(6, 4))
            plt.bar([f'{ms}z' for ms in MONITOR_SIZES], means, yerr=stds, capsize=4,
                    color='salmon', edgecolor='black')
            plt.xlabel('Monitor Size')
            plt.ylabel('Angular Error (rad)')
            plt.title(f'Mean Angular Error per Monitor Size\nTrajectory: {trajectory}')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_fp)
            plt.close()


def plot_mean_per_trajectory(mode, distances_per_trajectory, recalculate=False):
    ensure_dir(mode)
    per_image_means = np.mean(distances_per_trajectory, axis=3)  # [traj, ms, img]
    save_if_missing(f'{mode}/per_image_mean_by_traj_ms_img.npy', per_image_means, force=recalculate)

    data = [per_image_means[i].reshape(-1) for i in range(per_image_means.shape[0])]
    means_per_traj = np.array([d.mean() for d in data])
    stds_per_traj = np.array([d.std() for d in data])
    save_if_missing(f'{mode}/mean_distance_per_trajectory.npy', means_per_traj, force=recalculate)
    save_if_missing(f'{mode}/std_distance_per_trajectory.npy',  stds_per_traj,  force=recalculate)

    violin_fp = f'{mode}/mean_distance_per_trajectory_violin.png'
    if recalculate or not os.path.exists(violin_fp):
        plt.figure(figsize=(10, 5))
        plt.violinplot(data, showmeans=True, showextrema=True)
        plt.xticks(ticks=range(1, len(TRAJECTORIES) + 1), labels=TRAJECTORIES)
        plt.ylabel('Mean Euclidean Distance to Target')
        plt.title('Mean Distance per Trajectory (distribution across images and monitor sizes)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(violin_fp)
        plt.close()


def plot_mean_per_trajectory_angular(mode, angular_per_trajectory, recalculate=False):
    ensure_dir(mode)
    per_image_means = np.mean(angular_per_trajectory, axis=3)
    save_if_missing(f'{mode}/per_image_mean_angular_by_traj_ms_img.npy', per_image_means, force=recalculate)

    data = [per_image_means[i].reshape(-1) for i in range(per_image_means.shape[0])]
    means_per_traj = np.array([d.mean() for d in data])
    stds_per_traj = np.array([d.std() for d in data])
    save_if_missing(f'{mode}/mean_angular_error_per_trajectory.npy', means_per_traj, force=recalculate)
    save_if_missing(f'{mode}/std_angular_error_per_trajectory.npy',  stds_per_traj,  force=recalculate)

    violin_fp = f'{mode}/mean_angular_error_per_trajectory_violin.png'
    if recalculate or not os.path.exists(violin_fp):
        plt.figure(figsize=(10, 5))
        plt.violinplot(data, showmeans=True, showextrema=True)
        plt.xticks(ticks=range(1, len(TRAJECTORIES) + 1), labels=TRAJECTORIES)
        plt.ylabel('Angular Error (rad)')
        plt.title('Angular Error per Trajectory (distribution across images and monitor sizes)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(violin_fp)
        plt.close()


def write_latex_table(mode, distances_per_trajectory, decimals=3, recalculate=False):
    ensure_dir(mode)
    mean_mat_fp = f'{mode}/mean_distance_matrix_traj_by_monitor.npy'
    std_mat_fp = f'{mode}/std_distance_matrix_traj_by_monitor.npy'
    table_fp = f'{mode}/mean_std_distance_table.tex'

    means = load_or_none(mean_mat_fp, ignore_cache=recalculate)
    stds  = load_or_none(std_mat_fp,  ignore_cache=recalculate)
    if means is None or stds is None:
        means = np.mean(distances_per_trajectory, axis=(2, 3))
        stds  = np.std(distances_per_trajectory,  axis=(2, 3))
        save_if_missing(mean_mat_fp, means, force=recalculate)
        save_if_missing(std_mat_fp,  stds,  force=recalculate)

    if recalculate or not os.path.exists(table_fp):
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
        with open(table_fp, 'w') as f:
            f.write('\n'.join(lines))


def write_latex_table_angular(mode, angular_per_trajectory, decimals=3, recalculate=False):
    ensure_dir(mode)
    mean_mat_fp = f'{mode}/mean_angular_error_matrix_traj_by_monitor.npy'
    std_mat_fp = f'{mode}/std_angular_error_matrix_traj_by_monitor.npy'
    table_fp = f'{mode}/mean_std_angular_error_table.tex'

    means = load_or_none(mean_mat_fp, ignore_cache=recalculate)
    stds  = load_or_none(std_mat_fp,  ignore_cache=recalculate)
    if means is None or stds is None:
        means = np.mean(angular_per_trajectory, axis=(2, 3))
        stds  = np.std(angular_per_trajectory,  axis=(2, 3))
        save_if_missing(mean_mat_fp, means, force=recalculate)
        save_if_missing(std_mat_fp,  stds,  force=recalculate)

    if recalculate or not os.path.exists(table_fp):
        header_cols = ' & '.join(f'{ms}z' for ms in MONITOR_SIZES)
        safe_label = mode.replace('/', '_').replace(' ', '_')

        lines = []
        lines.append(r'\begin{table}[ht]')
        lines.append(r'\centering')
        lines.append(fr'\caption{{Mean $\pm$ std of angular error (rad) over images per monitor size and trajectory ({mode}).}}')
        lines.append(fr'\label{{tab:mean_std_angular_error_{safe_label}}}')
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
        with open(table_fp, 'w') as f:
            f.write('\n'.join(lines))


def write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_distance_all_modes_table.tex'):
    modes_in_order = [m for m in MODES if m in all_mode_means]
    if not modes_in_order:
        return

    num_value_cols = len(TRAJECTORIES) * len(MONITOR_SIZES)
    alignment = 'l' + 'c' * num_value_cols

    header_row_1 = ['Mode']
    for traj in TRAJECTORIES:
        header_row_1.append(fr'\multicolumn{{{len(MONITOR_SIZES)}}}{{c}}{{{traj}}}')
    header_row_1 = ' & '.join(header_row_1) + r' \\'

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
        means = all_mode_means[mode]
        stds = all_mode_stds[mode]
        row_cells = [mode]
        for ti in range(len(TRAJECTORIES)):
            for mj in range(len(MONITOR_SIZES)):
                row_cells.append(f'${means[ti, mj]:.{decimals}f} \\pm {stds[ti, mj]:.{decimals}f}$')
        lines.append(' & '.join(row_cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outfile, 'w') as f:
        f.write('\n'.join(lines))


def write_all_modes_latex_table_angular(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_angular_error_all_modes_table.tex'):
    modes_in_order = [m for m in MODES if m in all_mode_means]
    if not modes_in_order:
        return

    num_value_cols = len(TRAJECTORIES) * len(MONITOR_SIZES)
    alignment = 'l' + 'c' * num_value_cols

    header_row_1 = ['Mode']
    for traj in TRAJECTORIES:
        header_row_1.append(fr'\multicolumn{{{len(MONITOR_SIZES)}}}{{c}}{{{traj}}}')
    header_row_1 = ' & '.join(header_row_1) + r' \\'

    header_row_2 = ['']
    header_row_2.extend([f'{ms}z' for _ in TRAJECTORIES for ms in MONITOR_SIZES])
    header_row_2 = ' & '.join(header_row_2) + r' \\'

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Mean $\pm$ std of angular error (rad) over images per monitor size, grouped by trajectory, for all modes.}')
    lines.append(r'\label{tab:mean_std_angular_error_all_modes}')
    lines.append(r'\begin{tabular}{' + alignment + '}')
    lines.append(r'\hline')
    lines.append(header_row_1)
    lines.append(header_row_2)
    lines.append(r'\hline')

    for mode in modes_in_order:
        means = all_mode_means[mode]
        stds = all_mode_stds[mode]
        row_cells = [mode]
        for ti in range(len(TRAJECTORIES)):
            for mj in range(len(MONITOR_SIZES)):
                row_cells.append(f'${means[ti, mj]:.{decimals}f} \\pm {stds[ti, mj]:.{decimals}f}$')
        lines.append(' & '.join(row_cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outfile, 'w') as f:
        f.write('\n'.join(lines))


# -------------------- Parallel runner helper --------------------

def run_mode_capture(mode, recalculate=False):
    """
    Run a single mode and capture results/errors. Designed for ProcessPoolExecutor.
    Returns: (mode, mode_means, mode_stds, ang_means, ang_stds, error_or_None)
    """
    try:
        runner = ModeRunner(mode, recalculate=recalculate)
        mode_means, mode_stds, ang_means, ang_stds = runner.run()
        return (mode, mode_means, mode_stds, ang_means, ang_stds, None)
    except Exception as e:
        return (mode, None, None, None, None, str(e))


# -------------------- Main --------------------

def gen_angular_errors(all_drone_poses, target_trajectory):
    return np.array([
        angular_error(all_drone_poses[j], target_trajectory[j])
        for j in range(len(target_trajectory))
    ])

def process_mode(mode, recalculate):
    try:
        gen_data(mode, recalculate=recalculate)

        distances_per_trajectory = []
        angular_errors_per_trajectory = []
        for trajectory in TRAJECTORIES:
            traj_dir = f'{mode}/{trajectory}'
            os.makedirs(traj_dir, exist_ok=True)

            cached_ms_fp = f'{traj_dir}/mean_distance_per_monitor_size.npy'
            cached_angular_fp = f'{traj_dir}/mean_angular_error_per_monitor_size.npy'
            if not recalculate and os.path.exists(cached_ms_fp) and os.path.exists(cached_angular_fp):
                mean_per_monitor_size = np.load(cached_ms_fp, allow_pickle=False)
                mean_angular_per_monitor_size = np.load(cached_angular_fp, allow_pickle=False)
            else:
                mean_per_monitor_size = []
                mean_angular_per_monitor_size = []
                for monitor_size in MONITOR_SIZES:
                    distance_per_size = gen_plot_per_monitor_size(mode, trajectory, monitor_size, recalculate=recalculate)
                    mean_per_monitor_size.append(distance_per_size)

                    # Generate angular errors
                    angular_errors = []
                    for img_idx in IMG_IDX:
                        if img_idx == 'random':
                            path = f'{mode}/{trajectory}/{monitor_size}z/random'
                        else:
                            path = f'{mode}/{trajectory}/{monitor_size}z/image_{img_idx}'
                        poses_fp = f'{path}/all_drone_poses.npy'
                        if os.path.exists(poses_fp):
                            all_drone_poses = np.load(poses_fp)
                            target_trajectory = gen_target_trajectory(trajectory).detach().cpu().numpy()
                            angular_errors.append(gen_angular_errors(all_drone_poses, target_trajectory))
                    if angular_errors:
                        mean_angular_errors = np.mean(angular_errors, axis=0)
                        mean_angular_per_monitor_size.append(mean_angular_errors)

                mean_per_monitor_size = np.array(mean_per_monitor_size)
                mean_angular_per_monitor_size = np.array(mean_angular_per_monitor_size)
                np.save(cached_ms_fp, mean_per_monitor_size)
                np.save(cached_angular_fp, mean_angular_per_monitor_size)

            distances_per_trajectory.append(mean_per_monitor_size)
            angular_errors_per_trajectory.append(mean_angular_per_monitor_size)

        distances_per_trajectory = np.array(distances_per_trajectory)  # [traj, ms, img, timestep]
        angular_errors_per_trajectory = np.array(angular_errors_per_trajectory)  # [traj, ms, img, timestep]

        mode_means = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
        mode_stds = np.std(distances_per_trajectory, axis=(2, 3))   # [traj, ms]

        mode_angular_means = np.mean(angular_errors_per_trajectory, axis=(2, 3))  # [traj, ms]
        mode_angular_stds = np.std(angular_errors_per_trajectory, axis=(2, 3))   # [traj, ms]

        # Generate violin plots for distances and angular errors
        mode_violin_fp = f'{mode}/all_trajectories_mean_distance_violin_plot.png'
        mode_angular_violin_fp = f'{mode}/all_trajectories_mean_angular_errors_violin_plot.png'
        if not os.path.exists(mode_violin_fp) or not os.path.exists(mode_angular_violin_fp):
            os.makedirs(mode, exist_ok=True)
            plt.figure(figsize=(15, 10))
            y_min, y_max = float('inf'), float('-inf')
            angular_y_min, angular_y_max = float('inf'), float('-inf')
            for i, trajectory in enumerate(TRAJECTORIES):
                data = np.mean(distances_per_trajectory[i], axis=1).T
                angular_data = np.mean(angular_errors_per_trajectory[i], axis=1).T
                if data.size > 0:
                    y_min = min(y_min, data.min())
                    y_max = max(y_max, data.max())
                if angular_data.size > 0:
                    angular_y_min = min(angular_y_min, angular_data.min())
                    angular_y_max = max(angular_y_max, angular_data.max())
            for i, trajectory in enumerate(TRAJECTORIES):
                plt.subplot(2, 3, i+1)
                data = np.mean(distances_per_trajectory[i], axis=1).T
                if data.size > 0:
                    plt.violinplot(data, showmeans=True)
                plt.xlabel('Monitor Size')
                plt.xticks(ticks=range(1, len(MONITOR_SIZES) + 1), labels=MONITOR_SIZES)
                plt.ylabel('Mean Euclidean Distance to Target')
                plt.title(f'{trajectory} Trajectory')
                if y_min < y_max:
                    plt.ylim(y_min, y_max + 1.)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_violin_fp)
            plt.close()

            plt.figure(figsize=(15, 10))
            for i, trajectory in enumerate(TRAJECTORIES):
                plt.subplot(2, 3, i+1)
                angular_data = np.mean(angular_errors_per_trajectory[i], axis=1).T
                if angular_data.size > 0:
                    plt.violinplot(angular_data, showmeans=True)
                plt.xlabel('Monitor Size')
                plt.xticks(ticks=range(1, len(MONITOR_SIZES) + 1), labels=MONITOR_SIZES)
                plt.ylabel('Mean Angular Error')
                plt.title(f'{trajectory} Trajectory')
                if angular_y_min < angular_y_max:
                    plt.ylim(angular_y_min, angular_y_max + 1.)
                plt.grid()
            plt.tight_layout()
            plt.savefig(mode_angular_violin_fp)
            plt.close()

        plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory)
        plot_mean_per_trajectory(mode, distances_per_trajectory)
        write_latex_table(mode, distances_per_trajectory)

        return mode, mode_means, mode_stds, mode_angular_means, mode_angular_stds
    except Exception as e:
        print(f"Error processing mode {mode}: {e}")
        return mode, None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and generate plots for drone trajectories.")
    parser.add_argument("--recalculate", action="store_true", help="Force recalculation of all data and plots.")
    args = parser.parse_args()

    recalculate = args.recalculate

    all_mode_means = {}
    all_mode_stds = {}
    all_mode_angular_means = {}
    all_mode_angular_stds = {}

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_mode, mode, recalculate): mode for mode in MODES}
        for future in tqdm(as_completed(futures), total=len(futures)):
            mode = futures[future]
            try:
                mode, mode_means, mode_stds, mode_angular_means, mode_angular_stds = future.result()
                if mode_means is not None and mode_stds is not None:
                    all_mode_means[mode] = mode_means
                    all_mode_stds[mode] = mode_stds
                if mode_angular_means is not None and mode_angular_stds is not None:
                    all_mode_angular_means[mode] = mode_angular_means
                    all_mode_angular_stds[mode] = mode_angular_stds
            except Exception as e:
                print(f"Error in parallel processing for mode {mode}: {e}")

    write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_distance_all_modes_table.tex')
    write_all_modes_latex_table(all_mode_angular_means, all_mode_angular_stds, decimals=3, outfile='mean_std_angular_error_all_modes_table.tex')


