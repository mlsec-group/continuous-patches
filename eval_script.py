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
    ay = a[3] if np.ndim(a) and len(a) > 3 else float(a)
    by = b[3] if np.ndim(b) and len(b) > 3 else float(b)
    return np.abs(normalize_yaw(ay - by))


# -------------------- Small helpers --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_if_missing(fp, arr):
    if not os.path.exists(fp):
        np.save(fp, arr)
    return fp

def load_or_none(fp):
    return np.load(fp) if os.path.exists(fp) else None

def image_dir(mode, trajectory, monitor_size, img_idx):
    suffix = 'random' if img_idx == 'random' else f'image_{img_idx}'
    return f'{mode}/{trajectory}/{monitor_size}z/{suffix}'


# -------------------- Core processors --------------------

class SeedProcessor:
    def __init__(self, img_path, seed_idx):
        self.seed_dir = f'{img_path}/{seed_idx}'
        ensure_dir(self.seed_dir)
        self.poses_fp = f'{self.seed_dir}/all_drone_poses.npy'
        # Distance files
        self.dist_fp = f'{self.seed_dir}/distances.npy'
        self.dist_mean_fp = f'{self.seed_dir}/mean_distance.npy'
        self.dist_std_fp = f'{self.seed_dir}/std_distance.npy'
        # Angular files
        self.ang_fp = f'{self.seed_dir}/angular_errors.npy'
        self.ang_mean_fp = f'{self.seed_dir}/mean_angular_error.npy'
        self.ang_std_fp = f'{self.seed_dir}/std_angular_error.npy'

    def process(self, target_trajectory):
        # If needed, compute missing time-series from poses
        distances = load_or_none(self.dist_fp)
        angular = load_or_none(self.ang_fp)

        need_dist = distances is None
        need_ang = angular is None
        if (need_dist or need_ang):
            if not os.path.exists(self.poses_fp):
                # Can't compute anything for this seed
                return None
            poses = np.load(self.poses_fp)
            if need_dist:
                distances = np.array([euclidean_distance(poses[j], target_trajectory[j])
                                      for j in range(len(target_trajectory))])
                save_if_missing(self.dist_fp, distances)
            if need_ang:
                angular = np.array([angular_error(poses[j, 3], target_trajectory[j, 3])
                                    for j in range(len(target_trajectory))])
                save_if_missing(self.ang_fp, angular)

        # Seed-level stats (load or compute)
        mean_d = load_or_none(self.dist_mean_fp)
        if mean_d is None and distances is not None:
            mean_d = float(np.mean(distances))
            save_if_missing(self.dist_mean_fp, np.array(mean_d))

        std_d = load_or_none(self.dist_std_fp)
        if std_d is None and distances is not None:
            std_d = float(np.std(distances))
            save_if_missing(self.dist_std_fp, np.array(std_d))

        mean_a = load_or_none(self.ang_mean_fp)
        if mean_a is None and angular is not None:
            mean_a = float(np.mean(angular))
            save_if_missing(self.ang_mean_fp, np.array(mean_a))

        std_a = load_or_none(self.ang_std_fp)
        if std_a is None and angular is not None:
            std_a = float(np.std(angular))
            save_if_missing(self.ang_std_fp, np.array(std_a))

        if mean_d is None and mean_a is None:
            return None
        return dict(mean_distance=mean_d, std_distance=std_d, mean_angular=mean_a, std_angular=std_a)


class ImageProcessor:
    def __init__(self, mode, trajectory, monitor_size, img_idx):
        self.path = image_dir(mode, trajectory, monitor_size, img_idx)
        ensure_dir(self.path)

    def ensure_seeds(self, target_trajectory):
        agg_dist_means, agg_ang_means = [], []
        for seed in range(10):
            sp = SeedProcessor(self.path, seed)
            stats = sp.process(target_trajectory)
            if stats is None:
                continue
            if stats.get('mean_distance') is not None:
                agg_dist_means.append(stats['mean_distance'])
            if stats.get('mean_angular') is not None:
                agg_ang_means.append(stats['mean_angular'])

        # Aggregate over available seeds
        if agg_dist_means:
            save_if_missing(f'{self.path}/mean_distance_over_seeds.npy', np.array(np.mean(agg_dist_means)))
            save_if_missing(f'{self.path}/std_distance_over_seeds.npy',  np.array(np.std(agg_dist_means)))
        if agg_ang_means:
            save_if_missing(f'{self.path}/mean_angular_error_over_seeds.npy', np.array(np.mean(agg_ang_means)))
            save_if_missing(f'{self.path}/std_angular_error_over_seeds.npy',  np.array(np.std(agg_ang_means)))

    def load_seed_series(self):
        # Collect distances/angles across existing seeds
        dist_series, ang_series = [], []
        for seed in range(10):
            dist_fp = f'{self.path}/{seed}/distances.npy'
            ang_fp  = f'{self.path}/{seed}/angular_errors.npy'
            if os.path.exists(dist_fp):
                dist_series.append(np.load(dist_fp))
            if os.path.exists(ang_fp):
                ang_series.append(np.load(ang_fp))
        dists = np.array(dist_series) if dist_series else np.array([])
        angs  = np.array(ang_series) if ang_series else np.array([])
        return dists, angs

    def ensure_image_level_stats_and_plots(self, dists, angs):
        # Distances: per-timestep mean/std across seeds, per-seed over time, plot
        if dists.size > 0:
            save_if_missing(f'{self.path}/mean_over_seeds_per_timestep.npy', dists.mean(axis=0))
            save_if_missing(f'{self.path}/std_over_seeds_per_timestep.npy',  dists.std(axis=0))
            save_if_missing(f'{self.path}/per_seed_mean_over_time.npy', dists.mean(axis=1))
            save_if_missing(f'{self.path}/per_seed_std_over_time.npy',  dists.std(axis=1))
            plot_fp = f'{self.path}/distance_plot.png'
            if not os.path.exists(plot_fp):
                plt.figure(figsize=(10, 6))
                for i in range(dists.shape[0]):
                    plt.plot(dists[i], label=f'Seed {i}')
                plt.xlabel('Time Step')
                plt.ylabel('Euclidean Distance to Target')
                plt.title('Distance to Target Trajectory Over Time')
                plt.legend()
                plt.grid()
                plt.savefig(plot_fp)
                plt.close()

        # Angular: per-timestep mean/std across seeds, per-seed over time, plot
        if angs.size > 0:
            save_if_missing(f'{self.path}/mean_over_seeds_per_timestep_angular.npy', angs.mean(axis=0))
            save_if_missing(f'{self.path}/std_over_seeds_per_timestep_angular.npy',  angs.std(axis=0))
            save_if_missing(f'{self.path}/per_seed_mean_over_time_angular.npy', angs.mean(axis=1))
            save_if_missing(f'{self.path}/per_seed_std_over_time_angular.npy',  angs.std(axis=1))
            plot_fp = f'{self.path}/angular_error_plot.png'
            if not os.path.exists(plot_fp):
                plt.figure(figsize=(10, 6))
                for i in range(angs.shape[0]):
                    plt.plot(angs[i], label=f'Seed {i}')
                plt.xlabel('Time Step')
                plt.ylabel('Angular Error (rad)')
                plt.title('Angular Error to Target Yaw Over Time')
                plt.legend()
                plt.grid()
                plt.savefig(plot_fp)
                plt.close()

    def run(self, target_trajectory):
        self.ensure_seeds(target_trajectory)
        dists, angs = self.load_seed_series()
        if dists.size == 0 and angs.size == 0:
            print(f"[ImageProcessor] No distances/angular found at {self.path}.")
            return np.array([]), np.array([])
        self.ensure_image_level_stats_and_plots(dists, angs)
        return dists, angs


class MonitorSizeProcessor:
    def __init__(self, mode, trajectory, monitor_size):
        self.mode = mode
        self.trajectory = trajectory
        self.monitor_size = monitor_size
        self.out_dir = f'{mode}/{trajectory}/{monitor_size}z'
        ensure_dir(self.out_dir)

    def run(self, target_trajectory):
        distances_per_size, angular_per_size, img_labels = [], [], []
        for img_idx in IMG_IDX:
            ip = ImageProcessor(self.mode, self.trajectory, self.monitor_size, img_idx)
            dists, angs = ip.run(target_trajectory)
            if dists.size == 0 and angs.size == 0:
                continue
            if dists.size > 0:
                distances_per_size.append(dists.mean(axis=0))  # curve per image (mean over seeds)
            if angs.size > 0:
                angular_per_size.append(angs.mean(axis=0))
            img_labels.append(img_idx)

        dps = np.array(distances_per_size) if len(distances_per_size) > 0 else np.array([])
        aps = np.array(angular_per_size) if len(angular_per_size) > 0 else np.array([])

        # Distance summaries/plot
        if dps.size > 0:
            save_if_missing(f'{self.out_dir}/per_timestep_mean_over_images.npy', dps.mean(axis=0))
            save_if_missing(f'{self.out_dir}/per_timestep_std_over_images.npy',  dps.std(axis=0))
            save_if_missing(f'{self.out_dir}/per_image_mean_over_time.npy', dps.mean(axis=1))
            save_if_missing(f'{self.out_dir}/per_image_std_over_time.npy',  dps.std(axis=1))
            save_if_missing(f'{self.out_dir}/overall_mean.npy', np.array(dps.mean()))
            save_if_missing(f'{self.out_dir}/overall_std.npy',  np.array(dps.std()))
            violin_fp = f'{self.out_dir}/mean_distance_violin_plot.png'
            if not os.path.exists(violin_fp):
                plt.figure(figsize=(10, 7))
                plt.violinplot(dps.T, showmeans=True)
                plt.xlabel('Image Index')
                plt.xticks(ticks=range(1, len(img_labels) + 1), labels=img_labels, rotation=45)
                plt.ylabel('Mean Euclidean Distance to Target')
                plt.title(f'Mean Distance to Target Trajectory for Monitor Size {self.monitor_size}z')
                plt.grid()
                plt.tight_layout()
                plt.savefig(violin_fp)
                plt.close()

        # Angular summaries/plot
        if aps.size > 0:
            save_if_missing(f'{self.out_dir}/per_timestep_mean_over_images_angular.npy', aps.mean(axis=0))
            save_if_missing(f'{self.out_dir}/per_timestep_std_over_images_angular.npy',  aps.std(axis=0))
            save_if_missing(f'{self.out_dir}/per_image_mean_over_time_angular.npy', aps.mean(axis=1))
            save_if_missing(f'{self.out_dir}/per_image_std_over_time_angular.npy',  aps.std(axis=1))
            save_if_missing(f'{self.out_dir}/overall_mean_angular.npy', np.array(aps.mean()))
            save_if_missing(f'{self.out_dir}/overall_std_angular.npy',  np.array(aps.std()))
            violin_fp = f'{self.out_dir}/mean_angular_error_violin_plot.png'
            if not os.path.exists(violin_fp):
                plt.figure(figsize=(10, 7))
                plt.violinplot(aps.T, showmeans=True)
                plt.xlabel('Image Index')
                plt.xticks(ticks=range(1, len(img_labels) + 1), labels=img_labels, rotation=45)
                plt.ylabel('Angular Error (rad)')
                plt.title(f'Mean Angular Error for Monitor Size {self.monitor_size}z')
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

    def run(self):
        target_trajectory = gen_target_trajectory(self.trajectory).detach().cpu().numpy()

        # Cached per-monitor-size arrays for this trajectory
        cached_dist_fp = f'{self.traj_dir}/mean_distance_per_monitor_size.npy'
        cached_ang_fp  = f'{self.traj_dir}/mean_angular_error_per_monitor_size.npy'

        need_compute = (not os.path.exists(cached_dist_fp)) or (not os.path.exists(cached_ang_fp))
        if not need_compute:
            dpt = np.load(cached_dist_fp, allow_pickle=False)
            apt = np.load(cached_ang_fp,  allow_pickle=False)
            return dpt, apt

        dpt_list, apt_list = [], []
        for ms in MONITOR_SIZES:
            msp = MonitorSizeProcessor(self.mode, self.trajectory, ms)
            dps, aps = msp.run(target_trajectory)  # [img, timestep] each
            dpt_list.append(dps)
            apt_list.append(aps)

        dpt = np.array(dpt_list)  # [ms, img, timestep]
        apt = np.array(apt_list)  # [ms, img, timestep]
        save_if_missing(cached_dist_fp, dpt)
        save_if_missing(cached_ang_fp,  apt)
        return dpt, apt


class ModeRunner:
    def __init__(self, mode):
        self.mode = mode

    def run(self):
        distances_per_trajectory = []
        angular_per_trajectory = []

        for trajectory in TRAJECTORIES:
            tp = TrajectoryProcessor(self.mode, trajectory)
            dpt, apt = tp.run()  # [ms, img, timestep] each
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
        if not os.path.exists(mode_violin_fp):
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
        if not os.path.exists(mode_violin_ang_fp):
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
        plot_mean_per_monitor_size_per_trajectory(self.mode, distances_per_trajectory)
        plot_mean_per_trajectory(self.mode, distances_per_trajectory)
        write_latex_table(self.mode, distances_per_trajectory)

        plot_mean_per_monitor_size_per_trajectory_angular(self.mode, angular_per_trajectory)
        plot_mean_per_trajectory_angular(self.mode, angular_per_trajectory)
        write_latex_table_angular(self.mode, angular_per_trajectory)

        return mode_means, mode_stds, ang_means, ang_stds


# -------------------- Summary plotting/table writers (reused) --------------------

def plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory):
    means_per_ms = np.mean(distances_per_trajectory, axis=(2, 3))  # [traj, ms]
    stds_per_ms = np.std(distances_per_trajectory, axis=(2, 3))    # [traj, ms]

    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]
        stds = stds_per_ms[i]
        traj_dir = f'{mode}/{trajectory}'
        ensure_dir(traj_dir)

        means_fp = f'{traj_dir}/mean_distance_per_monitor_size_values.npy'
        stds_fp = f'{traj_dir}/std_distance_per_monitor_size_values.npy'
        save_if_missing(means_fp, means)
        save_if_missing(stds_fp, stds)

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


def plot_mean_per_monitor_size_per_trajectory_angular(mode, angular_per_trajectory):
    means_per_ms = np.mean(angular_per_trajectory, axis=(2, 3))
    stds_per_ms = np.std(angular_per_trajectory, axis=(2, 3))

    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]
        stds = stds_per_ms[i]
        traj_dir = f'{mode}/{trajectory}'
        ensure_dir(traj_dir)

        means_fp = f'{traj_dir}/mean_angular_error_per_monitor_size_values.npy'
        stds_fp = f'{traj_dir}/std_angular_error_per_monitor_size_values.npy'
        save_if_missing(means_fp, means)
        save_if_missing(stds_fp, stds)

        plot_fp = f'{traj_dir}/mean_angular_error_per_monitor_size_bar.png'
        if not os.path.exists(plot_fp):
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


def plot_mean_per_trajectory(mode, distances_per_trajectory):
    ensure_dir(mode)
    per_image_means = np.mean(distances_per_trajectory, axis=3)  # [traj, ms, img]
    save_if_missing(f'{mode}/per_image_mean_by_traj_ms_img.npy', per_image_means)

    data = [per_image_means[i].reshape(-1) for i in range(per_image_means.shape[0])]
    means_per_traj = np.array([d.mean() for d in data])
    stds_per_traj = np.array([d.std() for d in data])
    save_if_missing(f'{mode}/mean_distance_per_trajectory.npy', means_per_traj)
    save_if_missing(f'{mode}/std_distance_per_trajectory.npy',  stds_per_traj)

    violin_fp = f'{mode}/mean_distance_per_trajectory_violin.png'
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


def plot_mean_per_trajectory_angular(mode, angular_per_trajectory):
    ensure_dir(mode)
    per_image_means = np.mean(angular_per_trajectory, axis=3)
    save_if_missing(f'{mode}/per_image_mean_angular_by_traj_ms_img.npy', per_image_means)

    data = [per_image_means[i].reshape(-1) for i in range(per_image_means.shape[0])]
    means_per_traj = np.array([d.mean() for d in data])
    stds_per_traj = np.array([d.std() for d in data])
    save_if_missing(f'{mode}/mean_angular_error_per_trajectory.npy', means_per_traj)
    save_if_missing(f'{mode}/std_angular_error_per_trajectory.npy',  stds_per_traj)

    violin_fp = f'{mode}/mean_angular_error_per_trajectory_violin.png'
    if not os.path.exists(violin_fp):
        plt.figure(figsize=(10, 5))
        plt.violinplot(data, showmeans=True, showextrema=True)
        plt.xticks(ticks=range(1, len(TRAJECTORIES) + 1), labels=TRAJECTORIES)
        plt.ylabel('Angular Error (rad)')
        plt.title('Angular Error per Trajectory (distribution across images and monitor sizes)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(violin_fp)
        plt.close()


def write_latex_table(mode, distances_per_trajectory, decimals=3):
    ensure_dir(mode)
    mean_mat_fp = f'{mode}/mean_distance_matrix_traj_by_monitor.npy'
    std_mat_fp = f'{mode}/std_distance_matrix_traj_by_monitor.npy'
    table_fp = f'{mode}/mean_std_distance_table.tex'

    means = load_or_none(mean_mat_fp)
    stds  = load_or_none(std_mat_fp)
    if means is None or stds is None:
        means = np.mean(distances_per_trajectory, axis=(2, 3))
        stds  = np.std(distances_per_trajectory,  axis=(2, 3))
        save_if_missing(mean_mat_fp, means)
        save_if_missing(std_mat_fp,  stds)

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
        with open(table_fp, 'w') as f:
            f.write('\n'.join(lines))


def write_latex_table_angular(mode, angular_per_trajectory, decimals=3):
    ensure_dir(mode)
    mean_mat_fp = f'{mode}/mean_angular_error_matrix_traj_by_monitor.npy'
    std_mat_fp = f'{mode}/std_angular_error_matrix_traj_by_monitor.npy'
    table_fp = f'{mode}/mean_std_angular_error_table.tex'

    means = load_or_none(mean_mat_fp)
    stds  = load_or_none(std_mat_fp)
    if means is None or stds is None:
        means = np.mean(angular_per_trajectory, axis=(2, 3))
        stds  = np.std(angular_per_trajectory,  axis=(2, 3))
        save_if_missing(mean_mat_fp, means)
        save_if_missing(std_mat_fp,  stds)

    if not os.path.exists(table_fp):
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


# -------------------- Main --------------------

if __name__ == "__main__":
    all_mode_means = {}
    all_mode_stds = {}
    all_mode_ang_means = {}
    all_mode_ang_stds = {}

    for mode in tqdm(MODES):
        try:
            runner = ModeRunner(mode)
            mode_means, mode_stds, ang_means, ang_stds = runner.run()
        except Exception as e:
            print(f"Error processing mode {mode}: {e}")
            continue

        all_mode_means[mode] = mode_means
        all_mode_stds[mode] = mode_stds
        all_mode_ang_means[mode] = ang_means
        all_mode_ang_stds[mode] = ang_stds

    write_all_modes_latex_table(all_mode_means, all_mode_stds, decimals=3, outfile='mean_std_distance_all_modes_table.tex')
    write_all_modes_latex_table_angular(all_mode_ang_means, all_mode_ang_stds, decimals=3, outfile='mean_std_angular_error_all_modes_table.tex')


