import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from attack_minimal_single import gen_target_trajectory, normalize_yaw

MODES = ['optimal/warm', 'timeout_10Hz/cold', 
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
    all_distances = np.array(all_distances)

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
        distances = gen_plot_per_image(mode, trajectory, monitor_size, img_idx)
        distances_per_size.append(np.mean(distances, axis=0))

    distances_per_size = np.array(distances_per_size)

    # Violin plot of the mean distances over images
    out_dir = f'{mode}/{trajectory}/{monitor_size}z'
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.violinplot(distances_per_size.T, showmeans=True)
    plt.xlabel('Image Index')
    plt.xticks(ticks=range(1, len(IMG_IDX) + 1), 
               labels=IMG_IDX, rotation=45)
    plt.ylabel('Mean Euclidean Distance to Target')
    plt.title(f'Mean Distance to Target Trajectory for Monitor Size {monitor_size}z')
    plt.grid()
    plt.savefig(f'{out_dir}/mean_distance_violin_plot.png')
    plt.close()
    return distances_per_size


def plot_mean_per_monitor_size_per_trajectory(mode, distances_per_trajectory):
    """
    distances_per_trajectory shape: [num_trajectories, num_monitor_sizes, num_images, num_timesteps]
    Produces a bar chart per trajectory with mean distance per monitor size (averaged over images and time).
    Saves figures and values per trajectory directory under the mode root.
    """
    means_per_ms = np.mean(distances_per_trajectory, axis=(2, 3))  # -> [traj, ms]
    for i, trajectory in enumerate(TRAJECTORIES):
        means = means_per_ms[i]  # [ms]
        traj_dir = f'{mode}/{trajectory}'
        os.makedirs(traj_dir, exist_ok=True)
        # Save values
        np.save(f'{traj_dir}/mean_distance_per_monitor_size_values.npy', means)
        # Plot
        plt.figure(figsize=(6, 4))
        plt.bar([f'{ms}z' for ms in MONITOR_SIZES], means, color='skyblue', edgecolor='black')
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
    Produces a single bar chart with mean distance per trajectory (averaged over monitor sizes, images, and time).
    Saves figure and values under the mode root.
    """
    means_per_traj = np.mean(distances_per_trajectory, axis=(1, 2, 3))  # -> [traj]
    os.makedirs(mode, exist_ok=True)
    np.save(f'{mode}/mean_distance_per_trajectory.npy', means_per_traj)
    plt.figure(figsize=(8, 4))
    plt.bar(TRAJECTORIES, means_per_traj, color='salmon', edgecolor='black')
    plt.xlabel('Trajectory')
    plt.ylabel('Mean Euclidean Distance to Target')
    plt.title('Mean Distance per Trajectory')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{mode}/mean_distance_per_trajectory_bar.png')
    plt.close()

def write_latex_table(mode, distances_per_trajectory, decimals=3):
    """
    distances_per_trajectory shape: [traj, monitor_size, image, timestep]
    Creates a LaTeX table of mean distances averaged over all images and timesteps
    for each (trajectory, monitor_size). One table per mode.
    """
    os.makedirs(mode, exist_ok=True)
    means = np.mean(distances_per_trajectory, axis=(2, 3))  # -> [traj, monitor_size]
    # Optional: also save the raw matrix
    np.save(f'{mode}/mean_distance_matrix_traj_by_monitor.npy', means)

    header_cols = ' & '.join(f'{ms}z' for ms in MONITOR_SIZES)
    safe_label = mode.replace('/', '_').replace(' ', '_')

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(fr'\caption{{Mean distance over all images per monitor size and trajectory ({mode})}}')
    lines.append(fr'\label{{tab:mean_distance_{safe_label}}}')
    lines.append(r'\begin{tabular}{l' + 'c' * len(MONITOR_SIZES) + '}')
    lines.append(r'\hline')
    lines.append(r'Trajectory & ' + header_cols + r' \\')
    lines.append(r'\hline')

    for i, traj in enumerate(TRAJECTORIES):
        row_vals = ' & '.join(f'{means[i, j]:.{decimals}f}' for j in range(len(MONITOR_SIZES)))
        lines.append(f'{traj} & {row_vals} \\\\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines)

    with open(f'{mode}/mean_distance_table.tex', 'w') as f:
        f.write(tex)

if __name__ == "__main__":
    # Set mode root (edit this line or wire up argparse)
    
    for mode in MODES:

        try:
            gen_data(mode)
        except Exception as e:
            print(f"Error generating data for mode {mode}: {e}")
            continue

        distances_per_trajectory = []
        for trajectory in TRAJECTORIES:
            mean_per_monitor_size = []
            for monitor_size in MONITOR_SIZES:
                print("Processing trajectory:", trajectory, "Monitor size:", monitor_size)
                distance_per_size = gen_plot_per_monitor_size(mode, trajectory, monitor_size)
                mean_per_monitor_size.append(distance_per_size)
            mean_per_monitor_size = np.array(mean_per_monitor_size)
            traj_dir = f'{mode}/{trajectory}'
            os.makedirs(traj_dir, exist_ok=True)
            np.save(f'{traj_dir}/mean_distance_per_monitor_size.npy', mean_per_monitor_size)
            distances_per_trajectory.append(mean_per_monitor_size)

        distances_per_trajectory = np.array(distances_per_trajectory) # [5, 4, 11, T]

        # Create the violin plot for each trajectory (under mode root)
        os.makedirs(mode, exist_ok=True)
        plt.figure(figsize=(15, 10))
        y_min, y_max = float('inf'), float('-inf')
        
        # Determine the global y-axis limits
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


