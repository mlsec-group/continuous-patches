import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from attack_minimal_single import gen_target_trajectory, normalize_yaw


MONITOR_SIZES = [30, 60, 90, 120]
IMG_IDX = [505, 4847, 3059, 1860, 3205, 4861, 2613, 2309, 5431, 2847, 'random']
TRAJECTORIES = ["figure8", "square", "circle", "line_y", "line_x"]


def euclidean_distance(a, b):
    distance = np.linalg.norm(a[:3] - b[:3])
    return distance

def angular_error(a, b):
    angular_loss = 1 - np.cos(normalize_yaw(a[3])) - normalize_yaw((b[3]))
    return angular_loss


def gen_data():

    for trajectory in tqdm(TRAJECTORIES):

        target_trajectory = gen_target_trajectory(trajectory).detach().cpu().numpy()

        for monitor_size in MONITOR_SIZES:
            for img_idx in IMG_IDX:
                if img_idx == 'random':
                    path = f'{trajectory}/{monitor_size}z/random'
                else:
                    path = f'{trajectory}/{monitor_size}z/image_{img_idx}'

                all_drone_poses_p_seed = [np.load(f'{path}/{i}/all_drone_poses.npy') for i in range(10)]
                all_drone_poses_p_seed = np.array(all_drone_poses_p_seed)

                # print(all_drone_poses_p_seed.shape)

                mean_distance_per_seed = []
                std_distance_per_seed = []
                for i in range(10):
                    # print("all_drone_poses_p_seed[i].shape:", all_drone_poses_p_seed[i].shape)
                    # print("all_drone_poses_p_seed[i]:", all_drone_poses_p_seed[i])
                    distances = [euclidean_distance(all_drone_poses_p_seed[i][j], target_trajectory[j]) for j in range(len(target_trajectory))]
                    mean_distance_per_seed.append(np.mean(distances))
                    std_distance_per_seed.append(np.std(distances))
                    # print(f'Seed {i}: Mean Distance: {mean_distance_per_seed[-1]}, Std Distance: {std_distance_per_seed[-1]}')
                    np.save(f'{path}/{i}/distances.npy', np.array(distances))
                    np.save(f'{path}/{i}/mean_distance.npy', np.array(mean_distance_per_seed[-1]))
                    np.save(f'{path}/{i}/std_distance.npy', np.array(std_distance_per_seed[-1]))

                mean_for_idx = np.mean(mean_distance_per_seed)
                std_for_idx = np.std(mean_distance_per_seed)
                # print(f'Overall Mean Distance: {mean_for_idx}, Overall Std Distance: {std_for_idx}')
                np.save(f'{path}/mean_distance_over_seeds.npy', np.array(mean_for_idx))
                np.save(f'{path}/std_distance_over_seeds.npy', np.array(std_for_idx))


def gen_plot_per_image(trajectory, monitor_size, img_idx):
    if img_idx == 'random':
        path = f'{trajectory}/{monitor_size}z/random'
    else:
        path = f'{trajectory}/{monitor_size}z/image_{img_idx}'
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

def gen_plot_per_monitor_size(trajectory, monitor_size):
    distances_per_size = []
    for img_idx in IMG_IDX:
        distances = gen_plot_per_image(trajectory, monitor_size, img_idx)
        distances_per_size.append(np.mean(distances, axis=0))

    distances_per_size = np.array(distances_per_size)
    # print("Shape of distances_per_size:", distances_per_size.shape)

    # Violin plot of the mean distances over images
    plt.figure(figsize=(10, 7))
    plt.violinplot(distances_per_size.T, showmeans=True)
    plt.xlabel('Image Index')
    plt.xticks(ticks=range(1, len(IMG_IDX) + 1), 
               labels=IMG_IDX, rotation=45)
    plt.ylabel('Mean Euclidean Distance to Target')
    plt.title(f'Mean Distance to Target Trajectory for Monitor Size {monitor_size}z')
    plt.grid()
    plt.savefig(f'{trajectory}/{monitor_size}z/mean_distance_violin_plot.png')
    plt.close()
    return distances_per_size


if __name__ == "__main__":
    # gen_data()

    # gen_plot_per_image()

    distances_per_trajectory = []
    for trajectory in TRAJECTORIES:

        mean_per_monitor_size = []
        for monitor_size in MONITOR_SIZES:
            print("Processing trajectory:", trajectory, "Monitor size:", monitor_size)
            distance_per_size = gen_plot_per_monitor_size(trajectory, monitor_size)
            mean_per_monitor_size.append(distance_per_size)
        mean_per_monitor_size = np.array(mean_per_monitor_size)
        np.save(f'{trajectory}/mean_distance_per_monitor_size.npy', mean_per_monitor_size)
        distances_per_trajectory.append(mean_per_monitor_size)

    distances_per_trajectory = np.array(distances_per_trajectory) # [5, 4, 11, 20] -> trajectories, monitor sizes, img indices, time steps

    #create the violin plot for each trajectory
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
        plt.ylim(y_min, y_max+1.)  # Set the same y-axis limits for all subplots
        plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'all_trajectories_mean_distance_violin_plot.png')
    plt.close()

    # print(distances_per_trajectory.shape)

    # # Violin plot of the mean distances over trajectories
    # plt.figure(figsize=(10, 7))
    # plt.violinplot(np.mean(distances_per_trajectory, axis=1).T, showmeans=True)
    # plt.xlabel('Trajectory')
    # plt.xticks(ticks=range(1, len(TRAJECTORIES) + 1), 
    #            labels=TRAJECTORIES, rotation=45)
    # plt.ylabel('Mean Euclidean Distance to Target')
    # plt.title(f'Mean Distance to Target Trajectory for Different Trajectories')
    # plt.grid()
    # plt.savefig(f'all_trajectories_mean_distance_violin_plot.png')
    # plt.close()


