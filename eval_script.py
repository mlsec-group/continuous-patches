import numpy as np

from attack_minimal_single import gen_target_trajectory, normalize_yaw


def euclidean_distance(a, b):
    distance = np.linalg.norm(a[:3] - b[:3])
    return distance

def angular_error(a, b):
    angular_loss = 1 - np.cos(normalize_yaw(a[3])) - normalize_yaw((b[3]))
    return angular_loss



target_trajectory = gen_target_trajectory('figure8').detach().cpu().numpy()

path = 'figure8/30z/image_505'

all_drone_poses_p_seed = [np.load(f'{path}/{i}/all_drone_poses.npy') for i in range(10)]
all_drone_poses_p_seed = np.array(all_drone_poses_p_seed)

print(all_drone_poses_p_seed.shape)

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

# mse_per_seed = [compute_mse(all_drone_poses_p_seed[i], target_trajectory) for i in range(10)]
# mean_mse = np.mean(mse_per_seed)
# std_mse = np.std(mse_per_seed)
# print(f'Mean MSE over seeds: {mean_mse}, Std MSE over seeds: {std_mse}')