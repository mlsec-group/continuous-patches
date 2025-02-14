import numpy as np
from flying.cf_control import CrazyflieControl, custom_sleep


from util import get_yaw, project_patch, PatchDisplayThread, PoseUpdater

import cv2

from time import sleep

import yaml

import os
from pathlib import Path


LEFT = np.array([1.0, 0.7, 1.0, 0.0])
RIGHT = np.array([1.0, -1.3, 1.0, 0.0])
FORWARD = np.array([1.5, -0.3, 1.0, 0.0]) 
BACKWARD = np.array([0.0, -0.3, 1.0, 0.0])

def load_patch_for_direction(path, direction, index=None):
    folder = Path(path / direction)

    if index is not None:
        folder_idx = folder / str(index)
        best_patch = cv2.imread(folder_idx / "best_patch.jpg")
        with open(folder_idx / "results.yaml") as file:
            results = yaml.load(file, Loader=yaml.FullLoader)

        loss = results["target"]

        T = np.zeros((3,3))
        T[0,0] = results["params"]["sf"]
        T[1,1] = results["params"]["sf"]
        T[0,2] = results["params"]["tx"]
        T[1,2] = results["params"]["ty"]
        T[2,2] = 1


        return best_patch, T, loss

    else:
        all_results = {}
        for i in range(3):
            folder_i = folder / str(i)
            with open(folder_i / "results.yaml") as file:
                results = yaml.load(file, Loader=yaml.FullLoader)
            all_results[i] = results

        # get index of result with hihest target value
        target_values = [r["target"] for r in all_results.values()]
        best_idx = np.argmax(target_values)
        
        best_patch = cv2.imread(folder / str(best_idx) / "best_patch.jpg")
        
        T = np.zeros((3,3))
        T[0,0] = all_results[best_idx]["params"]["sf"]
        T[1,1] = all_results[best_idx]["params"]["sf"]
        T[0,2] = all_results[best_idx]["params"]["tx"]
        T[1,2] = all_results[best_idx]["params"]["ty"]
        T[2,2] = 1
        
        return best_patch, T, target_values[best_idx]
        
def patch_decision(current_pose, target_pose, patches_dict):
    
    distances = target_pose - current_pose

    most_change = np.argmax(distances)
    sign = np.sign(distances[most_change])

    match most_change:
        case 0:
            if sign > 0:
                direction = 'forward'
            else:
                direction = 'backward'
        case 1:
            if sign > 0:
                direction = 'left'
            else:
                direction = 'right'

    return patches_dict[direction]['patch'], patches_dict[direction]['T']


def attack(target_trajectory, drone_config, patches_dict):

    display_thread = PatchDisplayThread("Patch", (2561, 0))
    projector_display_size = (1050, 1680)

    background = np.zeros((*projector_display_size, 3), dtype=np.uint8)
    display_thread.start()
    display_thread.update(background)

    drone = CrazyflieControl(drone_config)
    pose_getter = PoseUpdater(drone.pose)

    bat_v, bat_s = drone.battery

    while bat_v is None:
        bat_v, bat_s = drone.battery

    print(f"Battery voltage: {bat_v}, Battery state: {bat_s}")

    if bat_v <= 3900: 
        print('Battery below 3.9V!! Not flying.')
        drone.close()
        display_thread.close()
        return None

    drone.takeoff(1.0, 3.)
    custom_sleep(3., drone.occupied, True)

    drone.reset()
    custom_sleep(5., drone.occupied, True)

    

    all_poses = []

    current_pose, current_time = pose_getter.get_current_pose()
    all_poses.append((current_time, *current_pose))

    patch, T = patch_decision(current_pose[:2], target_trajectory[0][:2], patches_dict)

    projected_patch = project_patch(patch, T, background)
    display_thread.update(projected_patch)
    sleep(1.0)


    trajectory_idx = 0
    max_idx = len(target_trajectory) + 1

    drone.toggle_frontnet()

    try:

        while trajectory_idx < max_idx:
            bat_v, bat_s = drone.battery
            if bat_s == 3:
                print("Low battery!! Landing...")
                drone.toggle_frontnet()
                custom_sleep(2., drone.occupied, True)
                drone.land()
                custom_sleep(2., drone.occupied, True)
                pose_getter.close()
                drone.power_off()
                drone.close()

                print("Perform battery change, place drone where it was, and hit y if ready!")
                while True:
                    if input('Ready? ') == 'y':
                        break

                del drone
                del pose_getter
                drone = CrazyflieControl(config)
                pose_getter = PoseUpdater(drone.pose)

                while True:
                    if drone.connected:
                        break
                print("Continue flying...")
                drone.takeoff(1.0, 3)
                custom_sleep(2., drone.occupied, True)

            target_pose = target_trajectory[trajectory_idx]
            current_pose, current_time = pose_getter.get_current_pose()
            all_poses.append((current_time, *current_pose))

            patch, T = patch_decision(current_pose[:2], target_pose[:2], patches_dict)

            projected_patch = project_patch(patch, T, background)
            display_thread.update(projected_patch)

            sleep(0.2)
            current_pose, _ = pose_getter.get_current_pose()
            if np.linalg.norm(target_pose[:2] - current_pose[:2]) < 0.3:
                print("Reached target position!")
                trajectory_idx += 1
        
    except KeyboardInterrupt:
        print("Interrupted by user!")
        # drone.toggle_frontnet()
        # custom_sleep(2., drone.occupied, True)
        # drone.land()
        # custom_sleep(2., drone.occupied, True)
        # pose_getter.close()
        # drone.power_off()
        # drone.close()
        # display_thread.close()
        # return np.array(all_poses)

    current_pose, current_time = pose_getter.get_current_pose()
    all_poses.append((current_time, *current_pose))

    drone.toggle_frontnet()
    custom_sleep(1.5, drone.occupied, True)

    drone.land()
    custom_sleep(2., drone.occupied, True)

    pose_getter.close()
    drone.power_off()
    drone.close()
    display_thread.close()

    return np.array(all_poses)



if __name__ == "__main__":


    with open('flying/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    # load best patches + transformations for each direction
    results_dir = Path("results/random")

    patches_positions = {'left': {'patch': None, 'T': None, 'loss': None},
                         'right': {'patch': None, 'T': None, 'loss': None},
                         'forward': {'patch': None, 'T': None, 'loss': None},
                         'backward': {'patch': None, 'T': None, 'loss': None}}
    
    for direction in patches_positions.keys():
        patch, T, loss = load_patch_for_direction(results_dir, direction)
        patches_positions[direction]['patch'] = patch
        patches_positions[direction]['T'] = T
        patches_positions[direction]['loss'] = loss


    target_trajectory = [ [0.0, -0.3, 1.0, 0.0],
                          [0.3, -0.3, 1.0, 0.0],
                          [0.5, -0.3, 1.0, 0.0],
                          [0.7, -0.3, 1.0, 0.0],
                          [0.7, 0.0, 1.0, 0.0],
                          [0.7, 0.3, 1.0, 0.0],
                          [0.7, 0.0, 1.0, 0.0],
                          [0.7, -0.3, 1.0, 0.0],
                          [0.7, -0.6, 1.0, 0.0],
                          [0.7, -0.3, 1.0, 0.0],
                          [0.5, -0.3, 1.0, 0.0],
                          [0.3, -0.3, 1.0, 0.0],]


    resulting_trajectory = attack(target_trajectory, config, patches_positions)




    # sanity check
    # for direction in patches_positions.keys():
    #     print(direction)
    #     print(patches_positions[direction]['T'])
    #     print(patches_positions[direction]['loss'])
    #     print(patches_positions[direction]['patch'].shape)
    #     projected_patch = project_patch(patches_positions[direction]['patch'], patches_positions[direction]['T'], background)
    #     display_thread.update(projected_patch)
    #     sleep(3)