import numpy as np
from flying.cf_control import CrazyflieControl, custom_sleep


from util import get_yaw, project_patch, PatchDisplayThread, PoseUpdater

import cv2

from time import sleep

import yaml

import os
from pathlib import Path


def move_patch(T, delta_sf=0.0, delta_tx=0.0, delta_ty=0.0):
    T[0,0] += delta_sf
    T[1,1] += delta_sf
    T[0,2] += delta_tx
    T[1,2] += delta_ty

    return T

if __name__ == "__main__":

    with open('flying/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    results_dir = Path("results/test_y/")
    os.makedirs(results_dir, exist_ok=True)

    projector_display_size = (1050, 1680)

    background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

    # Start the PatchDisplayThread
    display_thread = PatchDisplayThread("Patch", (2561, 0))
    display_thread.start()
    display_thread.update(background)

    # load patch
    patch = cv2.imread("data/frontnet_gt_patch_x.jpg")

    T = np.zeros((3,3))
    T[0,0] = 5 # sf
    T[1,1] = 5 # sf
    T[0,2] = 750 # tx
    T[1,2] = 300 # ty
    T[2,2] = 1

    print(T)

    projected_patch = project_patch(patch, T, background)

    display_thread.update(projected_patch)


    cf = CrazyflieControl(config)

    bat_v, bat_s = cf.battery

    while bat_v is None:
        bat_v, bat_s = cf.battery

    print(f"Battery voltage: {bat_v}, Battery state: {bat_s}")

    if bat_v <= 3900: 
        print('Battery below 3.9V!! Not flying.')
        cf.close()
        display_thread.close()
        exit()


    poses = PoseUpdater(cf.pose)
    all_poses = []

    cf.takeoff(1.0, 3.)
    custom_sleep(3., cf.occupied, True)

    pose, timestamp = poses.get_current_pose()
    all_poses.append((timestamp, *pose))

    cf.toggle_frontnet()
    custom_sleep(2., cf.occupied, True)

    pose, timestamp = poses.get_current_pose()
    all_poses.append((timestamp, *pose))

    # move the patch all the way to the left
    for i in range(750):
        T = move_patch(T, delta_tx=-1)
        projected_patch = project_patch(patch, T, background)
        display_thread.update(projected_patch)
        pose, timestamp = poses.get_current_pose()
        all_poses.append((timestamp, *pose))
        sleep(0.01)

    # move the patch all the way to the right
    patch_size = patch.shape[0] * T[0,0]
    right_edge = int(projector_display_size[1] - patch_size)

    for i in range(right_edge):
        T = move_patch(T, delta_tx=1)
        projected_patch = project_patch(patch, T, background)
        display_thread.update(projected_patch)
        pose, timestamp = poses.get_current_pose()
        all_poses.append((timestamp, *pose))
        sleep(0.01)

    # for i in range(30):
    #     T = move_patch(T, delta_sf=0.1)
    #     projected_patch = project_patch(patch, T, background)
    #     display_thread.update(projected_patch)
    #     pose, timestamp = poses.get_current_pose()
    #     all_poses.append((timestamp, *pose))
    #     sleep(0.1)

    # for i in range(50):
    #     T = move_patch(T, delta_sf=-0.1)
    #     projected_patch = project_patch(patch, T, background)
    #     display_thread.update(projected_patch)
    #     pose, timestamp = poses.get_current_pose()
    #     all_poses.append((timestamp, *pose))
    #     sleep(0.1)

    pose, timestamp = poses.get_current_pose()
    all_poses.append((timestamp, *pose))

    np.save(str(results_dir/ "poses.npy"), np.array(all_poses))

    cf.toggle_frontnet()
    custom_sleep(1., cf.occupied, True)

    # while True:
    #     if display_thread._stay_alive == False:
    #         break

    cf.land()
    custom_sleep(5., cf.occupied, True)
    cf.close()
    display_thread.close()
    poses.close()