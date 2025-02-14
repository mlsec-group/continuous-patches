import numpy as np
from flying.cf_control import CrazyflieControl, custom_sleep


from util import get_yaw, project_patch, PatchDisplayThread, PoseUpdater

import cv2

from time import sleep

import yaml

import os
from pathlib import Path

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
        


if __name__ == "__main__":


    with open('flying/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    # load best patches + transformations for each direction
    results_dir = Path("results/bayes")

    patches_positions = {'left': {'patch': None, 'T': None, 'loss': None},
                         'right': {'patch': None, 'T': None, 'loss': None},
                         'forward': {'patch': None, 'T': None, 'loss': None},
                         'backward': {'patch': None, 'T': None, 'loss': None}}
    
    for direction in patches_positions.keys():
        patch, T, loss = load_patch_for_direction(results_dir, direction)
        patches_positions[direction]['patch'] = patch
        patches_positions[direction]['T'] = T
        patches_positions[direction]['loss'] = loss

    # print(patches_positions)

    projector_display_size = (1050, 1680)

    background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

    # Start the PatchDisplayThread
    display_thread = PatchDisplayThread("Patch", (2561, 0))
    display_thread.start()
    display_thread.update(background)


    # sanity check
    # for direction in patches_positions.keys():
    #     print(direction)
    #     print(patches_positions[direction]['T'])
    #     print(patches_positions[direction]['loss'])
    #     print(patches_positions[direction]['patch'].shape)
    #     projected_patch = project_patch(patches_positions[direction]['patch'], patches_positions[direction]['T'], background)
    #     display_thread.update(projected_patch)
    #     sleep(3)

    # cf = CrazyflieControl(config)