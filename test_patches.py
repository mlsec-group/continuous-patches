import numpy as np
from flying.cf_control import CrazyflieControl, custom_sleep, angular_distance


from util import get_yaw, project_patch, PatchDisplayThread

import cv2

from time import sleep


def move_patch(T, delta_sf=0.0, delta_tx=0.0, delta_ty=0.0):
    T[0,0] += delta_sf
    T[1,1] += delta_sf
    T[0,2] += delta_tx
    T[1,2] += delta_ty

    return T

if __name__ == "__main__":

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

    # move the patch all the way to the left
    for i in range(750):
        T = move_patch(T, delta_tx=-1)
        projected_patch = project_patch(patch, T, background)
        display_thread.update(projected_patch)
        sleep(0.01)

    # move the patch all the way to the right
    patch_size = patch.shape[0] * T[0,0]
    right_edge = int(projector_display_size[1] - patch_size)

    for i in range(right_edge):
        T = move_patch(T, delta_tx=1)
        projected_patch = project_patch(patch, T, background)
        display_thread.update(projected_patch)
        sleep(0.01)


    while True:
        if display_thread._stay_alive == False:
            break