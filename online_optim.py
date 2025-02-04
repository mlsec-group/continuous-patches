import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

from flying.cf_control import CrazyflieControl, custom_sleep

import os
import yaml

import cv2

from matplotlib import pyplot as plt
from threading import Thread
from collections import deque

import rowan

def get_euler_angles(quats):
    return rowan.to_euler(rowan.normalize(np.array(quats)))

def project_patch(patch, T, image):
    # using cv2 to project the patch instead of FAP place_patch() function,
    # since we don't need to calculate gradients
    width, height = image.shape[:2]
    # print(height, width)
    mask = np.ones_like(patch)

    warped_patch = cv2.warpPerspective(patch, T, (height, width), flags=cv2.INTER_NEAREST)
    mask = cv2.warpPerspective(mask, T, (height, width), flags=cv2.INTER_NEAREST)

    mod_img = image * ~mask.astype(bool)
    mod_img += warped_patch

    return mod_img # return a np array instead of jnp array and convert to double

class PatchDisplayThread(Thread):
    def __init__(self, name, position):
        super().__init__()
        self.name = name
        self.position = position
        self._stay_alive = True
        self.queue = deque(maxlen=5)

    def run(self):
        # Create a named window and move it to the second monitor
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, *self.position)  # Assuming the second monitor is to the right of the primary monitor
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while self._stay_alive:
            if self.queue:
                img = self.queue.popleft()
                cv2.imshow(self.name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.close()

    def update(self, img):
        # Add the image to the queue
        self.queue.append(img)

    def close(self):
        # Destroy the window
        self._stay_alive = False
        cv2.destroyAllWindows()


class PoseUpdater(Thread):
    def __init__(self, cf_pose, queue_size=100):
        super().__init__()
        self._stay_alive = True
        self.cf_pose = cf_pose
        self.pose_accu = deque(maxlen=queue_size)   # We're receiving data with roughly 100 Hz, if queue size is 100, we have ~1 second of data

        self.start()

    def run(self):
        while self._stay_alive:
            if self.cf_pose:
                self.pose_accu.append(np.array(self.cf_pose[0]))

    def get_current_pose(self):
        return np.array(self.pose_accu[-1])

    def close(self):
        self._stay_alive = False
            



if __name__ == "__main__":

    with open('flying/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    projector_display_size = (1050, 1680)

    background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

    # Start the PatchDisplayThread
    display_thread = PatchDisplayThread("Patch", (2561, 0))
    display_thread.start()
    display_thread.update(background)

    person = cv2.imread("/home/phanfeld/Downloads/istockphoto-625389694-612x612.jpg")

    T = np.zeros((3,3))
    T[0,0] = 1 # sf
    T[1,1] = 1 # sf
    T[0,2] = 750 # tx
    T[1,2] = 160 # ty
    T[2,2] = 1
    print(T)

    projected_patch = project_patch(person, T, background)

    display_thread.update(projected_patch)


    cf = CrazyflieControl(config)

    # random_patch = np.random.randint(255, size=(80,80,3),dtype=np.uint8)

    cf_poses = PoseUpdater(cf.pose)

    cf.takeoff()

    custom_sleep(5., cf.occupied)

    cf.toggle_frontnet()
    custom_sleep(10., cf.occupied, True)

    print(cf_poses.get_current_pose())

    cf.reset()
    custom_sleep(1., cf.occupied, True)

    cf.land()

    custom_sleep(5., cf.occupied)

    print("Closing...")
    cf_poses.close()
    cf.close()
