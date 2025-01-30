import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

from flying.cf_control import CrazyflieControl

import os
import yaml
from time import time

import cv2

from matplotlib import pyplot as plt
from threading import Thread
from queue import Queue

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
        self.queue = Queue()

    def run(self):
        # Create a named window and move it to the second monitor
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, *self.position)  # Assuming the second monitor is to the right of the primary monitor
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while self._stay_alive:
            if not self.queue.empty():
                img = self.queue.get()
                cv2.imshow(self.name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.close()

    def update(self, img):
        # Add the image to the queue
        self.queue.put(img)

    def close(self):
        # Destroy the window
        self._stay_alive = False
        cv2.destroyAllWindows()
            



if __name__ == "__main__":

    with open('flying/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    # cf = CrazyflieControl(config)

    projector_display_size = (1050, 1680)

    background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

    # Start the PatchDisplayThread
    display_thread = PatchDisplayThread("Patch", (2561, 0))
    display_thread.start()
    display_thread.update(background)




    random_patch = np.random.randint(255, size=(80,80,3),dtype=np.uint8)


    T = np.zeros((3,3))
    T[0,0] = 4 # sf
    T[1,1] = 4 # sf
    T[0,2] = 100 # tx
    T[1,2] = 100 # ty
    T[2,2] = 1
    print(T)

    projected_patch = project_patch(random_patch, T, background)

    display_thread.update(projected_patch)




    # cf.close()