import numpy as np
import math
import rowan
import cv2

from threading import Thread
from collections import deque

import time

def bernstein_poly(i, n, t):
    return math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, n_points=100):
    t = np.linspace(0, 1, n_points)
    n = len(control_points) - 1

    curve = np.zeros((n_points, 3))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), control_points[i])

    return curve

# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q


def scale_tx_ty(sf, tx, ty, patch_size=80, projector_size=(1050, 1680)):
    scaled_patch_size = patch_size * sf
    max_tx = projector_size[1] - scaled_patch_size
    max_ty = projector_size[0] - scaled_patch_size
    return tx * max_tx, ty * max_ty

def construct_T(sf, tx, ty):
    T = np.zeros((3,3))
    T[0,0] = sf
    T[1,1] = sf
    T[0,2] = tx
    T[1,2] = ty
    T[2,2] = 1
    return T


def get_yaw(quats):
    return rowan.to_euler(rowan.normalize(np.array(quats)))[0]      # returns yaw in radians

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
    def __init__(self, cf_pose, queue_size=5):
        super().__init__()
        self._stay_alive = True
        self.cf_pose = cf_pose
        self.pose_accu = deque(maxlen=queue_size)   # We're receiving data with roughly 100 Hz, if queue size is 100, we have ~1 second of data

        self.start()

    def run(self):
        while self._stay_alive:
            if self.cf_pose:
                self.pose_accu.append([np.array(self.cf_pose[0]), time.time()])

    def get_current_pose(self):
        return np.array(self.pose_accu[-1][0]), self.pose_accu[-1][1]

    def close(self):
        self._stay_alive = False