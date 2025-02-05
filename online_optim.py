import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from flying.cf_control import CrazyflieControl, custom_sleep

import os
import yaml

import cv2

from matplotlib import pyplot as plt
from threading import Thread
from collections import deque

import rowan

import time

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
                self.pose_accu.append([np.array(self.cf_pose[0]), time.time()])

    def get_current_pose(self):
        return np.array(self.pose_accu[-1][0]), self.pose_accu[-1][1]

    def close(self):
        self._stay_alive = False
            

def objective_function(current_pose, target_pose):
    position_dist = np.linalg.norm(np.array(current_pose[:3]) - np.array(target_pose[:3]))
    angle_dist = rowan.geometry.intrinsic_distance(current_pose[3:], target_pose[3:])
    # print("Quaternions: ", current_pose[3:], target_pose[3:])
    # print(f"Position distance: {position_dist}, Angle distance: {angle_dist}")

    return -(position_dist + (angle_dist/np.pi))
    # return -position_dist

def training(drone, config, display_update, patch, background, load_logs=False):

    pose_getter = PoseUpdater(drone.pose)
    pbounds = {'sf': (1, 10), 'tx': (0, 1680), 'ty': (0, 1050)}
    # acq = acquisition.UpperConfidenceBound(kappa=2.5)
    acq = acquisition.ExpectedImprovement(xi=0.03) # x = 0.0 -> exploitation, x = 0.1 -> exploration

    optimizer = BayesianOptimization(f=None,
                                     acquisition_function=acq,
                                     pbounds=pbounds,
                                     verbose=2,
                                     random_state=1,
                                     allow_duplicate_points = True)
    
    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

    if load_logs:
        load_logs(optimizer, logs=["results/logs.log"])

    os.makedirs("results", exist_ok=True)
    logger = JSONLogger(path="results/logs.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    target_pose = np.array([2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    try:
        for i in range(200):
            # quick battery check
            bat_v, bat_s = drone.battery
            if bat_s == 3:
                print("Low battery detected! Landing...")
                drone.land()
                custom_sleep(2., drone.occupied, True)
                pose_getter.close()
                drone.power_off()
                drone.close()

                print("Perform battery change and hit y if ready!")
                while True:
                    if input('Ready?') == 'y':
                        break
                
                del drone
                drone = CrazyflieControl(config)
                pose_getter = PoseUpdater(drone.pose)

                while True:
                    if drone.connected:
                        break
                print("Continue flying...")
                drone.takeoff(1.0, 3)
                custom_sleep(2., drone.occupied, True)
            
            
            drone.reset()
            custom_sleep(1., drone.occupied, True)
            start_pose, start_time = pose_getter.get_current_pose()
            start_yaw = get_euler_angles(start_pose)[2]


            params = optimizer.suggest()
            sf = params['sf']
            tx = params['tx']
            ty = params['ty']

            T = np.zeros((3,3))
            T[0,0] = sf # sf
            T[1,1] = sf # sf
            T[0,2] = tx # tx
            T[1,2] = ty # ty
            T[2,2] = 1

            projected_patch = project_patch(patch, T, background)
            display_update(projected_patch)

            drone.toggle_frontnet()

            custom_sleep(2., drone.occupied, True)

            drone.toggle_frontnet()
            current_pose, current_time = pose_getter.get_current_pose()
            current_yaw = get_euler_angles(start_pose)[2]

            # velocity = np.linalg.norm(np.array(current_pose[:3]) - np.array(start_pose[:3])) / (current_time - start_time)

            # print(f"Velocity: {velocity}")
            # print(f"Current position: {current_pose[:3]}")

            loss = objective_function(current_pose, target_pose)
            print(f"Loss: {loss}")
            optimizer.register(params=params, target=loss)

            drone.reset()
            custom_sleep(1.0, drone.occupied, True)
    except Exception as e:
        print(e)
        print("Training interrupted!")
        if drone.frontnet == '1':
            drone.toggle_frontnet()
            custom_sleep(.5, drone.occupied, True)
        drone.land()
        custom_sleep(5., cf.occupied)
        pose_getter.close()
        return optimizer.max
    
    pose_getter.close()

    return optimizer.max

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

    patch = cv2.imread("data/frontnet_gt_patch_distance.jpg")
    print(patch.shape)

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

    # if bat_v <= 3900: 
    #     print('Battery below 3.9V!! Not flying.')
    #     cf.close()
    #     display_thread.close()
    #     exit()

    # random_patch = np.random.randint(255, size=(80,80,3),dtype=np.uint8)

    cf.takeoff(1.0, 3)

    custom_sleep(5., cf.occupied)


    best_results = training(cf, config, display_thread.update, patch, background)
    

    print(f"Best loss: {best_results['target']}, Best parameters: {best_results['params']}")
    with open('results.yaml', 'w') as file:
        yaml.dump(best_results, file)


    # test
    # for i in range(3):  
    #     cf.toggle_frontnet()
    #     custom_sleep(2., cf.occupied, True)



    #     cf.reset()
    #     custom_sleep(1., cf.occupied, True)

    # cf.reset()
    # custom_sleep(1., cf.occupied, True)
    cf.land()

    custom_sleep(5., cf.occupied)

    print("Closing...")
    display_thread.close()
    cf.close()
