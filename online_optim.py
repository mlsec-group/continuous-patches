import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import SequentialDomainReductionTransformer

from flying.cf_control import CrazyflieControl, custom_sleep, angular_distance

import os
import yaml

import cv2

from matplotlib import pyplot as plt
from threading import Thread
from collections import deque

from util import PatchDisplayThread, PoseUpdater, project_patch, get_yaw

import time

from pathlib import Path            

from tqdm import trange

def objective_function(current_pose, target_pose):
    position_dist = np.linalg.norm(np.array(current_pose[:3]) - np.array(target_pose[:3]))
    # angle_dist = rowan.geometry.intrinsic_distance(current_pose[3:], target_pose[3:])
    # print("Quaternions: ", current_pose[3:])
    # print("Euler angles: ", get_euler_angles(current_pose[3:]))
    angle_dist = np.abs(angular_distance(get_yaw(current_pose[3:]), target_pose[3]))
    if angle_dist > np.radians(45):
        angle_dist = 4.
    # print(f"Position distance: {position_dist}, Angle distance: {angle_dist}")

    return -(position_dist + angle_dist)
    # return -position_dist

def scale_tx_ty(sf, tx, ty, patch_size=80, projector_size=(1050, 1680)):
    scaled_patch_size = patch_size * sf
    max_tx = projector_size[1] - scaled_patch_size
    max_ty = projector_size[0] - scaled_patch_size
    return tx * max_tx, ty * max_ty


# def scale_transform(sf_opt, tx_opt, ty_opt, drone_position, patch_size=80, projector_size=(1050, 1680)):
#     sf_alpha = 10.
#     display_x_world = 2.0

#     sf_scaled = sf_opt * sf_alpha * (np.linalg.norm(drone_position[0] - display_x_world) / display_x_world)
#     sf_scaled = max(2, sf_scaled)

#     scaled_patch_size = patch_size * sf_scaled
#     max_tx = projector_display_size[1] - scaled_patch_size

#     scaled_tx_pos = (drone_position[1] + 1.3) / 2 * max_tx
#     scaled_tx = tx_opt * scaled_tx_pos
#     scaled_tx = max(0, scaled_tx)

#     scaled_ty = ty_opt * (projector_display_size[0] - scaled_patch_size)
#     scaled_ty = max(0, scaled_ty)

#     return sf_scaled, scaled_tx, scaled_ty

def training(drone, config, display_update, patches, background, target_pose, result_dir, optim_seed, load_optim=False):

    # drone = CrazyflieControl(config)
    pose_getter = PoseUpdater(drone.pose)
    pbounds = {'patch_idx': (0, len(patches)-1), 'sf': (2, 10), 'tx': (0, 1), 'ty': (0, 1)}

    bounds_transformer = SequentialDomainReductionTransformer()

    # pbounds = {'tx': (0, 1680), 'ty': (0, 1050)}
    acq = acquisition.UpperConfidenceBound(kappa=2.5)
    # acq = acquisition.ExpectedImprovement(xi=0.01) # x = 0.0 -> exploitation, x = 0.1 -> exploration

    optimizer = BayesianOptimization(f=None,
                                     acquisition_function=acq,
                                     pbounds=pbounds,
                                     verbose=2,
                                     random_state=optim_seed,
                                     allow_duplicate_points = True,
                                     bounds_transformer=bounds_transformer)
    
    optimizer.set_gp_params(alpha=1e-2, n_restarts_optimizer=20)

    num_lines = 0
    counter = 0
    best_loss = -np.inf
    if load_optim:
        load_logs(optimizer, logs=[str(result_dir / "logs.log")])
        # count the entries in the log file
        with open(str(result_dir / "logs.log")) as f:
            num_lines = sum(1 for line in f)
            print(f"Loaded {num_lines} entries from log file!")
        
        best_loss = optimizer.max['target']

    logger = JSONLogger(path=str(result_dir / "logs.log"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    counter = 0

    try:
        for i in range(num_lines, 30):
            # quick battery check
            bat_v, bat_s = drone.battery
            if bat_s == 3:
                print("Low battery!! Landing...")
                drone.land()
                custom_sleep(2., drone.occupied, True)
                pose_getter.close()
                drone.power_off()
                drone.close()

                print("Perform battery change and hit y if ready!")
                while True:
                    if input('Ready? ') == 'y':
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
            start_yaw = get_yaw(start_pose)


            params = optimizer.suggest()
            patch_idx = int(np.round(params['patch_idx'], decimals=0))
            sf = params['sf']
            tx, ty = scale_tx_ty(sf, params['tx'], params['ty'])

            # automatic scaling
            # sf, tx, ty = scale_transform(params['sf'], params['tx'], params['ty'], start_pose[:3])

            T = np.zeros((3,3))
            T[0,0] = sf # sf
            T[1,1] = sf # sf
            T[0,2] = tx # tx
            T[1,2] = ty # ty
            T[2,2] = 1

            projected_patch = project_patch(patches[patch_idx], T, background)
            display_update(projected_patch)


            #  automatic scaling, needs to be moved directly into projector thread
            # for 2.5 seconds, update the display with the projected patch and the current pose
            # current_time = time.time()
            # drone.toggle_frontnet()
            # while time.time() - current_time < 2.5:
            #     current_pose, _ = pose_getter.get_current_pose()

            #     sf, tx, ty = scale_transform(params['sf'], params['tx'], params['ty'], current_pose[:3])

            #     T = np.zeros((3,3))
            #     T[0,0] = sf # sf
            #     T[1,1] = sf # sf
            #     T[0,2] = tx # tx
            #     T[1,2] = ty # ty
            #     T[2,2] = 1

            #     projected_patch = project_patch(patches[patch_idx], T, background)

            #     display_update(projected_patch)
            #     custom_sleep(0.1, drone.occupied, True)

            custom_sleep(0.2, drone.occupied, True)

            drone.toggle_frontnet()

            custom_sleep(2.5, drone.occupied, True)

            current_pose, current_time = pose_getter.get_current_pose()
            current_yaw = get_yaw(start_pose)

            drone.toggle_frontnet()
            custom_sleep(1.5, drone.occupied, True)

            # velocity = np.linalg.norm(np.array(current_pose[:3]) - np.array(start_pose[:3])) / (current_time - start_time)

            # print(f"Velocity: {velocity}")
            # print(f"Current position: {current_pose[:3]}")

            loss = objective_function(current_pose, target_pose)
            print(f"Loss {i}: {loss}")
            optimizer.register(params=params, target=loss)

            if loss > best_loss:
                counter += 1
                print(f"Found better parameters for {counter} time!")
                best_loss = loss


            drone.reset()
            custom_sleep(1.5, drone.occupied, True)
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
    
    if drone.frontnet == '1':
        drone.toggle_frontnet()
        custom_sleep(1.5, drone.occupied, True)

    drone.reset()
    custom_sleep(1.5, drone.occupied, True)
    drone.land()
    custom_sleep(1.0, cf.occupied, True)
    pose_getter.close()
    drone.close()


    return optimizer.max

def select_patches(drone, display_update, patches, num_patches):
    pose_getter = PoseUpdater(drone.pose)

    losses = [0.0] * len(patches)

    for patch_idx in range(len(patches)):
        # patch_idx = i % len(patches)

        bat_v, bat_s = drone.battery
        if bat_s == 3:
            print("Low battery!! Landing...")
            drone.land()
            custom_sleep(2., drone.occupied, True)
            pose_getter.close()
            drone.power_off()
            drone.close()

            print("Perform battery change and hit y if ready!")
            while True:
                if input('Ready? ') == 'y':
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
        custom_sleep(3.0, drone.occupied, True)

        patch = patches[patch_idx]

        T = np.zeros((3,3))
        T[0,0] = 7 # sf
        T[1,1] = 7 # sf
        T[0,2] = 750 # tx
        T[1,2] = 300 # ty
        T[2,2] = 1

        projected_patch = project_patch(patch, T, background)
        display_update(projected_patch)
        custom_sleep(0.2, drone.occupied, True)

        drone.toggle_frontnet()

        custom_sleep(2.5, drone.occupied, True)

        current_pose, current_time = pose_getter.get_current_pose()
        current_yaw = get_yaw(current_pose[3:])

        drone.toggle_frontnet()
        custom_sleep(1.5, drone.occupied, True)
        # poses = np.array([entry[0] for entry in pose_getter.pose_accu])
        # yaws = np.array([get_yaw(pose[3:]) for pose in poses])

        # loss = np.abs(angular_distance(np.mean(yaws), 0.0))
        loss = np.abs(angular_distance(current_yaw, 0.0))
        print(f"Loss {patch_idx}: {loss}")
        losses[patch_idx] += loss
        np.save('data/yolo_patches/patch_losses.npy', losses)


    drone.reset()
    custom_sleep(1.5, drone.occupied, True)
    drone.land()
    custom_sleep(1.0, cf.occupied, True)
    # drone.close()
    # drone.power_off()
    pose_getter.close()

    print("All losses: ", losses)
    np.save('data/yolo_patches/patch_losses.npy', losses)
    # return the top num_patches patches
    return drone, [patches[i] for i in np.argsort(losses)[:num_patches]]


def brute_force(drone, config, display_update, patches, background, target_pose, result_dir, seed):
    pose_getter = PoseUpdater(drone.pose)


    rng = np.random.default_rng(seed)

    results = []

    for i in range(30):
        bat_v, bat_s = drone.battery
        if bat_s == 3:
            print("Low battery!! Landing...")
            drone.land()
            custom_sleep(2., drone.occupied, True)
            pose_getter.close()
            drone.power_off()
            drone.close()

            print("Perform battery change and hit y if ready!")
            while True:
                if input('Ready? ') == 'y':
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
        custom_sleep(3.0, drone.occupied, True)

        patch_idx = rng.integers(0, len(patches))
        sf = rng.random() * (10-2) + 2   # random scaling factor between 2 and 10
        tx, ty = scale_tx_ty(sf, rng.random(), rng.random())

        T = np.zeros((3,3))
        T[0,0] = sf # sf
        T[1,1] = sf # sf
        T[0,2] = tx # tx
        T[1,2] = ty # ty
        T[2,2] = 1

        projected_patch = project_patch(patches[patch_idx], T, background)
        display_update(projected_patch)
        custom_sleep(0.2, drone.occupied, True)

        drone.toggle_frontnet()

        custom_sleep(2.5, drone.occupied, True)

        current_pose, current_time = pose_getter.get_current_pose()
        current_yaw = get_yaw(current_pose[3:])


        drone.toggle_frontnet()
        custom_sleep(1., drone.occupied, True)

        loss = objective_function(current_pose, target_pose)
        print(f"Loss {i}: {loss}")
        results.append((loss, patch_idx, sf, tx, ty))


    while drone.frontnet == '1':
        drone.toggle_frontnet()
        custom_sleep(.5, drone.occupied, True)

    drone.reset()
    custom_sleep(1.5, drone.occupied, True)
    drone.land()
    custom_sleep(1.0, cf.occupied, True)
    pose_getter.close()
    drone.close()

    results = np.array(results)
    np.save(str(result_dir / 'all_losses.npy'), results)
    best_idx = np.argmax(results[:, 0])
    best_params = {'target': results[best_idx, 0], 'params': {'patch_idx': results[best_idx, 1], 'sf': results[best_idx, 2], 'tx': results[best_idx, 3], 'ty': results[best_idx, 4]}}

    return best_params


def brute_force_2(drone, config, display_update, patches, background, target_pose, result_dir, seed):
    pose_getter = PoseUpdater(drone.pose)


    rng = np.random.default_rng(seed)

    results = []
    #try loading results from folder if they exist
    if os.path.exists(result_dir / 'all_poses.npy'):
        results = np.load(result_dir / 'all_poses.npy').tolist()
        print("Loaded previous results!")

    initial_idx = len(results)

    for i in trange(500):
        bat_v, bat_s = drone.battery
        if bat_s == 3:
            print("Low battery!! Landing...")
            drone.land()
            custom_sleep(2., drone.occupied, True)
            pose_getter.close()
            drone.power_off()
            drone.close()

            print("Perform battery change and hit y if ready!")
            while True:
                if input('Ready? ') == 'y':
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

        # check if CF fell down
        current_pose, _ = pose_getter.get_current_pose()
        if current_pose[2] < 0.5:
            del drone
            drone = CrazyflieControl(config)
            pose_getter = PoseUpdater(drone.pose)

            while True:
                if drone.connected:
                    break
            print("Continue flying...")
            drone.takeoff(1.0, 3)
            custom_sleep(2., drone.occupied, True)

        # if drone.scf.is_link_open() == False:
        #     print("Connection lost! Reconnecting...")
        #     del drone
        #     drone = CrazyflieControl(config)
        #     pose_getter = PoseUpdater(drone.pose)

        #     while True:
        #         if drone.connected:
        #             break
        #     print("Continue flying...")
        #     drone.reset()
        #     custom_sleep(3., drone.occupied, True)
            
        
        if i > initial_idx:
            drone.reset()
            custom_sleep(3.0, drone.occupied, True)


        patch_idx = rng.integers(0, len(patches))
        sf = rng.random() * (10-2) + 2   # random scaling factor between 2 and 10
        tx, ty = scale_tx_ty(sf, rng.random(), rng.random())

        T = np.zeros((3,3))
        T[0,0] = sf # sf
        T[1,1] = sf # sf
        T[0,2] = tx # tx
        T[1,2] = ty # ty
        T[2,2] = 1

        # don't actually fly the drone if we are just loading previous results
        # this is just to get the rng to the correct state for reproducibility and to ensure we don't always start from the same point
        # again after loading
        if i-1 < initial_idx:
            continue

        projected_patch = project_patch(patches[patch_idx], T, background)
        display_update(projected_patch)
        custom_sleep(0.2, drone.occupied, True)

        drone.toggle_frontnet()

        custom_sleep(2.5, drone.occupied, True)

        current_pose, current_time = pose_getter.get_current_pose()
        current_yaw = get_yaw(current_pose[3:])


        drone.toggle_frontnet()
        custom_sleep(1., drone.occupied, True)

        # loss = objective_function(current_pose, target_pose)
        # print(f"Loss {i}: {loss}")
        results.append((i, patch_idx, sf, tx, ty, *current_pose, current_yaw))
        np.save(str(result_dir / 'all_poses.npy'), results)


    while drone.frontnet == '1':
        drone.toggle_frontnet()
        custom_sleep(.5, drone.occupied, True)

    drone.reset()
    custom_sleep(1.5, drone.occupied, True)
    drone.land()
    custom_sleep(1.0, cf.occupied, True)
    pose_getter.close()
    drone.close()

    results = np.array(results)
    np.save(str(result_dir / 'all_poses.npy'), results)
    # best_idx = np.argmax(results[:, 0])
    # best_params = {'target': results[best_idx, 0], 'params': {'patch_idx': results[best_idx, 1], 'sf': results[best_idx, 2], 'tx': results[best_idx, 3], 'ty': results[best_idx, 4]}}

    # return best_params


def ensure_types_for_save(result_dictionary):
    result_dictionary['target'] = float(result_dictionary['target'])
    result_dictionary['params']['patch_idx'] = int(np.round(result_dictionary['params']['patch_idx'], 0))
    result_dictionary['params']['sf'] = float(result_dictionary['params']['sf'])
    result_dictionary['params']['tx'] = int(np.round(result_dictionary['params']['tx'], 0))
    result_dictionary['params']['ty'] = int(np.round(result_dictionary['params']['ty'], 0))

    return result_dictionary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str, default='results/')
    parser.add_argument('direction', type=str, choices=['left', 'right', 'forward', 'backward'], default='left')
    parser.add_argument('trial', type=int, default=0)
    parser.add_argument('--brute_force', action='store_true', default=False)
    parser.add_argument('--load_selected', action='store_true', default=False)
    parser.add_argument('--load_optimizer', action='store_true', default=False)
    parser.add_argument('--patches', type=str, choices=['frontnet', 'yolo'], default='yolo')

    args = parser.parse_args()

    result_dir = Path(f"{args.result_dir}/{args.direction}/{args.trial}/")
    os.makedirs(result_dir, exist_ok=True)

    load_selected = args.load_selected
    load_optimizer = args.load_optimizer

    with open('flying/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    projector_display_size = (1050, 1680)
    background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

    # Start the PatchDisplayThread
    display_thread = PatchDisplayThread("Patch", (2561, 0))
    display_thread.start()
    display_thread.update(background)


    T = np.zeros((3,3))
    T[0,0] = 5 # sf
    T[1,1] = 5 # sf
    T[0,2] = 750 # tx
    T[1,2] = 300 # ty
    T[2,2] = 1


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

    cf.takeoff(1.0, 3)

    custom_sleep(2., cf.occupied)

    if load_selected:
        print("Loading preselected patches...")
        patches = np.load(f'results/preselected_patches_{args.patches}.npy')
    else:
        print("Selecting patches...")
        files = [f for f in sorted(os.listdir(f"data/{args.patches}_patches")) if f.endswith('.jpg')]
        patches = [cv2.imread(f"data/{args.patches}_patches/" + f) for f in files]
        face_patch = cv2.imread("data/custom_patch80x80.jpg")
        patches.append(face_patch)

        random_patch = np.random.randint(255, size=(80,80,3),dtype=np.uint8)
        patches.append(random_patch)

        np.save(f'results/all_patches_{args.patches}.npy', np.array(patches))
        
        
        # patches = np.load('results/all_patches.npy')

        cf, patches = select_patches(cf, display_thread.update, patches, 4)

        np.save(f'results/preselected_patches_{args.patches}.npy', np.array(patches))


    projected_patch = project_patch(patches[0], T, background)
    display_thread.update(projected_patch)

    seed = args.trial

    match args.direction:
        # center is at [0.0, -0.3, 1.]
        case 'left':
            target_pose = np.array([1.0, 0.7, 1.0, 0.0]) # x, y, z, yaw
        case 'right':
            target_pose = np.array([1.0, -1.3, 1.0, 0.0])
            seed += 10 
        case 'forward':
            target_pose = np.array([1.5, -0.3, 1.0, 0.0]) 
            seed += 20
        case 'backward':
            target_pose = np.array([0.0, -0.3, 1.0, 0.0])
            seed += 30

    
    rng = np.random.default_rng(seed)
    optim_seed = int(rng.integers(0, 1e6))
    print(f"Seed: {seed}, Optim seed: {optim_seed}")
    

    if args.brute_force:
        best_results = brute_force(cf, config, display_thread.update, patches, background, target_pose, result_dir, seed) 
        # brute_force_2(cf, config, display_thread.update, patches, background, target_pose, result_dir, seed)   
    else:
        best_results = training(cf, config, display_thread.update, patches, background, target_pose, result_dir, optim_seed=optim_seed, load_optim=load_optimizer)
        scaled_tx, scaled_ty = scale_tx_ty(best_results['params']['sf'], best_results['params']['tx'], best_results['params']['ty'])
        # scaled_sf, scaled_tx, scaled_ty = scale_transform(best_results['params']['sf'], best_results['params']['tx'], best_results['params']['ty'], target_pose[:3])
        best_results['params']['tx'] = scaled_tx
        best_results['params']['ty'] = scaled_ty


    best_results = ensure_types_for_save(best_results)


    print(f"Best loss: {best_results['target']}, Best parameters: {best_results['params']}")
    with open(result_dir / 'results.yaml', 'w') as file:
        yaml.dump(best_results, file)

    best_patch = patches[int(np.round(best_results['params']['patch_idx'], decimals=0))]
    cv2.imwrite(str(result_dir / 'best_patch.jpg'), best_patch)

    np.save(result_dir / 'best_patch.npy', best_patch)
    
    cf = CrazyflieControl(config)

    cf.takeoff(1.0, 3)
    custom_sleep(3., cf.occupied, True)
    cf.reset()
    custom_sleep(3., cf.occupied, True)

    T = np.zeros((3,3))
    T[0,0] = best_results['params']['sf'] # sf
    T[1,1] = best_results['params']['sf'] # sf
    T[0,2] = best_results['params']['tx'] # tx
    T[1,2] = best_results['params']['ty'] # ty
    T[2,2] = 1

    

    projected_patch = project_patch(best_patch, T, background)
    display_thread.update(projected_patch)
    custom_sleep(0.2, cf.occupied, True)

    cf.toggle_frontnet()
    custom_sleep(10., cf.occupied, True)
    print("Pose: ", cf.pose[0][:3], np.degrees(get_yaw(cf.pose[0][3:])))
    cf.toggle_frontnet()
    custom_sleep(2., cf.occupied, True)

    cf.land()
    custom_sleep(5., cf.occupied, True)

    print("Closing...")
    display_thread.close()
    cf.close()
