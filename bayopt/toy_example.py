import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from matplotlib import pyplot as plt

from tqdm import tqdm
import os
import yaml
from time import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize a patch using Bayesian Optimization.')
    parser.add_argument('--patch_size', type=int, default=5, help='Size of the patch to optimize.')
    parser.add_argument('--num_seeds', type=int, default=10, help='Number of seeds for the optimization.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for the optimization.')

    args = parser.parse_args()
    
    # # Define the cube size
    # size = 500
    # cube_shape = (size, size, size)

    # # Specify the center and radius of the sphere
    # center = (20, 20, 20)
    # radius = 300

    # # Create a grid of coordinates
    # x, y, z = np.indices(cube_shape)

    # # Calculate the distance of each point from the sphere's center
    # distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # # Normalize the distances: values from 1 (periphery) to 0 (center)
    # normalized = np.clip((radius - distance) / radius, 0, 1)

    # # Set values outside the sphere to 0
    # sphere = normalized * (distance <= radius)

    # plt.figure()
    # plt.imshow(sphere[:, :, center[2]], cmap='viridis')
    # plt.colorbar(label='Value')
    # plt.title('Slice of the cube where the center of the sphere is 0')
    # plt.savefig('cube_slice.png')



    # # on to the optimization:
    # # define bounds:
    # pbounds = {'sf': (0, 99), 'tx': (0, 99), 'ty': (0, 99)}

    # # define the function to optimize
    # def black_box_function(sf, tx, ty):
    #     return sphere[int(sf), int(tx), int(ty)]

    
    # # optimizer.maximize(
    # # init_points=5,
    # # n_iter=100,
    # # )
    
    initial_patch = np.random.rand(args.patch_size, args.patch_size)
    pbounds = {'opt_patch_{}_{}'.format(i, j): (0, 1) for i in range(initial_patch.shape[0]) for j in range(initial_patch.shape[1])}
    print(pbounds)
    
    def black_box_function(opt_patch_dict):
        opt_patch =  np.array(list(opt_patch_dict.values())).reshape(initial_patch.shape)
        return -np.mean((opt_patch - initial_patch)**2)
    
    acq = acquisition.UpperConfidenceBound(kappa=2.5)
    
    optimizer = BayesianOptimization(f=None,
                                     acquisition_function=acq,
                                     pbounds=pbounds,
                                     verbose=2,
                                     random_state=1,)
    

    results_dir = f'results/{initial_patch.shape[0]}x{initial_patch.shape[1]}'
    os.makedirs(results_dir, exist_ok=True)

    success = 0
    avg_iterations = []
    best_losses = []
    avg_time = []

    seed_list = np.random.randint(0, 100, args.num_seeds)

    for seed in tqdm(seed_list):
        start_time = time()
        #define acquisiting function
        acq = acquisition.UpperConfidenceBound(kappa=2.5)

        optimizer = BayesianOptimization(f=None, # because we don't know f usually
                    acquisition_function = acq,
                    pbounds=pbounds,
                    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state=5,
                    )

        optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

        # for i in range(200):
        #     next_point = optimizer.suggest()
        #     target = black_box_function(**next_point)
        #     optimizer.register(params=next_point, target=target)
        #     #print(target, next_point)
        #     if target >= 0.99:
        #         #print("Found max after {} iterations!".format(i))
        #         success += 1
        #         avg_iterations.append(i)
        #         break
        
        losses = []
        
        for i in range(args.epochs):
            next_patch = optimizer.suggest()
            # next_patch = np.array(list(next_patch.values())).reshape(initial_patch.shape)
            target = black_box_function(next_patch)
            optimizer.register(params=next_patch, target=target)
            #print(target)
            losses.append([i, target])
            # if target >= -0.01:
            #     print("Found max after {} iterations!".format(i))
            #     success += 1
            #     break
            
        avg_time.append(time() - start_time)
            
        np.save(f'{results_dir}/losses_{seed}.npy', np.array(losses))
        avg_iterations.append(np.argmax(losses, axis=0)[1])
        
        if len(best_losses) == 0:
            np.save(f'{results_dir}/best_patch.npy', np.array(list(optimizer.max['params'].values())).reshape(initial_patch.shape))
            best_seed = seed
        elif optimizer.max['target'] > max(best_losses):
            print("New best patch with loss: ", optimizer.max['target'], " seed: ", seed)
            np.save(f'{results_dir}/best_patch.npy', np.array(list(optimizer.max['params'].values())).reshape(initial_patch.shape))
            best_seed = seed
            
        best_losses.append(optimizer.max['target'])
        
        
        
        optimal_patch = np.array(list(optimizer.max['params'].values())).reshape(initial_patch.shape)
        np.save(f'{results_dir}/optimal_patch_{seed}.npy', optimal_patch)        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(optimal_patch, cmap='gray')
        ax[0].set_title('Optimized patch')
        ax[1].imshow(initial_patch, cmap='gray')
        ax[1].set_title('Initial patch')
        plt.savefig(f'{results_dir}/patch_comparison_{seed}.png')
        

    
    #print("Success rate: ", success/len(seed_list))
    print("Average iterations: ", np.mean(avg_iterations))
    print("Average best losses: ", np.mean(best_losses))
    
    # elapsed_time = time() - time
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # print(optimizer.max)
    
    results = {
        #'success_rate': float(success / len(seed_list)),
        'average_iterations': float(np.mean(avg_iterations)),
        'average_best_losses': float(np.mean(best_losses)),
        'average_time': float(np.mean(avg_time)),
        'best_seed': int(best_seed),
        'num_epochs': int(args.epochs),
    }
        
        
        
    with open(os.path.join(results_dir, 'results.yaml'), 'w') as file:
        yaml.dump(results, file)

    # results:
    # 3 parameters sf, tx, ty in 500x500x500 cube
    # Success rate:  1
    # Average iterations:  26.25