import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    # create list of all possible values for the scale factor, tx and ty
    possible_values = np.linspace(-1, 1, 100)

    
    # example_search_space = np.ones((len(possible_values), len(possible_values), len(possible_values)))


    # Define the cube size
    size = 100
    cube_shape = (size, size, size)

    # Specify the center and radius of the sphere
    center = (20, 20, 20)
    radius = 70

    # Create a grid of coordinates
    x, y, z = np.indices(cube_shape)

    # Calculate the distance of each point from the sphere's center
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Normalize the distances: values from 1 (periphery) to 0 (center)
    normalized = np.clip((radius - distance) / radius, 0, 1)

    # Set values outside the sphere to 0
    sphere = normalized * (distance <= radius)

    plt.figure()
    plt.imshow(sphere[:, :, center[2]], cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Slice of the cube where the center of the sphere is 0')
    plt.savefig('cube_slice.png')



    # on to the optimization:
    # define bounds:
    pbounds = {'sf': (0, 99), 'tx': (0, 99), 'ty': (0, 99)}

    # define the function to optimize
    def black_box_function(sf, tx, ty):
        return sphere[int(sf), int(tx), int(ty)]

    #define acquisiting function
    acq = acquisition.UpperConfidenceBound(kappa=5)

    optimizer = BayesianOptimization(f=None, # because we don't know f usually
                acquisition_function = acq,
                pbounds=pbounds,
                verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=5,
                )

    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

    # optimizer.maximize(
    # init_points=5,
    # n_iter=100,
    # )

    for i in range(200):
        next_point = optimizer.suggest()
        target = black_box_function(**next_point)
        optimizer.register(params=next_point, target=target)
        print(target, next_point)
        if target >= 0.985:
            print("Found max after {} iterations!".format(i))
            break


    print(optimizer.max)    
