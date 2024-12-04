import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition


if __name__ == "__main__":
    
    # create list of all possible values for the scale factor, tx and ty
    possible_values = np.linspace(-1, 1, 100)

    
    example_search_space = np.ones((len(possible_values), len(possible_values), len(possible_values)))


    # Define dimensions of the cube
    size = 60
    radius = size // 2  # Radius of the sphere

    # Create a grid of coordinates
    x, y, z = np.indices((size, size, size)) - radius  # Center the grid at (10, 10, 10)

    distance = np.sqrt(x**2 + y**2 + z**2)

    normalized = np.clip((radius - distance) / radius, 0, 1)

    sphere = normalized * (distance <= radius)

    import matplotlib.pyplot as plt

    # Visualize the middle slice of the sphere
    plt.imshow(sphere[:, :, radius], cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Middle slice of the sphere')
    plt.savefig('sphere.png')


    # Define the "optimal" minimum of the example_search_space
    center = np.array((40, 40, 50))
    print("Optimal values for the scale factor, tx and ty: ", possible_values[center[0]], possible_values[center[1]], possible_values[center[2]])

    # Calculate the start and end indices for the sphere within the example_search_space
    start_idx = center - radius
    end_idx = center + radius

    start_idx = np.maximum(start_idx, 0)
    end_idx = np.minimum(end_idx, len(example_search_space))

    # Delete the values of the sphere from the example_search_space
    example_search_space[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]] -= sphere

    plt.figure()
    plt.imshow(example_search_space[:, :, center[2]], cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Slice of the cube where the center of the sphere is 0')
    plt.savefig('cube_slice.png')



    # on to the optimization:
    # define bounds:
    pbounds = {'sf': (0, 99), 'tx': (0, 99), 'ty': (0, 99)}

    # define the function to optimize
    def black_box_function(sf, tx, ty):
        return -example_search_space[int(sf), int(tx), int(ty)]


    optimizer = BayesianOptimization(f=black_box_function,
                pbounds=pbounds,
                verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=1,
                )

    optimizer.maximize(
    init_points=5,
    n_iter=10,
    )
