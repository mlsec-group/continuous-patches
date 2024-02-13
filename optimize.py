from simulators.hoverair import HoverSim
import matplotlib.pyplot as plt

import numpy as np
import math

import jax
from evosax import CMA_ES

def bernstein_poly(i, n, t):
    return math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, n_points=100):
    t = np.linspace(0, 1, n_points)
    n = len(control_points) - 1

    curve = np.zeros((n_points, 3))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), control_points[i])

    return curve


if __name__ == '__main__':
    hover_sim = HoverSim()
    
    control_points_bezier = np.array([[0, 0, 1], [1, 3, 1], [2, -1, 1], [3, 2, 1]])
    resulting_curve = bezier_curve(control_points_bezier, 20)

    all_optimal_poses = []
    all_optimal_poses.append(hover_sim.pose)
    
    for waypoint in resulting_curve:
        strategy = CMA_ES(popsize=20, num_dims=3)   # popsize => num candidates, num_dims => num_params to actually optimize
        es_params = strategy.default_params
        rng = jax.random.PRNGKey(0)
        state = strategy.initialize(rng, es_params)

        for step_idx in range(50):
            rng, rng_gen, rng_eval = jax.random.split(rng, 3)
            params, state = strategy.ask(rng_gen, state, es_params)
            scale_factors, txs, tys = params.T
            # print(scale_factors, txs, tys)
            new_poses = np.array([hover_sim.sim_new_pose(sf, tx, ty) for sf, tx, ty in params])
            # print(new_poses)
            fitness = np.array([hover_sim.eval(pose, waypoint) for pose in new_poses])
            # print(fitness)
            state = strategy.tell(params, fitness, state, es_params)
            
            # if (step_idx + 1) % 10 == 0:
            #     print(f"CMA-ES - # GEN: {step_idx+1} | Fitness: {state.best_fitness} | Params: {state.best_member}")
        
        hover_sim.pose = hover_sim.sim_new_pose(*state.best_member)
        all_optimal_poses.append(hover_sim.pose)

    # for _ in range(len(resulting_curve)):
    all_optimal_poses = np.array(all_optimal_poses)    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(resulting_curve[:, 0], resulting_curve[:, 1], resulting_curve[:, 2], label='Desired Trajectory')
    ax.plot(all_optimal_poses[:, 0], all_optimal_poses[:, 1], all_optimal_poses[:, 2], label='Optimized')
    ax.legend()
    plt.show()