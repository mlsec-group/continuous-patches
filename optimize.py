from simulators.hoverair import HoverSim
from util import bezier_curve
from plot_results import plot_results
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm, trange

import jax
from evosax import CMA_ES

if __name__ == '__main__':
    hover_sim = HoverSim()
    
    control_points_bezier = np.array([[0, 0, 0], [1, 3, 0.2], [2, -1, 0.6], [3, 2, 1]])
    target_trajectory = bezier_curve(control_points_bezier, 100)

    all_optimal_poses = []
    all_optimal_poses.append(hover_sim.pose)
    
    for waypoint in tqdm(target_trajectory):
        strategy = CMA_ES(popsize=20, num_dims=3)   # popsize => num candidates, num_dims => number of parameters to actually optimize
        es_params = strategy.default_params
        rng = jax.random.PRNGKey(0)
        state = strategy.initialize(rng, es_params)

        for step_idx in range(50):
            rng, rng_gen = jax.random.split(rng, 2)
            params, state = strategy.ask(rng_gen, state, es_params)
            scale_factors, txs, tys = params.T
            new_poses = np.array([hover_sim.sim_new_pose(sf, tx, ty) for sf, tx, ty in params])
            fitness = np.array([hover_sim.eval(pose, waypoint) for pose in new_poses])
            state = strategy.tell(params, fitness, state, es_params)
            
            # if (step_idx + 1) % 10 == 0:
            #     print(f"CMA-ES - # GEN: {step_idx+1} | Fitness: {state.best_fitness} | Params: {state.best_member}")

        hover_sim.pose = hover_sim.sim_new_pose(*state.best_member)
        all_optimal_poses.append(hover_sim.pose)

    # for _ in range(len(target_trajectory)):
    all_optimal_poses = np.array(all_optimal_poses)    

    plot_results('.', target_trajectory, all_optimal_poses)
