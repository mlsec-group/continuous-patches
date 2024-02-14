from simulators.hoverair import HoverSim
from util import bezier_curve
from plot_results import plot_results

from pathlib import Path

import numpy as np

from tqdm import tqdm

import jax
from evosax import Strategies

if __name__ == '__main__':
    
    control_points_bezier = np.array([[0, 0, 0], [1, 3, 0.2], [2, -1, 0.6], [3, 2, 1]])
    target_trajectory = bezier_curve(control_points_bezier, 20)

    # strategy_names = ["SimpleES", "SimpleGA", "PSO", "DE", "CMA_ES", "Sep_CMA_ES",
    #            "Full_iAMaLGaM", "Indep_iAMaLGaM", "MA_ES", "LM_MA_ES",
    #            "RmES", "GLD", "SimAnneal", "GESMR_GA", "SAMR_GA"]

    strategy_names = ['CMA_ES', 'PSO', "Sep_CMA_ES"]
    
    all_trajectories = []
    
    for strategy_name in tqdm(strategy_names):
        print(strategy_name)
        hover_sim = HoverSim()
        optimized_trajectory = []
        optimized_trajectory.append(hover_sim.pose)

    
        for waypoint in tqdm(target_trajectory):
            strategy = Strategies[strategy_name](popsize=20, num_dims=3)   # popsize => num candidates, num_dims => number of parameters to actually optimize
            es_params = strategy.default_params
            rng = jax.random.PRNGKey(0)
            state = strategy.initialize(rng, es_params)

            for step_idx in range(50):
                rng, rng_gen = jax.random.split(rng, 2)
                params, state = strategy.ask(rng_gen, state, es_params)
                scale_factors, txs, tys = params.T
                new_poses = np.array([hover_sim.sim_new_pose(sf, tx, ty) for sf, tx, ty in params])
                fitness = hover_sim.eval(new_poses, waypoint)
                state = strategy.tell(params, fitness, state, es_params)

            hover_sim.pose = hover_sim.sim_new_pose(*state.best_member)
            optimized_trajectory.append(hover_sim.pose)


        optimized_trajectory = np.array(optimized_trajectory)
        all_trajectories.append((strategy_name, optimized_trajectory))    

    print(all_trajectories)

    print("Saving results...")
    path = "results/"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, trajectory in all_trajectories:
        print(name, trajectory)
        print(f"Similarity Score for {name}: {np.linalg.norm(trajectory[1:]-target_trajectory, ord=2)}")
        np.save(path / f"{name}.npy", trajectory)

    plot_results(path, target_trajectory, all_trajectories)
