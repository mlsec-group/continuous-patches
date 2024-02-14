from simulators.hoverair import HoverSim
from util import bezier_curve
from plot_results import plot_results

from pathlib import Path

import numpy as np

from tqdm import tqdm

import jax
from evosax import Strategies

class BBOptimizer():
    def __init__(self, strategy, target_trajectory, simulator, num_params, num_candidates, num_optim_steps=50):
        self.strategy = strategy
        self.target_trajectory = target_trajectory
        self.simulator = simulator
        self.num_params = num_params
        self.num_candidates = num_candidates
        self.num_optim_steps = num_optim_steps

    def _single_optimize(self, target_pose):
        strategy = Strategies[self.strategy](popsize=self.num_candidates, num_dims=self.num_params)   # popsize => num candidates, num_dims => number of parameters to actually optimize
        es_params = strategy.default_params
        rng = jax.random.PRNGKey(0)
        state = strategy.initialize(rng, es_params)

        for _ in range(self.num_optim_steps):
            rng, rng_gen = jax.random.split(rng, 2)
            params, state = strategy.ask(rng_gen, state, es_params)
            new_poses = np.array([self.simulator.sim_new_pose(sf, tx, ty) for sf, tx, ty in params])
            fitness = self.simulator.eval(new_poses, target_pose)
            state = strategy.tell(params, fitness, state, es_params)

        return state


    def optimze(self):
        optimized_trajectory = []
        optimized_trajectory.append(self.simulator.pose)
        
        optimized_params = []
        for waypoint in tqdm(target_trajectory):
            # optimize params for single desired waypoint
            optim_state = self._single_optimize(waypoint)
            # update simulator pose
            self.simulator.pose = self.simulator.sim_new_pose(*optim_state.best_member)
            # bookkeeping
            optimized_trajectory.append(self.simulator.pose)
            optimized_params.append(optim_state.best_member)


        optimized_trajectory = np.array(optimized_trajectory)
        optimized_params = np.array(optimized_params)

        return optimized_trajectory, optimized_params


if __name__ == '__main__':
    
    control_points_bezier = np.array([[0, 0, 0], [1, 3, 0.2], [2, -1, 0.6], [3, 2, 1]])
    target_trajectory = bezier_curve(control_points_bezier, 20)

    # strategy_names = ["SimpleES", "SimpleGA", "PSO", "DE", "CMA_ES", "Sep_CMA_ES",
    #            "Full_iAMaLGaM", "Indep_iAMaLGaM", "MA_ES", "LM_MA_ES",
    #            "RmES", "GLD", "SimAnneal", "GESMR_GA", "SAMR_GA"]

    strategy_names = ['CMA_ES']#, 'PSO', "Sep_CMA_ES"]
    
    results = []
    
    for strategy_name in tqdm(strategy_names):
        print(strategy_name)
        # reset simulator
        hover_sim = HoverSim()

        # initialize optimizer
        optimizer = BBOptimizer(strategy_name, target_trajectory, hover_sim, 
                                num_params=3, num_candidates=20, num_optim_steps=50)

        # optmize for whole trajectory
        optimized_trajectory, optimized_params = optimizer.optimze()

        # bookkeeping
        results.append((strategy_name, optimized_trajectory, optimized_params))    

    print("Saving results...")
    path = "results/"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, trajectory, params in results:
        # print(name, trajectory)
        print(f"Similarity Score for {name}: {np.linalg.norm(trajectory[1:]-target_trajectory, ord=2)}")
        np.save(path / f"{name}.npy", {'trajectory': trajectory, 'params': params})

    plot_results(path, target_trajectory, results)
