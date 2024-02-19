from simulators.hoverair import HoverSim
from util import bezier_curve
from plot_results import plot_results

from pathlib import Path

import numpy as np


from tqdm import tqdm, trange

import jax
import jax.numpy as jnp
from evosax import Strategies, NetworkMapper, ParameterReshaper, FitnessShaper

class BBOptimizer():
    def __init__(self, strategy_name, target_trajectory, simulator, num_params, num_candidates, num_optim_steps=50):
        # self.strategy_name = strategy_name
        self.rng = jax.random.PRNGKey(0)
        self.target_trajectory = target_trajectory
        self.simulator = simulator

        self.num_params = num_params
        self.num_candidates = num_candidates
        self.num_optim_steps = num_optim_steps

        self.policy = NetworkMapper["MLP"](
            num_hidden_units = 32,
            num_hidden_layers = 2,
            num_output_units = 3,
            hidden_activation = "relu"
        )

        placeholder_input = jnp.zeros((3,))  # input to the policy will be the vector from current to desired pose
        pholder = jnp.zeros((3,))
        params = self.policy.init(
                    self.rng,
                    x=pholder,
                    rng=self.rng,
                )

        self.param_reshaper = ParameterReshaper(params)
        
        # print("Test reshaper")
        # self.param_reshaper.reshape(np.zeros(self.param_reshaper.total_params,))

        self.strategy = Strategies[strategy_name](popsize=20, num_dims=self.param_reshaper.total_params)   # popsize => num candidates, num_dims => number of parameters to actually optimize
        self.es_params = self.strategy.default_params
        
        self.fit_shaper = FitnessShaper(w_decay=0.1)

        

    # def _single_optimize(self, target_pose):
    #     strategy = Strategies[self.strategy](popsize=self.num_candidates, num_dims=self.num_params)   # popsize => num candidates, num_dims => number of parameters to actually optimize
    #     es_params = strategy.default_params
    #     rng = jax.random.PRNGKey(0)
    #     state = strategy.initialize(rng, es_params)
    #     # print(state)

    #     for _ in range(self.num_optim_steps):
    #         rng, rng_gen = jax.random.split(rng, 2)
    #         params, state = strategy.ask(rng_gen, state, es_params)
    #         # print(state)
    #         new_poses = np.array([self.simulator.sim_new_pose(sf, tx, ty) for sf, tx, ty in params])
    #         fitness = self.simulator.eval(new_poses, target_pose)
    #         state = strategy.tell(params, fitness, state, es_params)

    #     return state


    def optimize_strategy(self):
        # optimized_trajectory = []
        # optimized_trajectory.append(self.simulator.pose)
        
        # optimized_params = []

        state = self.strategy.initialize(self.rng, self.es_params)
        step_idx = 0

        for epoch in range(1):
            for waypoint in self.target_trajectory[:1]:
                rng, rng_ask, rng_eval = jax.random.split(self.rng, 3)
                x, state = self.strategy.ask(rng_ask, state, self.es_params)
                print(x.shape)
                reshaped_x = self.param_reshaper.reshape(x)
                print(reshaped_x)
                vector_to_waypoint = waypoint - self.simulator.pose
                sf, tx, ty = jax.jit(self.policy.apply(reshaped_x, vector_to_waypoint))
                print(sf, tx, ty)

        # for epoch in range(200):
        #     for waypoint in target_trajectory:
        #         for _ in range(self.num_optim_steps):
        #             rng, rng_ask = jax.random.split(self.rng, 2)
        #             params, state = self.strategy.ask(rng_ask, state, self.es_params)
        #             print(params.shape, params.dtype)
        #             reshaped_params = self.param_reshaper.reshape(params)
        #             vector_to_waypoint = waypoint - self.simulator.pose
        #             sf, tx, ty = jax.jit(self.policy.apply(reshaped_params, vector_to_waypoint)) 
        #             new_pose = self.simulator.sim_new_pose(sf, tx, ty)#np.array([self.simulator.sim_new_pose(sf, tx, ty) for sf, tx, ty in params])
        #             loss = self.simulator.eval(new_pose, waypoint)
        #             fitness = self.fit_shaper.apply(params, loss)
        #             state = self.strategy.tell(params, fitness, state, self.es_params)
        #             if (step_idx + 1) % 50 == 0:
        #                 print(f"CMA-ES - # GEN: {step_idx+1} | Fitness: {state.best_fitness} | Params: {state.best_member}")
        #             step_idx += 1

        # returns best set of parameters for policy
        return state.best_member

    def evaluate_strategy(self, optimized_params_policy):
        optimized_trajectory = []
        # optimized_trajectory.append(self.simulator.pose)
        
        optimized_params = []
        for waypoint in tqdm(target_trajectory):
            sf, tx, ty = jax.jit(self.policy.apply)(optimized_params_policy, waypoint - self.simulator.pose, self.rng) 
            optimized_params.append([sf, tx, ty])
            new_pose = self.simulator.sim_new_pose(sf, tx, ty)
            optimized_trajectory.append(new_pose)
            self.simulator.pose = new_pose
        
        optimized_trajectory = np.array(optimized_trajectory)
        tracking_error = np.linalg.norm((optimized_trajectory - self.target_trajectory), ord=2)

        optimized_params = np.array(optimized_params)

        return tracking_error, optimized_params, optimized_trajectory


if __name__ == '__main__':
    
    control_points_bezier = np.array([[0, 0, 0], [1, 3, 0.2], [2, -1, 0.6], [3, 2, 1]])
    target_trajectory = bezier_curve(control_points_bezier, 20)[3]

    # strategy_names = ["SimpleES", "SimpleGA", "PSO", "DE", "CMA_ES", "Sep_CMA_ES",
    #            "Full_iAMaLGaM", "Indep_iAMaLGaM", "MA_ES", "LM_MA_ES",
    #            "RmES", "GLD", "SimAnneal", "GESMR_GA", "SAMR_GA"]

    strategy_names = ['CMA_ES']#, 'PSO', "Sep_CMA_ES"]
    
    results = []
    
    for strategy_name in strategy_names:
        print(strategy_name)
        # reset simulator
        hover_sim = HoverSim()

        # initialize optimizer
        optimizer = BBOptimizer(strategy_name, target_trajectory, hover_sim, 
                                num_params=3, num_candidates=1, num_optim_steps=10)


        optimal_policy_params = optimizer.optimize_strategy()

        tracking_error, optimized_patch_params, optimized_trajectory = optimizer.evaluate_strategy(optimal_policy_params)
        print("Tracking error: ", tracking_error)

        # bookkeeping
        results.append((strategy_name, optimized_trajectory, optimized_patch_params))    

    print("Saving results...")
    path = "results/"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, trajectory, params in results:
        # print(name, trajectory)
        # print(f"Similarity Score for {name}: {np.linalg.norm(trajectory[1:]-target_trajectory, ord=2)}")
        np.save(path / f"{name}.npy", {'trajectory': trajectory, 'params': params})

    plot_results(path, target_trajectory, results)
