import jax
import jax.numpy as jnp

from evosax import Strategies, ParameterReshaper, NetworkMapper, FitnessShaper
from evosax.utils import ESLog
# from evosax.problems import VisionFitness

from simulators.hoverair import HoverSim

from util import bezier_curve

import numpy as np

class BBOptimizer():
    def __init__(self, rng, strategy_name, network):
        self.rng = rng
        self.strategy_name = strategy_name
        self.network = network

        pholder = jnp.zeros((3,))
        params = self.network.init(
            rng,
            x=pholder,
            rng=rng,
        )
        
        self.param_reshaper = ParameterReshaper(params)

        # vmap_network_apply = jax.vmap(network.apply, in_axes=(0, None))
        # vmap_new_poses = jax.vmap(hover_sim.sim_new_pose, in_axes=0)
        self.vmap_loss = jax.vmap(self.calc_tracking_error, in_axes=0)

        self. strategy = Strategies[strategy_name](popsize=10, num_dims=self.param_reshaper.total_params, opt_name="adam")

        self.es_params = self.strategy.default_params

        self.fit_shaper = FitnessShaper(centered_rank=False,
                            w_decay=0.1,
                            maximize=False)

        self.es_logging = ESLog(self.param_reshaper.total_params,
                        num_generations=100,
                        top_k=5,
                        maximize=False)
        
        self.log = self.es_logging.initialize()

    def optimize_policy(self):
        state = self.strategy.initialize(self.rng)

        rng = self.rng

        for i in range(100):                                                        # for 100 epochs
            rng, rng_ask = jax.random.split(rng, 2)                                 # split rng key into two variables
            params, state = self.strategy.ask(rng_ask, state, self.es_params)       # ask the strategy for new NN parameter candidates and update the state of the startegy

            losses = self.vmap_loss(params)                                         # calc losses for all candidates in parallel

            fitness_reshaped = self.fit_shaper.apply(params, losses)                # get losses in correct shape for strategy
            state = self.strategy.tell(params, fitness_reshaped, state)             # update strategy with losses
            self.log = self.es_logging.update(self.log, params, losses)             # bookkeeping

            if (i+1) % 10 == 0:
                print("Epoch: ", i+1, "Performance: ", self.log["log_top_1"][i])

        return self.param_reshaper.reshape_single(state.best_member), self.es_logging, self.log  # return best parameter set and logger for bookkeeping

    def calc_tracking_error(self, params_unshaped):
        # random trajectory
        # control_points_bezier = np.random.uniform(-3., 3., (4,3))#np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
        # target_trajectory = np.array(bezier_curve(control_points_bezier, 20))
        
        # single tranjectory
        control_points_bezier = np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
        target_trajectory = np.array(bezier_curve(control_points_bezier, 20))
        
        sim = HoverSim(target_trajectory)                               # initialize new simulator for each parameter candidate
        sim.pose = target_trajectory[0]
        rng = jax.random.PRNGKey(0)

        params = self.param_reshaper.reshape_single(params_unshaped)    # get parameter candidate in correct shape so that they can be used as NN parameters
        distances = []

        for desired_pose in sim.target_trajectory:
            rng, _ = jax.random.split(rng, 2)           # second rng key probably not needed
            desired_vector = desired_pose - sim.pose    # calculate relative vector between desired and current pose
            out = self.network.apply(params, desired_vector) # predict new transformation parameters
            new_pose = sim.sim_new_pose(out)            # get new pose from simulator
            distances.append(jnp.linalg.norm(new_pose-desired_pose, ord=2)) # calculate l2 distance between new and desired pose
            sim.update(new_pose)                        # update the simulator


        distances = jnp.array(distances)
        tracking_error = jnp.sum(distances)
        loss = tracking_error + jnp.max(distances)      # loss is sum of all l2 distances + highest error
        return loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = Path('results/')
    path.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(0)                         # set rng key

    network = NetworkMapper["MLP"](                     # define NN architecture for policy
                num_hidden_units = 5,
                num_hidden_layers = 2,
                num_output_units = 3,
                hidden_activation = "relu"
            )
    

    bb_optimizer = BBOptimizer(rng, 'OpenES', network)  # initialize black box optimizer


    current_best_params, logger, log = bb_optimizer.optimize_policy()   # optimize parameter candidates for the policy
    
    logger_fig, logger_ax = logger.plot(log, "Train loss")              
    logger_fig.savefig(path / "openes_losses.jpg", dpi=120)
    

    # eval
    control_points_bezier = np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
    target_trajectory = np.array(bezier_curve(control_points_bezier, 20))

    optimized_trajectory = []
    distances = []

    hover_sim = HoverSim(target_trajectory)
    hover_sim.reset()

    for desired_pose in hover_sim.target_trajectory:
        rng, rng_eval = jax.random.split(rng, 2)
        desired_vector = desired_pose - hover_sim.pose
        # out = network.apply(current_best_params, jnp.array([*hover_sim.pose, *desired_pose]))
        out = network.apply(current_best_params, desired_vector)
        new_pose = hover_sim.sim_new_pose(out)
        distances.append(jnp.linalg.norm(new_pose-desired_pose, ord=2)) # calculate l2 distance between new and desired pose

        hover_sim.update(new_pose)
        optimized_trajectory.append(new_pose)

    optimized_trajectory = np.array(optimized_trajectory)
    distances = jnp.array(distances)
    final_error = jnp.sum(distances) + jnp.max(distances)      # loss is sum of all l2 distances + highest error

    print("Final error: ", final_error)


    # loss_plot = plt.figure()
    # ax = loss_plot.add_subplot(111)
    # ax.plot(log["log_top_1"])
    # ax.set_xlabel("Number of Generations")
    # ax.set_ylabel("Fitness Score")
    # loss_plot.savefig(path / "loss_best.jpg", dpi=200)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], label='Desired Trajectory')
    ax.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], optimized_trajectory[:, 2], label='optimized')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    # pdf.savefig(fig)
    fig.savefig(path / "final_trajectory.jpg", dpi=200)
    plt.show()
    # plt.close(fig)