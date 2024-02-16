import jax
import jax.numpy as jnp

from evosax import OpenES, CMA_ES, ParameterReshaper, NetworkMapper, FitnessShaper
# from evosax.problems import VisionFitness

import chex
from simulators.hoverair import HoverSim

from util import bezier_curve

control_points_bezier = jnp.array([[0, 0, 0], [1, 3, 0.2], [2, -1, 0.6], [3, 2, 1]])
target_trajectory = bezier_curve(control_points_bezier, 20)

class Evaluator():
    def __init__(self, simulator):
        self.simulator = simulator
        self.n_devices = jax.local_device_count()

    def set_apply_fn(self, network):
        """Set the network forward function."""
        self.network = network
        self.rollout_pop = jax.vmap(self.rollout_ffw, in_axes=(None, 0))
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout = self.rollout_pmap
            print(
                f"VisionFitness: {self.n_devices} devices detected. Please make"
                " sure that the ES population size divides evenly across the"
                " number of devices to pmap/parallelize over."
            )
        else:
            self.rollout = jax.jit(self.rollout_vmap)

    def rollout_vmap(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ):
        """Vectorize rollout. Reshape output correctly."""
        loss, params = self.rollout_pop(rng_input, network_params)
        loss_re = loss.reshape(-1, 1)
        params_re = params.reshape(-1, 1)
        # acc_re = acc.reshape(-1, 1)
        return loss_re, params#, acc_re

    def rollout_pmap(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1))
        loss_dev, params = jax.pmap(self.rollout_pop)(
            keys_pmap, network_params
        )
        loss_re = loss_dev.reshape(-1, 1)
        params_re = params.reshape(-1, 1)
        # acc_re = acc_dev.reshape(-1, 1)
        return loss_re, params_re

    def rollout_ffw(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Evaluate a network on a supervised learning task."""
        rng_net, rng_sample = jax.random.split(rng_input)
        current_waypoint = self.simulator.target_trajectory[self.simulator.current_idx]
        desired_vector = current_waypoint - self.simulator.pose
        sf, tx, ty = self.network(network_params, desired_vector, rng_net)
        new_pose = self.simulator.sim_new_pose(sf, tx, ty)
        loss = self.simulator.eval(new_pose, current_waypoint)

        # X, y = self.dataloader.sample(rng_sample)
        # y_pred = self.network(network_params, X, rng_net)
        # loss, acc = loss_and_acc(y_pred, y, self.num_classes)
        # Return negative loss to maximize!
        return loss, jnp.array([sf, tx, ty])
    
        
    # def get_new_pose(self, rng_net: chex.PRNGKey, network_params: chex.ArrayTree,
    # ) -> chex.ArrayTree:
    #     current_waypoint = self.simulator.target_trajectory[self.simulator.current_idx]
    #     desired_vector = current_waypoint - self.simulator.pose
    #     sf, tx, ty = self.network(network_params, desired_vector, rng_net)
    #     new_pose = self.simulator.sim_new_pose(sf, tx, ty)

    #     return new_pose

rng = jax.random.PRNGKey(0)

# The CNN architecture uses two differently sized (kernel etc.) conv blocks
network = NetworkMapper["MLP"](
            num_hidden_units = 32,
            num_hidden_layers = 2,
            num_output_units = 3,
            hidden_activation = "relu"
        )
pholder = jnp.zeros((3,))
params = network.init(
    rng,
    x=pholder,
    rng=rng,
)

param_reshaper = ParameterReshaper(params)
fit_shaper = FitnessShaper(centered_rank=True, w_decay=0.1)

hover_sim = HoverSim(target_trajectory)
evaluator = Evaluator(hover_sim)
evaluator.set_apply_fn(network.apply)

strategy = OpenES(popsize=100, num_dims=param_reshaper.total_params, opt_name="adam")
# Update basic parameters of PGPE strategy
es_params = strategy.default_params

state = strategy.initialize(rng, es_params)

for epoch in range(500):
    rng, rng_ask, rng_eval, rng_update = jax.random.split(rng, 4)
    x, state = strategy.ask(rng_ask, state, es_params)
    # print(x.shape)
    reshaped_x = param_reshaper.reshape(x)
    # print(reshaped_x)
    # pred = network.apply(reshaped_x, pholder, rng_eval)
    loss, opt_params = evaluator.rollout(rng_eval, reshaped_x)
    # print(loss, opt_params)
    fitness_reshaped = fit_shaper.apply(x, loss)
    # print(fitness_reshaped)
    state = strategy.tell(x, fitness_reshaped, state)

    current_best_params = param_reshaper.reshape(state.best_member.reshape(1, -1))
    _, opt_params = evaluator.rollout(rng_update, current_best_params)
    # current_waypoint = evaluator.simulator.target_trajectory[evaluator.simulator.current_idx]
    # desired_vector = current_waypoint - evaluator.simulator.pose

    # opt_params = network.apply(param_reshaper.reshape(state.best_member), 
    #                                desired_vector, rng_update)
    new_pose = evaluator.simulator.sim_new_pose(*opt_params[0])
    evaluator.simulator.update(new_pose)

    if (epoch + 1) % 50 == 0:
        print(f"OpenES - # GEN: {epoch+1} | Fitness: {state.best_fitness}")