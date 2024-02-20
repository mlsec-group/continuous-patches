import jax
import jax.numpy as jnp

from evosax import OpenES, CMA_ES, PGPE, ParameterReshaper, NetworkMapper, FitnessShaper
from evosax.utils import ESLog
# from evosax.problems import VisionFitness

import chex
from simulators.hoverair import HoverSim

from util import bezier_curve

import numpy as np

control_points_bezier = np.array([[0, 0, 0.0], [1, 3, 0.4], [2, -1, 0.8], [3, 2, 1]])
target_trajectory = np.array(bezier_curve(control_points_bezier, 20))

def calc_tracking_error(params_unshaped):
    sim = HoverSim(target_trajectory)
    rng = jax.random.PRNGKey(0)
    # optimized_trajectory = []
    params = param_reshaper.reshape_single(params_unshaped)
    distances = []

    for desired_pose in sim.target_trajectory:
        rng, rng_eval = jax.random.split(rng, 2)
        # desired_vector = desired_pose - sim.pose
        out = network.apply(params, jnp.array([*sim.pose, *desired_pose]))
        # out = network.apply(params, desired_vector)
        new_pose = sim.sim_new_pose(out)
        # print(new_pose)
        distances.append(jnp.linalg.norm(new_pose-desired_pose, ord=2))
        sim.update(new_pose)
        # print(sim.pose)
        # optimized_trajectory.append(new_pose)

    distances = jnp.array(distances)
    # print(distances.shape)
    tracking_error = jnp.sum(distances)
    loss = tracking_error + jnp.max(distances)
    return loss

    # optimized_trajectory = np.array(optimized_trajectory)

rng = jax.random.PRNGKey(0)

network = NetworkMapper["MLP"](
            num_hidden_units = 5,
            num_hidden_layers = 2,
            num_output_units = 3,
            hidden_activation = "relu"
        )

pholder = jnp.zeros((6,))
params = network.init(
    rng,
    x=pholder,
    rng=rng,
)

# print(params)
param_reshaper = ParameterReshaper(params)
# flattened_params = param_reshaper.flatten_single(params)
# reshaped_params = param_reshaper.reshape_single(flattened_params)
# out = network.apply(reshaped_params, pholder)
# print(out)




hover_sim = HoverSim(target_trajectory)

vmap_network_apply = jax.vmap(network.apply, in_axes=(0, None))
vmap_new_poses = jax.vmap(hover_sim.sim_new_pose, in_axes=0)
vmap_loss = jax.vmap(calc_tracking_error, in_axes=0)

# strategy = CMA_ES(popsize=10, num_dims=param_reshaper.total_params, elite_ratio=0.1, mean_decay=0.1)
strategy = OpenES(popsize=100, num_dims=param_reshaper.total_params, opt_name="adam", lrate_init=0.1)
# strategy = PGPE(popsize=100, num_dims=param_reshaper.total_params,
#                 elite_ratio=0.1, opt_name="adam")

es_params = strategy.default_params

state = strategy.initialize(rng)

fit_shaper = FitnessShaper(centered_rank=False,
                           w_decay=0.1,
                           maximize=False)

es_logging = ESLog(param_reshaper.total_params,
                   num_generations=500,
                   top_k=5,
                   maximize=False)
log = es_logging.initialize()

for i in range(500):
    rng, rng_ask = jax.random.split(rng, 2)
    params, state = strategy.ask(rng_ask, state, es_params)
    # print(params.shape)
    reshaped_params = param_reshaper.reshape(params)
    # print(reshaped_params.)
    desired_vector = hover_sim.target_trajectory[hover_sim.current_idx] - hover_sim.pose
    # print(desired_vector)
    # out = [network.apply(reshaped_param, desired_vector) for reshaped_param in reshaped_params]


    #outputs_batch = vmap_network_apply(reshaped_params, desired_vector)
    # print(outputs_batch.shape)
    #new_poses_batch = vmap_new_poses(outputs_batch)
    # print(new_poses_batch.shape)
    #losses = vmap_loss(new_poses_batch, hover_sim.target_trajectory[hover_sim.current_idx])
    # print(reshaped_params)
    losses = vmap_loss(params)#jnp.array([calc_tracking_error(param_set) for param_set in params])
    # print(losses)
    fitness_reshaped = fit_shaper.apply(params, losses)
    # print(fitness_reshaped.shape)
    state = strategy.tell(params, fitness_reshaped, state)
    log = es_logging.update(log, params, losses)

    if (i+1) % 10 == 0:
        # print("best fitness: ", state.best_fitness)
        print("Generation: ", i+1, "Performance: ", log["log_top_1"][i])
    # best_params = param_reshaper.reshape_single(state.best_member)
    # out = network.apply(best_params, desired_vector)
    # new_pose = hover_sim.sim_new_pose(out)
    # hover_sim.update(new_pose)


logger_fig, logger_ax = es_logging.plot(log, "Train loss")
logger_fig.savefig("results/openes_losses.jpg", dpi=120)

# eval
optimized_trajectory = []
hover_sim.reset()
current_best_params = param_reshaper.reshape_single(state.best_member)

for desired_pose in hover_sim.target_trajectory:
    rng, rng_eval = jax.random.split(rng, 2)
    # desired_vector = desired_pose - hover_sim.pose
    out = network.apply(current_best_params, jnp.array([*hover_sim.pose, *desired_pose]))
    # out = network.apply(current_best_params, desired_vector)
    new_pose = hover_sim.sim_new_pose(out)
    hover_sim.update(new_pose)
    optimized_trajectory.append(new_pose)

optimized_trajectory = np.array(optimized_trajectory)

print("Final Tracking error: ", np.linalg.norm(optimized_trajectory-target_trajectory, ord=2))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], label='Desired Trajectory')
ax.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], optimized_trajectory[:, 2], label='optimized')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
# pdf.savefig(fig)
plt.savefig("results/debugging_openes.jpg", dpi=120)
plt.show()
# plt.close(fig)


# class Evaluator():
#     def __init__(self, simulator):
#         self.simulator = simulator
#         self.n_devices = jax.local_device_count()

#     def set_apply_fn(self, network):
#         """Set the network forward function."""
#         self.network = network
#         self.rollout_pop = jax.vmap(self.rollout_ffw, in_axes=(None, 0))
#         # pmap over popmembers if > 1 device is available - otherwise pmap
#         if self.n_devices > 1:
#             self.rollout = self.rollout_pmap
#             print(
#                 f"VisionFitness: {self.n_devices} devices detected. Please make"
#                 " sure that the ES population size divides evenly across the"
#                 " number of devices to pmap/parallelize over."
#             )
#         else:
#             self.rollout = jax.jit(self.rollout_vmap)

#     def rollout_vmap(
#         self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
#     ):
#         """Vectorize rollout. Reshape output correctly."""
#         loss, params = self.rollout_pop(rng_input, network_params)
#         loss_re = loss.reshape(-1, 1)
#         params_re = params.reshape(-1, 1)
#         # acc_re = acc.reshape(-1, 1)
#         return loss_re, params#, acc_re

#     def rollout_pmap(
#         self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
#     ):
#         """Parallelize rollout across devices. Split keys/reshape correctly."""
#         keys_pmap = jnp.tile(rng_input, (self.n_devices, 1))
#         loss_dev, params = jax.pmap(self.rollout_pop)(
#             keys_pmap, network_params
#         )
#         loss_re = loss_dev.reshape(-1, 1)
#         params_re = params.reshape(-1, 1)
#         # acc_re = acc_dev.reshape(-1, 1)
#         return loss_re, params_re

#     def rollout_ffw(
#         self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree,
#     ) -> chex.ArrayTree:
#         """Evaluate a network on a supervised learning task."""
#         rng_net, rng_sample = jax.random.split(rng_input)
#         current_waypoint = self.simulator.target_trajectory[jax.random.randint(rng_sample, (1,), minval=0, maxval=len(self.simulator.target_trajectory)-1)]
#         # print("curr_pose: ", self.simulator.pose, self.simulator.pose.dtype)
#         # print("curr desired waypoit: ", current_waypoint, current_waypoint.dtype)
#         # print(self.simulator.target_trajectory)
#         desired_vector = jnp.subtract(current_waypoint, self.simulator.pose)

#         sf, tx, ty = self.network(network_params, desired_vector, rng_net)
#         new_pose = self.simulator.sim_new_pose(sf, tx, ty)
#         loss = self.simulator.eval(new_pose, current_waypoint)
#         # self.simulator.update(new_pose)

#         # X, y = self.dataloader.sample(rng_sample)
#         # y_pred = self.network(network_params, X, rng_net)
#         # loss, acc = loss_and_acc(y_pred, y, self.num_classes)
#         # Return negative loss to maximize!
#         return loss, jnp.array([sf, tx, ty])
    
        
#     # def get_new_pose(self, rng_net: chex.PRNGKey, network_params: chex.ArrayTree,
#     # ) -> chex.ArrayTree:
#     #     current_waypoint = self.simulator.target_trajectory[self.simulator.current_idx]
#     #     desired_vector = current_waypoint - self.simulator.pose
#     #     sf, tx, ty = self.network(network_params, desired_vector, rng_net)
#     #     new_pose = self.simulator.sim_new_pose(sf, tx, ty)

#     #     return new_pose

# rng = jax.random.PRNGKey(0)

# # The CNN architecture uses two differently sized (kernel etc.) conv blocks
# network = NetworkMapper["MLP"](
#             num_hidden_units = 32,
#             num_hidden_layers = 2,
#             num_output_units = 3,
#             hidden_activation = "relu"
#         )
# pholder = jnp.zeros((3,))
# params = network.init(
#     rng,
#     x=pholder,
#     rng=rng,
# )

# param_reshaper = ParameterReshaper(params)
# fit_shaper = FitnessShaper(centered_rank=False, w_decay=0.1)

# hover_sim = HoverSim(target_trajectory)
# evaluator = Evaluator(hover_sim)
# evaluator.set_apply_fn(network.apply)

# strategy = OpenES(popsize=100, num_dims=param_reshaper.total_params, opt_name="adam")
# # Update basic parameters of PGPE strategy
# es_params = strategy.default_params

# state = strategy.initialize(rng, es_params)

# for epoch in range(10):
#     rng, rng_ask, rng_eval, rng_update = jax.random.split(rng, 4)
#     x, state = strategy.ask(rng_ask, state, es_params)
#     # print(x.shape)
#     reshaped_x = param_reshaper.reshape(x)
#     # print(reshaped_x)
#     # pred = network.apply(reshaped_x, pholder, rng_eval)
#     loss, opt_params = evaluator.rollout(rng_eval, reshaped_x)
#     # print(loss, opt_params)
#     fitness_reshaped = fit_shaper.apply(x, loss.mean(axis=1))
#     # print(fitness_reshaped)
#     state = strategy.tell(x, fitness_reshaped, state)

#     current_best_params = param_reshaper.reshape(state.mean.reshape(1, -1))
#     _, opt_params = evaluator.rollout(rng_update, current_best_params)
#     # current_waypoint = evaluator.simulator.target_trajectory[evaluator.simulator.current_idx]
#     # desired_vector = current_waypoint - evaluator.simulator.pose

#     # opt_params = network.apply(param_reshaper.reshape(state.best_member), 
#     #                                desired_vector, rng_update)
#     new_pose = evaluator.simulator.sim_new_pose(*opt_params[0])
#     evaluator.simulator.update(new_pose)
#     print(new_pose)
#     print(evaluator.simulator.pose)
#     # print(evaluator.simulator.pose, evaluator.simulator.current_idx)

#     if (epoch + 1) % 1 == 0:
#         print(f"OpenES - # GEN: {epoch+1} | Fitness: {state.best_fitness}")

# # eval
# optimized_trajectory = []
# evaluator.simulator.reset()
# current_best_params = param_reshaper.reshape(state.best_member.reshape(1, -1))

# for desired_pose in evaluator.simulator.target_trajectory:
#     rng, rng_eval = jax.random.split(rng, 2)
#     _, opt_params = evaluator.rollout(rng_update, current_best_params)
#     new_pose = evaluator.simulator.sim_new_pose(*opt_params[0])
#     optimized_trajectory.append(new_pose)

# optimized_trajectory = np.array(optimized_trajectory)

# print("Final Tracking error: ", np.linalg.norm(optimized_trajectory-target_trajectory, ord=2))

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], label='Desired Trajectory')
# ax.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], optimized_trajectory[:, 2], label='optimized')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# # pdf.savefig(fig)
# plt.show()
# # plt.close(fig)