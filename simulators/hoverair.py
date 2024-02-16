import numpy as np
from matplotlib import pyplot as plt

import jax.numpy as jnp


class HoverSim():
    def __init__(self, target_trajectory):
        
        self.pose = np.array([0., 0., 0.])#, 0.]) # x, y, z, yaw
        self.target_trajectory = target_trajectory
        self.current_idx = 0
    
    def sim_new_pose(self, params):
        # We assume that the Hoverair will follow any image of a human face.
        # The function takes in relevant parts of the transformation matrix of an image of a human
        # and simulate a possible reaction of the Hoverair.
        # A positive change of the scale factor will cause to a negative change in x direction.
        # A positive change in ty will cause a positive change in y direction.
        # A positive change in tx will cause a positive change in z direction.
        # We assume that a the Hoverair will always keep safety-distance of 1 m to the target.
        # These parameters could later be used to calculate the realtive pose of the image as in the flying_adversarial_patch paper.
        scale_factor, tx, ty = params
        action = jnp.array([(1 - scale_factor), ty, tx])
        # print(action)
        new_pose = self.pose + action * 0.5 # simulate reaction after 0.5 secs
        new_pose += np.random.normal(0.0, 0.1, (3,))
        return new_pose
    
    def update(self, pose):
        self.pose = pose
        self.current_idx += 1
        if self.current_idx >= len(self.target_trajectory):
            self.reset()

    def reset(self):
        self.pose = jnp.array([0., 0., 0.])
        self.current_idx = 0
    
    def eval(self, pose, desired_pose):
        # print(pose.shape, desired_pose.shape)
        l2_distances = jnp.linalg.norm((pose - desired_pose), ord=2)#, axis=1)
        return l2_distances

if __name__ == '__main__':

    hover = HoverSim()
    all_poses = []
    all_poses.append(hover.pose)

    # check for random parameters
    for _ in range(1000):
        scale_factor = np.random.uniform(0.01, 1.5)
        tx = np.random.normal()
        ty = np.random.normal()
        hover.pose = hover.sim_new_pose(scale_factor, tx, ty)
        all_poses.append(hover.pose)
    
    all_poses = np.array(all_poses)
    colors = np.linspace(0, 1, len(all_poses))

    fig, ax = plt.subplots(1,1)
    ax.scatter(all_poses.T[1], all_poses.T[0], c=colors, cmap='Spectral')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    plt.show()