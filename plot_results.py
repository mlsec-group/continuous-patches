import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np

from util import get_yaw

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation


# def plot_results(path, target_trajectory, optimized_trajectories):

#     path = Path(path)
#     path.mkdir(parents=True, exist_ok=True)

#     with PdfPages(path / 'result.pdf') as pdf:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], label='Desired Trajectory')
#         for name, trajectory, _ in optimized_trajectories:
#             ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=name)
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel('z')
#         ax.legend()
#         pdf.savefig(fig)
#         plt.close(fig)

# # copied from https://github.com/rdeits/meshcat-python/blob/master/src/meshcat/geometry.py#L83
# # since the latest pip-version doesn't include it yet
class Plane(g.Geometry):

    def __init__(self, width=1, height=1, widthSegments=1, heightSegments=1):
        super(Plane, self).__init__()
        self.width = width
        self.height = height
        self.widthSegments = widthSegments
        self.heightSegments = heightSegments

    def lower(self, object_data):
        return {
            u"uuid": self.uuid,
            u"type": u"PlaneGeometry",
            u"width": self.width,
            u"height": self.height,
            u"widthSegments": self.widthSegments,
            u"heightSegments": self.heightSegments,
        }


if __name__ == '__main__':

    results = np.load('results/single_patch/2/poses.npy')

    timestamps = results[:, 0]
    print(timestamps)
    # timestamps = (timestamps - timestamps[0])  # convert to seconds

    positions = results[:, 1:4]
    rotations = results[:, 4:]

    yaws = np.array([get_yaw(rot) for rot in rotations])



    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(timestamps, positions[:, 0])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('X (m)')
    axs[1].plot(timestamps, positions[:, 1])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Y (m)')
    axs[2].plot(timestamps, positions[:, 2])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z (m)')
    axs[3].plot(timestamps, np.degrees(yaws))
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Yaw (deg)')
    plt.tight_layout()
    plt.savefig('results/single_patch/2/plot_pose_over_time.jpg', dpi=200)
    # plt.show()
    plt.close()


    # plot trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #set ranges for axes
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 1.5)
    plt.savefig('results/single_patch/2/plot_trajectory_3d.jpg', dpi=200)
    # plt.show()
    plt.close()

    # visualize in meshcat
    vis = meshcat.Visualizer()
    # vis.open()

    vis["/Cameras/default"].set_transform(
        tf.translation_matrix([0, 0, 0]).dot(
        tf.euler_matrix(0, np.radians(-30), -np.pi/2)))

    vis["/Cameras/default/rotated/<object>"].set_transform(
        tf.translation_matrix([1, 0, 0]))

    vis["Quadrotor"].set_object(
        g.StlMeshGeometry.from_file('data/cf2_assembly.stl'))

    vis["projector"].set_object(Plane())

    # # # Little experiment to support hyperplanes specified by n and a; This doesn't include all corner cases yet
    # n = np.array([0,0,1])
    # a = -1

    # xd = np.array([0,1,0])
    # zd = n
    # yd = np.cross(xd, zd)
    # R = tf.identity_matrix()
    # R[:3,0] = xd
    # R[:3,1] = yd
    # R[:3,2] = zd
    # p = np.array([0,0,-a/n[2]])

    p = np.array([2, -0.3, 1])
    R = tf.identity_matrix()
    # rotate R 90 degrees around y-axis
    R[:3, 0] = [0, 0, 1]
    R[:3, 1] = [0, 1, 0]
    R[:3, 2] = [-1, 0, 0]


    anim = Animation()

    for idx, timestamp in enumerate(timestamps):
        with anim.at_frame(vis, timestamp) as frame:
            frame["Quadrotor"].set_transform(
                tf.translation_matrix([*positions[idx]]).dot(
                    tf.quaternion_matrix(rotations[idx])))
            frame["projector"].set_transform(tf.translation_matrix(p).dot(R))
        # time.sleep(0.1)
    vis.set_animation(anim)
    res = vis.static_html()
    # save to a file
    Path("results/single_patch/2/").mkdir(exist_ok=True)
    with open(Path("results/single_patch/2") / "meshcat_example.html", "w") as f:
        f.write(res)