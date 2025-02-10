import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np

from util import get_yaw

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

if __name__ == '__main__':

    results = np.load('results/poses.npy')

    timestamps = results[:, 0]
    timestamps = (timestamps - timestamps[0])  # convert to seconds

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
    plt.show()
    plt.close()