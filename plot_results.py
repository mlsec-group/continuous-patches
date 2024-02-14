import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def plot_results(path, target_trajectory, optimized_trajectories):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    with PdfPages(path / 'result.pdf') as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], label='Desired Trajectory')
        for name, trajectory, _ in optimized_trajectories:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)