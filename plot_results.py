import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def plot_results(path, target_trajectory, optimized_trajectory):

    with PdfPages(Path(path) / 'result.pdf') as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], label='Desired Trajectory')
        ax.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], optimized_trajectory[:, 2], label='Optimized')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)