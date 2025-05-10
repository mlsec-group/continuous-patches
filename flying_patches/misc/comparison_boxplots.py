import numpy as np
import glob
import matplotlib.pyplot as plt
import sys, os
import torch
import argparse
from pathlib import Path

sys.path.insert(0,'src/')
# sys.path.insert(0,'pulp-frontnet/PyTorch/')

from util import load_quantized

def load_images(path, idcs):
    images = []
    for img in sorted(glob.glob(str(path) + "/raw/*.npy"))[idcs[0]:idcs[-1]]:
        images.append(np.load(img))
    return np.array(images)

def get_poses(path, idcs):
    data = np.load(path / 'data.npy')
    return data[idcs[0]:idcs[-1], :3]

def predict_poses(images, model):
    images_t = torch.tensor(images).unsqueeze(1).float()
    poses_t = torch.stack(model(images_t))[:3].squeeze(2).mT
    return poses_t.detach().numpy()

def calc_MSE(poses, target):
    return ((poses - target)**2).mean(axis=1)

def calc_boxplot_data(path, idcs, target, model):
    poses_fp = get_poses(path, idcs)
    mse_fp = calc_MSE(poses_fp, target)

    images_q = load_images(path, idcs)
    poses_q = predict_poses(images_q, model)

    mse_q = calc_MSE(poses_q, target)

    return mse_fp, mse_q


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_idx', type=int, nargs='+', action='append')
    parser.add_argument('--clear_idx', type=int, nargs='+', action='append')
    parser.add_argument('--targets', type=int, nargs='+', action='append')
    parser.add_argument('--path_patch', type=str, default='~/')
    parser.add_argument('--path_clear', type=str, default='~/')
    parser.add_argument('--path_model', type=str, default='~/')
    parser.add_argument('--path_figure', type=str, default='./')
    parser.add_argument('--name_figure', type=str, default='plot')
    args = parser.parse_args()


    path_patch = Path(args.path_patch)
    path_clear = Path(args.path_clear)
    path_model = Path(args.path_model)
    path_figure = Path(args.path_figure)
    
    os.makedirs(path_figure, exist_ok=True)

    name = args.name_figure

    patch_idx_left = args.patch_idx[0]
    patch_idx_right = args.patch_idx[1]

    clear_idx_left = args.clear_idx[0]
    clear_idx_right = args.clear_idx[1]

    target_left = args.targets[0]
    target_right = args.targets[1]

    print(target_left, target_right)

    device = torch.device('cpu')
    model = load_quantized(path_model, device)
    model.eval()

    patch_left_fp, patch_left_q = calc_boxplot_data(path_patch, patch_idx_left, target_left, model)
    patch_right_fp, patch_right_q  = calc_boxplot_data(path_patch, patch_idx_left, target_right, model)

    clear_left_fp, clear_left_q = calc_boxplot_data(path_clear, clear_idx_left, target_left, model)
    clear_right_fp, clear_right_q  = calc_boxplot_data(path_clear, clear_idx_left, target_right, model)


    fig, axs = plt.subplots(1, 2, figsize=(11,3))

    axs[0].boxplot([clear_left_fp, clear_left_q, patch_left_fp, patch_left_q], labels=['no patch', 'no patch, q', 'with patch', 'with patch, q'])
    axs[1].boxplot([clear_right_fp, clear_right_q, patch_right_fp, patch_right_q], labels=['no patch', 'no patch, q', 'with patch', 'with patch, q'])
    for ax in axs:
        ax.set_yticks(np.linspace(0.26, 0.55, num=6))
        ax.grid(axis='y')

    # fig.title('Attack on full-precision network')
    # axs[0].set_ylabel(r'MSE to $\bar\mathbf{p}^h_{1}$')
    axs[0].set_ylabel('MSE to [1, 1, 0]')
    axs[0].set_title('UAV visible to the left')
    # axs[1].set_ylabel(r'MSE to $\bar\mathbf{p}^h_{2}$')
    axs[1].set_ylabel('MSE to [1, -1, 0]')
    axs[1].set_title('UAV visible to the right')
    print(f"Saving to file {str(path_figure)+'/'+name+'.jpg'} ....")
    plt.savefig(str(path_figure)+'/'+name+'.jpg', dpi=200)


    #     model_path = '/home/pia/Documents/Coding/adversarial_frontnet/Results/160x32/Export/Frontnet.onnx'
