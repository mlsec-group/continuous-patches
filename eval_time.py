import torch
import torch.nn as nn
import numpy as np

from diffusion.example_unet import UNet
from eval_diffusion import get_alpha_betas, sample

from tqdm import trange
from time import time

import argparse

def get_time_diffusion(n_runs, n_samples):

    all_times = []
    for i in trange(n_runs):
        t_start = time()
        model = UNet(in_size=1, out_size=1, device=device)
        model.to(device)

        model.load_state_dict(torch.load('conditioned_unet_80x80_1000_3256i_255.pth', map_location=device))
        model.eval()
        x = np.random.uniform(0,2,n_samples)
        y = np.random.uniform(-1,1,n_samples,)
        z = np.random.uniform(-0.5,0.5,n_samples,)
        targets = torch.tensor(np.stack((x, y, z)).T, dtype=torch.float32)

        if len(targets.shape) < 2.:
            targets = targets.unsqueeze(0)
        
        with torch.no_grad():
                samples = sample(model, targets, device, n_samples=n_samples, patch_size=[80, 80], n_steps=1_000)

        all_times.append(time()-t_start)
        del model

    all_times = np.array(all_times)
    np.save(f'eval/comp_time_diffusion_{n_samples}.npy', all_times)

    print("Computation time mean: ", np.mean(all_times), ', std: ', np.std(all_times))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', type=int)
    args = parser.parse_args()

    np.random.seed(2562)
    torch.manual_seed(2562)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_runs = 100

    get_time_diffusion(n_runs, n_samples=args.samples)