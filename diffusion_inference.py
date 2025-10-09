import torch 
import numpy as np
from diffusion.diffusion_model import DiffusionModel

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion_model = DiffusionModel(device, lr=1e-5)
diffusion_model.load('results/diffusion_training/trained_model.pth')



time_start = time.time()
n_samples = 1
sf = np.random.uniform(0.4,0.8,n_samples)
tx = np.random.uniform(0.,1.,n_samples)
ty = np.random.uniform(0.,1.,n_samples)
#TODO: generate target based on target trajectory
x = np.random.uniform(0,2,n_samples)
y = np.random.uniform(-1,1,n_samples,)
z = np.random.uniform(-0.5,0.5,n_samples,)

r_targets = torch.tensor(np.stack((sf, tx, ty, x, y, z)).T, dtype=torch.float32)

samples = diffusion_model.sample(n_samples, r_targets, device, patch_size=[80,80], n_steps=100).detach().to('cpu').numpy()
patch = samples[0, 0] * 255.

print("Inference time: ", time.time()-time_start)