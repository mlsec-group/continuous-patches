import numpy as np


# load all losses_train.npy files from results/test/frontnet/80x80/[0-9]*/losses_train.npy
import glob
from pathlib import Path

path = Path('results/test/frontnet/80x80/')
file_paths = list(path.glob('[0-9]*/losses_test.npy'))

losses = []
for file_path in file_paths:
    losses.append(np.load(file_path)[-1])


print(f"Mean loss: {np.mean(losses)}")