import yaml
from multiprocessing import Pool
import argparse
import tempfile
from pathlib import Path
import subprocess
import signal
import os

def run_attack(settings):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname) / 'settings.yaml'
        with open(filename, 'w') as f:
            yaml.dump(settings, f)

        path = Path(settings['path']) 
        os.makedirs(path, exist_ok = True)
        with open(path / "stderr.txt", "w") as stderr:
            with open(path / "stdout.txt", "w") as stdout:
                subprocess.run(["python3", "src/attacks.py", "--file", filename], stderr=stderr, stdout=stdout)

def on_sigterm():
    # kill the whole process group
    os.killpg(os.getpid(), signal.SIGTERM)

def exp(my_exp):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='exp1.yaml')
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('--quantized', action='store_true')
    parser.add_argument('-j', type=int, default=3)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--mode', nargs='+', default=['all']) # mode can be 'all' or ['fixed', 'joint', 'split', 'hybrid']
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, on_sigterm)

    # SETTINGS
    with open(args.file) as f:
        base_settings = yaml.load(f, Loader=yaml.FullLoader)


    if 'all' in args.mode:
        modes = ['fixed', 'joint', 'split', 'hybrid']
    elif set(args.mode) & set(['fixed', 'joint', 'split', 'hybrid']):
        modes = args.mode
    else:
        print("Mode can be either 'all' or a combination from ['fixed', 'joint', 'split', 'hybrid']")
        
    all_settings = my_exp.create_settings(base_settings, args.trials, modes, args.quantized)

    if not args.norun:
        # start 4 worker processes
        with Pool(processes=args.j) as pool:
            for i in pool.imap_unordered(run_attack, all_settings):
                pass
    else:            
        my_exp.stats(all_settings)

if __name__ == '__main__':
    pass