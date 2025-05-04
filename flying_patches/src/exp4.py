import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp
from plots import eval_multi_run
from matplotlib import pyplot as plt
import torch


from util import load_patch, load_position, load_dataset, load_quantized, get_transformation
from attacks import calc_eval_loss

class Experiment4:
    def create_settings(self, base_settings, trials, modes, quantized):
        all_settings = []
        base_path = Path(base_settings['path'])
        for i in range(trials):
            for mode in modes:
                s = copy.copy(base_settings)
                s['mode'] = mode
                s['path'] = str(base_path / (mode + str(i)))
                s['quantized'] = quantized
                all_settings.append(s)
        return all_settings

    def stats(self, all_settings):
        # output statistics
        print("Stats for patches calculated on quantized NN:")
        result = defaultdict(list)
        for settings in all_settings:
            p = Path(settings['path'])
            test_losses = np.load(p / 'losses_test.npy')
            # print(settings['mode'], test_losses[-1])
            result[settings['mode']].append(test_losses[-1])

        for k, v in result.items():
            all = np.stack(v)
            print(k, "mean", np.mean(all), "std", np.std(all))

        # change settings to match latex
        plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": "Helvetica",
                "font.size": 12,
                "figure.figsize": (5, 2),
                "mathtext.fontset": 'stix'
        })
        eval_multi_run(p.parent, list(result.keys())) 


        print("Stats for patches calculated on full-precision NN:")
        model_path = 'misc/Frontnet.onnx'
        model = load_quantized(path=model_path, device='cpu')
        dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
        test_set = load_dataset(path=dataset_path, batch_size=settings['batch_size'], shuffle=True, drop_last=False, train=False, num_workers=0)


        p = Path('eval/exp1/') # change path to experiment calculated on full-precision network!
        # calc loss for each mode
        for mode in ['fixed', 'joint', 'split', 'hybrid']:
            # load all best patches for current mode
            patches = torch.tensor(load_patch(p, mode)).unsqueeze(1).unsqueeze(1)
            # load all optimized positions for all targets
            positions = torch.tensor(load_position(p, mode))
            
            # get target values in correct shape and move tensor to device
            targets = [values for _, values in settings['targets'].items()]
            targets = np.array(targets, dtype=float).T
            targets = torch.tensor(targets)

            target_loss = []
            # for each target
            for target_idx, target in enumerate(targets):
                # get best patch and best positon
                for position, patch in zip(positions[target_idx], patches):
                    # translate saved position into full transformation
                    transformation_matrix = get_transformation(*position)
                    # calculate loss on whole training set
                    target_loss.append(calc_eval_loss(test_set, patch, transformation_matrix, model, target).detach().cpu().numpy())

            print(mode, "mean", np.mean(target_loss), "std", np.std(target_loss))


def main():
    e = Experiment4()
    exp.exp(e)


if __name__ == '__main__':
    main()