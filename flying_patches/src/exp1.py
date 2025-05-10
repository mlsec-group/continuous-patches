import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp
from plots import eval_multi_run
from matplotlib import pyplot as plt


class Experiment1:
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

        # with PdfPages(p.parent / 'exp1.pdf') as pdf:
        #     fig, ax = plt.subplots(constrained_layout=True)


def main():
    e = Experiment1()
    exp.exp(e)


if __name__ == '__main__':
    main()
