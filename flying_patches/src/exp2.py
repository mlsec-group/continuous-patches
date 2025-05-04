import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp

# plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Experiment2:
    def create_settings(self, base_settings, trials, mode, quantized):
        all_settings = []
        for i in range(trials):
            for p in range(1, base_settings['num_patches']+1, 2):
                for j in range(1, len(base_settings['targets']['x'])+1, 2):
                    s = copy.deepcopy(base_settings)
                    s['targets']['x'] = base_settings['targets']['x'][0:j]
                    s['targets']['y'] = base_settings['targets']['y'][0:j]
                    s['targets']['z'] = base_settings['targets']['z'][0:j]
                    s['path'] = str(Path(base_settings['path']) / ("patch" + str(p) + "_target" + str(j) + "_trial" + str(i)))
                    s['quantized'] = quantized
                    s['num_patches'] = p
                    all_settings.append(s)
        return all_settings

    def stats(self, all_settings):
        # output statistics
        result = defaultdict(list)
        for settings in all_settings:
            p = Path(settings['path'])
            if (p / 'losses_test.npy').exists():
                test_losses = np.load(p / 'losses_test.npy')
                print(settings['num_patches'], len(settings['targets']['x']), np.mean(test_losses[-1]))
                result[(settings['num_patches'], len(settings['targets']['x']))].append(test_losses[-1])
            else:
                print("Warning ", p / 'losses_test.npy', " not found")

        xs = defaultdict(list)
        ys = defaultdict(list)
        yerr = defaultdict(list)
        for (num_patches, num_targets), v in result.items():
            print(num_patches, num_targets, v)
            all = np.stack(v)
            print(num_patches, num_targets, "mean", np.mean(all), "std", np.std(all))
            xs[num_patches].append(num_targets)

            # loss_per_run = np.mean(all, axis=1)
            ys[num_patches].append(np.mean(all))
            yerr[num_patches].append(np.std(all))

        # change settings to match latex
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "font.size": 12,
            "figure.figsize": (5, 2.5),
        })

        with PdfPages(p.parent / 'exp2.pdf') as pdf:
            fig, ax = plt.subplots(constrained_layout=True)

            for num_patches in xs.keys():
                label = "1 patch" if num_patches == 1 else "{} patches".format(num_patches)
                ax.plot(xs[num_patches], ys[num_patches], label=label)
                ax.fill_between(xs[num_patches], np.asarray(ys[num_patches])+np.asarray(yerr[num_patches]), np.asarray(ys[num_patches])-np.asarray(yerr[num_patches]), alpha=0.15)

            ax.set_ylabel('Test Loss per Target [m]')
            ax.set_xlabel('Number of Targets per Patch')
            ax.legend()
            # ax.set_ylim(0,0.2)

            # ax.bar(xs, ys)
            # ax.errorbar(xs, ys, yerr)

            pdf.savefig(fig)
            plt.close()

def main():
    e = Experiment2()
    exp.exp(e)


if __name__ == '__main__':
    main()