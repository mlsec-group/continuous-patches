import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp


class Experiment3:
    def create_settings(self, base_settings, trials, modes, quantized):
        all_settings = []
        for patch_mode in ['face_2', 'face_3', 'face_4', 'white', 'random']:
            for i in range(trials):
                for mode in modes:
                    s = copy.copy(base_settings)
                    s['mode'] = mode
                    s['path'] = "eval/exp3"+ patch_mode + '/' + mode + str(i)
                    s['quantized'] = quantized
                    if 'face' in patch_mode:
                        s['patch']['mode'] = 'face'
                        patch_num = patch_mode.split('_')[1] 
                        s['patch']['path'] = 'src/custom_patches/custom_patch' + patch_num + '_resized.npy'
                    else:
                        s['patch']['mode'] = patch_mode
                        s['patch']['path'] = None
                    all_settings.append(s)
        return all_settings

    def stats(self, all_settings):
        # output statistics
        result = defaultdict(list)
        for settings in all_settings:
            p = Path(settings['path'])
            test_losses = np.load(p / 'losses_test.npy')
            # print(settings['mode'], test_losses[-1])
            patch_mode = settings['patch']['mode']
            opt_mode = settings['mode']
            result[f'{patch_mode}_{opt_mode}'].append(test_losses[-1])

        for k, v in result.items():
            all = np.stack(v)
            print(k, "mean", np.mean(all), "std", np.std(all))

def main():
    e = Experiment3()
    exp.exp(e)


if __name__ == '__main__':
    main()
