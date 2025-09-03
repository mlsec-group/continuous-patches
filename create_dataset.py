# import yaml
# import subprocess 
# import numpy as np
# from pathlib import Path
# import os
# import shlex
# import pickle
# import argparse

import torch
# from patch_placement import place_patch
# from util import load_model, load_dataset

# from yolo_bounding import YOLOBox

import yaml
import subprocess
import numpy as np
from pathlib import Path
import os
import shlex
import pickle
import argparse

def gen_T(coeffs):
    T = np.zeros((2,3))
    T[0, 0] = T[1, 1] = coeffs[0] # sf
    T[0, 2] = coeffs[1] # tx
    T[1, 2] = coeffs[2] # ty
    # T[2, 2] = 1.

    return torch.tensor(T, dtype=torch.float32)

def load_targets(patch_path):
    parent_folder = patch_path.parent
    with open(parent_folder / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).float().unsqueeze(0)

    if len(targets.shape) > 2.:
        targets = targets.squeeze(0)
    return targets

def get_coeffs(patch_path, idx=-1):
    parent_folder = patch_path.parent
    # T_coeffs = np.load(parent_folder / 'positions_norm.npy') # shape: [sf, tx, ty], epochs, num_targets, 1, 1
    # # should shape: num_targets, [sf, tx, ty]
    # T_coeffs = T_coeffs[:, idx, :, 0, 0].T

    position = np.load(parent_folder / 'positions_initial.npy').T  # T_coeffs is of shape (1,3) in the end, positions_initial is of shape (3,1) without the transpose

    return position

def get_Ts(T_coeffs):
    Ts = [gen_T(coeffs) for coeffs in T_coeffs]
    return Ts

def calc_loss(target, patch, T, img, model):
    mod_img_pt = place_patch(img.to(patch.device), patch, T.unsqueeze(0).to(patch.device), random_perspection=False) 
    x, y, z, yaw = model(mod_img_pt)
    predicted_pose = torch.hstack((x, y, z))[0].detach().cpu()
    # print(predicted_pose, target)
    mse = torch.nn.functional.mse_loss(target.detach().cpu(), predicted_pose).item()
    return mse

def idx_best_target(targets, patch, Ts, img, model):
    # print("inside best target")
    # print(targets)
    # print(Ts)
    losses = np.array([calc_loss(target, patch, T, img, model) for target, T in zip(targets, Ts)])
    return np.argmin(losses)

def get_best_target_pos(patch, patch_path, dataset, model, device):
    targets = load_targets(patch_path).to(device)
    coeffs = get_coeffs(patch_path)
    Ts = get_Ts(coeffs)

    patch = torch.tensor(patch, device=device).float().unsqueeze(0).unsqueeze(0)

    best_targets = []
    for data in dataset:
        img, _ = data
        best_targets.append(idx_best_target(targets, patch, Ts, img, model))

    idx_best = np.argmax(np.bincount(np.array(best_targets)))

    return targets[idx_best].detach().cpu().numpy(), coeffs[idx_best]


def train(idx_start=0, idx_end=100, model='frontnet'):
    # missing_indices = np.load('missing_indices.npy')
    # for i in missing_indices:
    for i in range(idx_start, idx_end):
        with open('dataset.yaml') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        patch_size = settings['patch']['size']
        path = Path(f"{settings['path']}/{model}/{patch_size[0]}x{patch_size[1]}/{i}/")
        print(path)

        # targets = [values for _, values in settings['targets'].items()]
        # targets = np.array(targets, dtype=float).T

        number_targets = 1#np.random.randint(1, 4)

        random_target_x = np.random.uniform(0,2,number_targets)
        random_target_y = np.random.uniform(-1,1,number_targets,)
        random_target_z = np.random.uniform(-0.5,0.5,number_targets,)

        # overwrite settings
        settings['path'] = str(path)
        settings['targets']['x'] = random_target_x.tolist()
        settings['targets']['y'] = random_target_y.tolist()
        settings['targets']['z'] = random_target_z.tolist()
        settings['patch']['size'] = patch_size
        
        sf, tx_min, tx_max, ty_min, ty_max = gen_random_monitor_space(sf_min=settings['sf_min'], sf_max=settings['sf_max'], camera_image_size=(160, 96), patch_size=(80,80))

        settings['sf'] = float(sf)
        settings['tx_min'] = float(tx_min)
        settings['tx_max'] = float(tx_max)
        settings['ty_min'] = float(ty_min)
        settings['ty_max'] = float(ty_max)

        os.makedirs(path, exist_ok = True)
        with open(path / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f)

        # command = shlex.split(f"sbatch dataset.sh {str(path / 'settings.yaml')} {model}")
        # print(command)
        command = shlex.split(f"python attacks.py --file {str(path / 'settings.yaml')} --model {model}")
        subprocess.run(command)
        del settings

def read_data(path, idx_start=0, idx_end=100, model='frontnet', idx=-1):
    with open('dataset.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    
    patch_size = settings['patch']['size']
    path = Path(path)

    file_paths = list(path.glob('[0-9]*/patches.npy'))
    file_paths.sort(key=lambda path: int(path.parent.name))

    print(len(file_paths))

    # figure out which indices are missing if we should have 0-999 in file paths
    # missing_indices = []
    # for i in range(0, 1000):
    #     if i not in [int(path.parent.name) for path in file_paths]:
    #         missing_indices.append(i)
    # print(f"Missing indices: {missing_indices}")
    # np.save('missing_indices.npy', missing_indices)

    # return

    # np.save('missing_indices.npy', missing_indices)

    # from util import load_dataset
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    # dataset = load_dataset(dataset_path, batch_size=1, train=False, train_set_size=0.9)

    # if model == 'frontnet':
    #     from util import load_model
    #     model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    #     model_config = '160x32'
    #     model = load_model(path=model_path, device=device, config=model_config)
    #     model.eval()
    # else:
    #     from yolo_bounding import YOLOBox
    #     model = YOLOBox()
    #     model.to(device)

    # print(model)
    
    patches = np.array([np.load(file_path)[idx][0][0] for file_path in file_paths[idx_start:idx_end]])
    print(patches.shape, patches.min(), patches.max())
    # all_images = dataset.dataset.data
    # print(all_images.shape)

    targets = [load_targets(patch_path) for patch_path in file_paths[idx_start:idx_end]]
    print(len(targets))
    
    positions = [get_coeffs(patch_path, idx) for patch_path in file_paths[idx_start:idx_end]]

    print(len(positions))

    # best_targets = []
    # best_coeffs = []
    # with torch.no_grad():
    #     for i, (patch, p_targets, p_coeffs) in enumerate(zip(patches, targets, coeffs)):
    #         losses = []
    #         patch = torch.tensor(patch, device=device).float().unsqueeze(0).unsqueeze(0)
    #         Ts = get_Ts(p_coeffs)
            
    #         for idx_target in range(len(p_targets)):
    #             T = Ts[idx_target]
    #             mod_img_pt = place_patch(all_images.to(patch.device), patch.repeat((len(all_images), 1, 1, 1)), T.unsqueeze(0).repeat(len(all_images), 1, 1).to(patch.device), random_perspection=False) 
    #             if model == 'frontnet':
    #                 x, y, z, yaw = model(mod_img_pt)
    #                 predicted_pose = torch.hstack((x, y, z)).detach().cpu()
    #             else: 
    #                 predicted_pose = model(mod_img_pt)
    #             print(all_images.shape)
    #             print(predicted_pose.shape)
    #             print(p_targets, p_targets.shape, p_targets[idx_target], p_targets[idx_target].shape)
                
    #             # print(p_targets[idx_target].repeat(len(all_images), 1).shape)
    #             losses.append(torch.nn.functional.mse_loss(p_targets[idx_target].repeat(len(all_images), 1), predicted_pose).detach().cpu().item())
            
    #         best_targets.append(p_targets[np.argmin(losses)].detach().cpu().numpy())
    #         best_coeffs.append(p_coeffs[np.argmin(losses)])

    # targets = np.array(best_targets)
    # positions = np.array(best_coeffs)

    # print(targets.shape)
    # print(positions.shape)

    return patches, targets, positions
    # out = np.array([get_best_target_pos(patch, path, dataset, model, device) for patch, path in zip(patches, file_paths[idx_start:idx_end+1])])
    # targets, positions = np.split(out, 2, axis=1)
    
def save_pickle(path, out_name, idx_start=0, idx_end=100, model='frontnet'):
    patches, targets, positions = read_data(path, idx_start, idx_end, model)
    with open(out_name, 'wb') as f:
        pickle.dump([[patch, target, position] for patch, target, position in zip(patches, targets, positions)], f, pickle.HIGHEST_PROTOCOL)


def gen_random_monitor_space(sf_min, sf_max, camera_image_size=(160, 96), patch_size=(80, 80)):
    overlap = False
    while not overlap:
        random_scale_factor = np.random.uniform(0.2, 1.5)
        random_tx = np.random.uniform((-1.5*160), (1.5*160))
        random_ty = np.random.uniform((-1.5*96), (1.5*96))
        T = np.zeros((3,3))
        T[0, 0] = T[1, 1] = random_scale_factor
        T[0, 2] = random_tx 
        T[1, 2] = random_ty
        T[2, 2] = 1.0 
        corners_camera = np.array([(0, 0), (camera_image_size[0], 0), (0, camera_image_size[1]), (camera_image_size[0], camera_image_size[1])], dtype=np.float32)
        corners_monitor = np.array([T @ np.array((x, y, 1)) for x, y in corners_camera])

        if corners_monitor.T[0].min() < 0 or corners_monitor.T[1].min() < 0  or corners_monitor.T[0].max() > 160 or corners_monitor.T[1].max() > 96:
            # print("Monitor space is not valid, generating new one...")
            continue
        else:
            overlap = True

    # print(corners_camera)
    # print(T)
    # print(corners_monitor)

    intersection_ul = (max(corners_monitor[0, 0], corners_camera[0, 0]), 
                        max(corners_monitor[0, 1], corners_camera[0, 1]))
    intersection_ur = (min(corners_monitor[1, 0], corners_camera[1, 0]),
                        max(corners_monitor[1, 1], corners_camera[1, 1]))
    intersection_ll = (max(corners_monitor[2, 0], corners_camera[2, 0]),
                        min(corners_monitor[2, 1], corners_camera[2, 1]))
    intersection_lr = (min(corners_monitor[3, 0], corners_camera[3, 0]),
                        min(corners_monitor[3, 1], corners_camera[3, 1]))

    print(intersection_ul) 
    print(intersection_ur)
    print(intersection_ll)
    print(intersection_lr)
    
    height = min(intersection_ll[1], intersection_lr[1]) - max(intersection_ur[1], intersection_ul[1])
    width = min(intersection_ur[0], intersection_lr[0]) - max(intersection_ul[0], intersection_ll[0])

    max_dim = min(height, width)
    possible_max_sf = max_dim / min(patch_size[0], patch_size[1])
    random_sf = min(np.random.uniform(sf_min, possible_max_sf), sf_max)

    min_tx = intersection_ul[0]
    max_tx = max(intersection_ur[0], intersection_lr[0]) - (patch_size[0] * random_sf) # 

    min_ty = intersection_ul[1]

    max_ty = max(intersection_ll[1], intersection_lr[1]) - (patch_size[1] * random_sf)
    

    print(f"sf: {random_sf}, min tx: {min_tx}, max tx: {max_tx}, min ty: {min_ty}, max ty: {max_ty}")
    return random_sf, min_tx, max_tx, min_ty, max_ty
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'save'])
    parser.add_argument('idx', type=int, metavar='N', nargs='+', default=[100])
    parser.add_argument('--folder', type=str, default='results/test/')
    parser.add_argument('--out', type=str, default='dataset.pickle')
    parser.add_argument('--model', type=str, default='frontnet', choices=['frontnet', 'yolov5'])
    args = parser.parse_args()


    # gen_random_monitor_space()


    if len(args.idx) < 2:
        idx_start = 0
        idx_end = args.idx[0]
    else:
        idx_start, idx_end = np.sort(args.idx)

    if args.mode == 'train':
        train(idx_start, idx_end, args.model)
    
    if args.mode == 'save':
        save_pickle(args.folder, args.out, idx_start, idx_end, args.model)
