import numpy as np
import torch
from util import load_dataset, load_model

from tqdm import trange
import os
from pathlib import Path

from camera import Camera

def normalize_yaw(yaw):
    return np.atan2(np.sin(yaw), np.cos(yaw))

def normalize_yaw_t(yaw):
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))


def _perspective_grid(
coeffs: list[float], 
w: int, h: int, 
ow: int, oh: int, 
dtype: torch.dtype, 
device: torch.device,
center = None,
) -> torch.Tensor:
    # source: https://github.com/pytorch/pytorch/issues/100526#issuecomment-1610226058
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    batch_size = coeffs.shape[0]
    theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)

    theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0) # theta2 is a matrix of shape [2, 3], it is the last row of the original transformation matrix repeated 2x
    theta2 = theta2.reshape(batch_size, 2, 3) # reshape from [batch_size*2, 3] to [batch_size, 2, 3] 

    d = 0.5
    base_grid = torch.empty(batch_size, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    shape = (batch_size, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    if center is not None:
        center = torch.tensor(center, dtype=dtype, device=device)
    else:
        center = 1.0

    output_grid = output_grid1.div_(output_grid2).sub_(center)
    return output_grid.view(batch_size, oh, ow, 2)
    
def project_patch(patches, T_matrices, images):
    """ Project the patches on the batch of camera images.
    Args:
        patches: Tensor of shape [B, C, H_p, W_p] (batch of patches).
        T_matrices: Tensor of shape [B, 3, 3] (batch of transformation matrices).
        images: Tensor of shape [B, C, H_i, W_i] (batch of images).
    Returns:
        Tensor of manipulated images of shape [B, C, H_i, W_i].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patches = patches.to(device)
    T_matrices = T_matrices.to(device)
    images = images.to(device)

    batch_size, _, p_height, p_width = patches.shape
    # _, _, i_height, i_width = images.shape

    i_height, i_width = images.shape[-2::]
    while len(images.shape) < 4:
        images = images.unsqueeze(0)

    # Create masks for patches
    masks = torch.ones_like(patches, dtype=torch.float32, device=device)

    # print(T_matrices[0])

    # Invert transformation matrices
    inv_T_matrices = torch.inverse(T_matrices)

    # Flatten transformation matrices for grid computation
    # print(inv_T_matrices.shape)
    coeffs = inv_T_matrices.reshape(batch_size, -1)
    # print(coeffs.shape)

    # Generate perspective grids for batch
    grids = _perspective_grid(
        coeffs, w=p_width, h=p_height, ow=i_width, oh=i_height, 
        dtype=torch.float32, device=device, center=[1., 1.]
    )

    # Apply grid sampling for patches and masks
    transformed_patches = torch.nn.functional.grid_sample(
        patches, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    )
    bit_masks = torch.nn.functional.grid_sample(
        masks, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    ).bool()

    # Combine transformed patches with original images
    manipulated_images = images * ~bit_masks
    manipulated_images += transformed_patches

    return manipulated_images

def gen_random_monitor_space(sf_min, sf_max, cam, camera_image_size=(160, 96), patch_size=(80, 80)):
    max_camera_image_width = camera_image_size[0] - (sf_min * patch_size[0])
    max_camera_image_height = camera_image_size[1] - (sf_min * patch_size[1])
    # print("max_camera_image_width:", max_camera_image_width)
    # print("max_camera_image_height:", max_camera_image_height)


    tx_min = np.random.randint(0, max_camera_image_width)
    ty_min = np.random.randint(0, max_camera_image_height)


    sf_max = camera_image_size[1] / patch_size[1]

    patch_monitor_max_x = min(tx_min + (sf_max * patch_size[0]), camera_image_size[0])
    patch_monitor_max_y = min(ty_min + (sf_max * patch_size[1]), camera_image_size[1])

    patch_monitor_width = patch_monitor_max_x - tx_min
    patch_monitor_height = patch_monitor_max_y - ty_min
    # # print("patch_monitor_width:", patch_monitor_width)
    # # print("patch_monitor_height:", patch_monitor_height)
    
    max_dim = min(patch_monitor_width, patch_monitor_height)
    possible_max_sf = max_dim / min(patch_size[0], patch_size[1])
    
    random_sf = np.random.uniform(sf_min, possible_max_sf)

   
    tx_max = patch_monitor_max_x - (random_sf * patch_size[0])
    ty_max = patch_monitor_max_y - (random_sf * patch_size[1])

   

    # calculate a target yaw based on the possible patch monitor space
    bb = np.array([tx_min, ty_min, patch_monitor_max_x, patch_monitor_max_y])
    xyz_drone = cam.xyz_from_bb(bb)
    target_yaw = normalize_yaw(np.arctan2(xyz_drone[1], xyz_drone[0]))
    

    print(f"sf: {random_sf}, min tx: {tx_min}, max tx: {tx_max}, min ty: {ty_min}, max ty: {ty_max}, target yaw: {target_yaw}")
    return random_sf, tx_min, tx_max, ty_min, ty_max, target_yaw


def norm_transformation(sf, tx, ty, sf_min, sf_max, tx_min, tx_max, ty_min, ty_max):
    # tx_tanh = torch.tanh(tx) #* 0.8
    # ty_tanh = torch.tanh(ty) #* 0.8

    # print("Inside norm_transformation:")
    # print("sf:", sf)
    # print("tx:", tx)
    # print("ty:", ty)
    # print("tx_min, tx_max, ty_min, ty_max:", tx_min, tx_max, ty_min, ty_max)

    # sf_norm = (sf_max - sf_min) * (torch.tanh(sf) + 1) * 0.5 + sf_min
    # scaled_patch_size = 80 * sf
    # tx_max -= scaled_patch_size
    # ty_max -= scaled_patch_size

    # new patch placement implementation might need different tx, ty limits!:
    sf_norm = (sf_max - sf_min) * (torch.tanh(sf) + 1) * 0.5 + sf_min
    tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min


    # if ty_norm < ty_min:
    #     print("Something weird happend: ")
    #     print("ty: ", ty)
    #     print("ty_norm: ", ty_norm)
    #     print("ty_min, ty_max:", ty_min, ty_max)

    # if tx_norm < tx_min:
    #     print("Something weird happend: ")
    #     print("tx: ", tx)
    #     print("tx_norm: ", tx_norm)
    #     print("tx_min, tx_max:", tx_min, tx_max)

    # scaling_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min # normalizes scaling factor to range [0.3, 0.5]

    #return scaling_norm, tx_norm, ty_norm
    return sf_norm, tx_norm, ty_norm

def noisy_transformations(sf, tx, ty):
    sf_n = sf + np.random.normal(0.0, 0.1)
    tx_n = tx + np.random.normal(0.0, 0.1)
    ty_n = ty + np.random.normal(0.0, 0.1)
    return sf_n, tx_n, ty_n

def construct_T_matrix(sf, tx, ty, sf_min=0.5, sf_max=1.2, tx_min=0., tx_max=160., ty_min=0., ty_max=96., noise=True):
    # print("Inside construct T:")
    # print("sf:", sf)
    # print("tx:", tx)
    # print("ty:", ty)
    # print("tx_min, tx_max, ty_min, ty_max:", tx_min, tx_max, ty_min, ty_max)

    if noise:
        sf, tx, ty = noisy_transformations(sf, tx, ty)
    # noisy_sf, noisy_tx, noisy_ty = noisy_transformations(sf, tx, ty)
    # print("ty_min, ty_max:", ty_min, ty_max)
    # print("tx_min, tx_max:", tx_min, tx_max)
    norm_sf, norm_tx, norm_ty = norm_transformation(sf=sf, tx=tx, ty=ty, sf_min=sf_min, sf_max=sf_max, tx_min=tx_min, tx_max=tx_max, ty_min=ty_min, ty_max=ty_max)
    # print(norm_tx, norm_ty)

    T_matrix = torch.zeros((3, 3), device=sf.device)
    scale_T = torch.eye(2, device=sf.device) * norm_sf
    T_matrix[:2, :2] = scale_T
    T_matrix[0, 2] = norm_tx
    T_matrix[1, 2] = norm_ty
    T_matrix[2, 2] = 1.0
    # print(T_matrix)
    return T_matrix

def calc_eval_loss(test_set, model, patch, T, target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = patch.to(device) / 255.0  # Normalize patch
    T = T.to(device)

    total_loss = 0.0
    with torch.no_grad():
        for data in test_set:
            batch, _ = data
            batch = batch.to(device) / 255.0
            
            # print(batch.shape, patch.shape, T.shape)

            batch_patch = patch.expand(batch.size(0), -1, -1, -1)
            batch_T = T.expand(batch.size(0), -1, -1)

            #manipulated_images = torch.stack([project_patch(patch, T, img) for img in batch], dim=0).squeeze(1)
            manipulated_images = project_patch(batch_patch, batch_T, batch)
            manipulated_images.data.clamp_(0., 1.)  # Ensure values are in [0, 1]
            x, y, z, yaw = model(manipulated_images*255.)
            prediction = torch.stack([x, y, z, yaw])
            prediction = prediction.squeeze(2).mT

            l2_distance = torch.linalg.norm(prediction[:, :3] - target[:3], dim=1, ord=2)
            angular_loss = 1 - torch.cos(normalize_yaw_t(prediction[:, 3]) - normalize_yaw_t(target[3]))

            loss = (l2_distance + angular_loss).mean()
            total_loss += loss.item()

    average_loss = total_loss / len(test_set)
    print(f"Evaluation Loss: {average_loss}")
    return average_loss

def joint(train_set, test_set, model, target, patch, sf_min, sf_max, tx_min, tx_max, ty_min, ty_max, epochs, lr, device):

    patch_t = patch.clone().to(device).requires_grad_(True)

    #sf_t = torch.tensor(sf, device=device, dtype=torch.float32, requires_grad=False)
    sf_t = torch.FloatTensor(1).uniform_(-1, 1).to(device).requires_grad_(True)
    tx_t = torch.FloatTensor(1).uniform_(-1, 1).to(device).requires_grad_(True)
    ty_t = torch.FloatTensor(1).uniform_(-1, 1).to(device).requires_grad_(True)


    optimizer = torch.optim.Adam([patch_t, sf_t, tx_t, ty_t], lr=lr, eps=1e-4)
    best_loss = np.inf
    best_patch = initial_patch.clone().detach() * 255.
    best_T = construct_T_matrix(sf_t.clone(), tx_t.clone(), ty_t.clone(), sf_min, sf_max, tx_min, tx_max, ty_min, ty_max, noise=False).detach()

    train_losses = []
    eval_losses = []

    for epoch in trange(epochs):
        epoch_loss = torch.tensor(0., device=device)

        for data in train_set:
            
            batch, _ = data

            batch = batch.to(device) / 255.
            # print(batch.min(), batch.max())
            # print(patch_t.min(), patch_t.max())
            # img = train_set.dataset[0][0].to(device).unsqueeze(0) / 255.0  # Use the first image in the batch for patch projection
            # batch = img.expand(batch.size(0), -1, -1, -1)  # Expand to match batch size
            
            # print("Inside joint:")
            # print("sf_t:", sf_t)
            # print("tx_t:", tx_t)
            # print("ty_t:", ty_t)
            # print("sf_min, sf_max:", sf_min, sf_max)
            # print("tx_min, tx_max, ty_min, ty_max:", tx_min, tx_max, ty_min, ty_max)
            batch_T = torch.stack([construct_T_matrix(sf_t, tx_t, ty_t, sf_min, sf_max, tx_min, tx_max, ty_min, ty_max) for _ in range(batch.size(0))], dim=0)
            batch_patch = patch_t.expand(batch.size(0), -1, -1, -1)
            # print(batch.shape, batch_T.shape, batch_patch.shape)

            # print(batch.dtype, batch_T.dtype, batch_patch.dtype)
            #manipulated_images = torch.stack([project_patch(batch_patch[i], batch_T[i], batch[i]) for i in range(batch.size(0))], dim=0).squeeze(1)
            manipulated_images = project_patch(batch_patch, batch_T, batch)
            manipulated_images += torch.distributions.normal.Normal(loc=0.0, scale=0.1).sample(manipulated_images.shape).to(patch.device)
            
            manipulated_images.data.clamp_(0., 1.)  # Ensure values are in [0, 1]

            x, y, z, yaw = model(manipulated_images*255.)
            prediction = torch.stack([x, y, z, yaw])
            prediction = prediction.squeeze(2).mT

            # l2_distance = torch.stack([torch.linalg.norm((pred[:3] - target[:3]), ord=2) for pred in prediction])
            # angular_loss = torch.stack([1 - torch.cos(normalize_yaw_t(pred[3]) - normalize_yaw_t(target[3])) for pred in prediction])

            # print(l2_distance, angular_loss)

            l2_distance = torch.linalg.norm(prediction[:, :3] - target[:3], dim=1, ord=2)
            angular_loss = 1 - torch.cos(normalize_yaw_t(prediction[:, 3]) - normalize_yaw_t(target[3]))

            loss = (l2_distance + angular_loss).mean()

            # loss = torch.stack([sim.eval(new_pose) for new_pose in new_poses]).mean()
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            patch_t.data.clamp_(0, 1.)  # Ensure patch values are in [0, 255]
            # sf_t.data.clamp_(0.1, max_scale)
            # tx_t.data.clamp_(0, 1.)
            # ty_t.data.clamp_(0, 1.)

            # print(construct_T_matrix(sf_t, tx_t, ty_t))
            
        epoch_loss /= len(train_set)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.item()}, l2: {l2_distance.mean().item()}, angular: {angular_loss.mean().item()}")
        train_losses.append(epoch_loss.item())
        # if epoch_loss.item() < 0.1:
        #     print("Early stopping due to low loss")
        #     break
        if epoch_loss.item() < best_loss:
            best_patch = patch_t.clone().detach() * 255.
            best_T = best_T = construct_T_matrix(sf_t.clone(), tx_t.clone(), ty_t.clone(), sf_min, sf_max, tx_min, tx_max, ty_min, ty_max, noise=False).detach()
            best_loss = epoch_loss.item()
        
        eval_loss = calc_eval_loss(test_set, model, best_patch, best_T, target)
        eval_losses.append(eval_loss)
    
    return best_patch, best_T, train_losses, eval_losses


if __name__ == "__main__":

    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='results/minimal_test/', help='Path to save results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    # parser.add_argument('--target', type=float, nargs='+', help='Target pose as a list of floats [x, y, z, yaw]')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate for the optimizer')
    args = parser.parse_args()

    results_path = Path(args.results_path)
    os.makedirs(results_path, exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_set = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0, IMRC=True)
    # print("DATASET SIZE: ", len(train_set.dataset))
    
    test_set = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, train=False, num_workers=0, IMRC=True)


    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()

    initial_patch = torch.rand((1, 1, 45, 80), device=device, dtype=torch.float32)

    # cam = Camera('camera_calibration.yaml')

    # sf, tx_min, tx_max, ty_min, ty_max, target_yaw = gen_random_monitor_space(0.2, 1.5, cam, camera_image_size=(160, 96), patch_size=(80, 80))

    sf_min = 0.5
    sf_max = 1.1
    tx_min = 0.
    tx_max = 160.
    ty_min = 0.
    ty_max = 96.



    test_img = test_set.dataset[0][0]
    print(test_img.shape)


    epochs = args.epochs
    lr = args.lr

    target = torch.tensor([1., 0., 0., 0.], device=device, dtype=torch.float32)  # Example target pose


    settings = {}
    settings['results_path'] = str(results_path)
    settings['seed'] = seed
    settings['patch_size'] = list(initial_patch.shape[-2::])
    settings['sf_min'] = sf_min
    settings['sf_max'] = sf_max
    settings['tx_min'] = tx_min
    settings['tx_max'] = tx_max
    settings['ty_min'] = ty_min
    settings['ty_max'] = ty_max
    settings['epochs'] = epochs
    settings['lr'] = lr
    settings['target'] = target.clone().cpu().numpy().tolist()

    with open(results_path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)

    patch, T, train_losses, eval_losses = joint(train_set, test_set, model, target, initial_patch, sf_min, sf_max, tx_min, tx_max, ty_min, ty_max, epochs, lr, device)

    np.save(results_path / 'train_losses.npy', np.array(train_losses))
    np.save(results_path / 'eval_losses.npy', np.array(eval_losses))
    np.save(results_path / 'patch.npy', patch.cpu().numpy())
    np.save(results_path / 'T.npy', T.cpu().numpy())

    # eval

    print(tx_min, tx_max, ty_min, ty_max)
    print(T)


    # tx_max = patch_monitor_max_x - (random_sf * patch_size[0])
    # ty_max = patch_monitor_max_y - (random_sf * patch_size[1])
    # patch_monitor_max_x = tx_max + (sf * initial_patch.shape[2])
    # patch_monitor_max_y = ty_max + (sf * initial_patch.shape[3])

    manipulated_image = project_patch(patch, T, test_img).detach().cpu().numpy()
    print("Manipulated image shape:", manipulated_image.shape)

    import matplotlib.pyplot as plt
    plt.imshow(manipulated_image[0][0], cmap='gray')
    plt.scatter(tx_min, ty_min)
    plt.scatter(tx_max, ty_max)
    # plt.scatter(patch_monitor_max_x, patch_monitor_max_y, color='red', label='Patch Monitor Max')
    # plt.scatter(tx_max, ty_max, color='red', label='Min TX, TY')
    # plt.scatter(tx_max, ty_min, color='green', label='Min TX, Max TY')
    # plt.scatter(tx_min, ty_max, color='orange', label='Max TX, Min TY')
    # plt.scatter(tx_min, ty_min, color='blue', label='Max TX, TY')
    plt.savefig(results_path / 'test_image.png')
    plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_path / 'losses.png')
    plt.close()