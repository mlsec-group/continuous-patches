import numpy as np
import torch
from util import load_dataset, load_model

from tqdm import trange


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
    
def project_patch(patch, T, img):
    """ Project the patch on the camera image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(patch):
        patch = torch.tensor(patch, dtype=torch.float64, device=device)
    # patches should be of shape [B, 1, H, W]
    if len(patch.shape) < 3:
        patch = patch.unsqueeze(0)
    if len(patch.shape) == 3:
        patch = patch.unsqueeze(1)
    
    # print("Inside project patch")
    # print("patch shape:", patch.shape)
    
    if not torch.is_tensor(img):
        img = torch.tensor(img, device=device)
    # img should be of shape [1, 1, H, W]
    while len(img.shape) < 4:
        img = img.unsqueeze(0)

    # print("img shape:", img.shape)

    if not torch.is_tensor(T):
        T = torch.tensor(T, dtype=torch.float32, device=device) # shape should be [3, 3]

    p_height, p_width = patch.shape[-2:]
    i_height, i_width = img.shape[-2:]

    mask = torch.ones_like(patch, dtype=torch.float32, device=device)

    try:
        inv_t = torch.inverse(T)
    except Exception as e:
        print("Error inverting transformation matrix:", e)
        print("Transformation matrix T:", T)

    coeffs = inv_t.flatten().unsqueeze(0)
    
    grid = _perspective_grid(coeffs, w=p_width, h=p_height, dtype=torch.float32, ow=i_width, oh=i_height, device=device, center = [1., 1.])

    bit_mask = torch.nn.functional.grid_sample(mask, grid, mode='bilinear', align_corners=False, padding_mode='zeros').bool()
    transformed_patch = torch.nn.functional.grid_sample(patch, grid, mode='bilinear', align_corners=False, padding_mode='zeros')

    modified_image = img * ~bit_mask.bool()
    modified_image += transformed_patch

    return modified_image

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

    # print(intersection_ul) 
    # print(intersection_ur)
    # print(intersection_ll)
    # print(intersection_lr)
    
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


def norm_transformation(sf, tx, ty, tx_min, tx_max, ty_min, ty_max):
    # tx_tanh = torch.tanh(tx) #* 0.8
    # ty_tanh = torch.tanh(ty) #* 0.8

    # sf_norm = (sf_max - sf_min) * (torch.tanh(sf) + 1) * 0.5 + sf_min
    scaled_patch_size = 80 * sf
    tx_max -= scaled_patch_size
    ty_max -= scaled_patch_size

    # new patch placement implementation might need different tx, ty limits!:
    tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min

    # scaling_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min # normalizes scaling factor to range [0.3, 0.5]

    #return scaling_norm, tx_norm, ty_norm
    return tx_norm, ty_norm

def noisy_transformations(sf, tx, ty):
    sf_n = sf + np.random.normal(0.0, 0.1)
    tx_n = tx + np.random.normal(0.0, 0.1)
    ty_n = ty + np.random.normal(0.0, 0.1)
    return sf_n, tx_n, ty_n

def construct_T_matrix(sf, tx, ty, tx_min=48., tx_max=128., ty_min=20., ty_max=66., noise=True):
    if noise:
        sf, tx, ty = noisy_transformations(sf, tx, ty)
    # noisy_sf, noisy_tx, noisy_ty = noisy_transformations(sf, tx, ty)
    norm_tx, norm_ty = norm_transformation(sf, tx, ty, tx_min, tx_max, ty_min, ty_max)
    # print(norm_sf, norm_tx, norm_ty)

    T_matrix = torch.zeros((3, 3), device=sf.device)
    scale_T = torch.eye(2, device=sf.device) * sf
    T_matrix[:2, :2] = scale_T
    T_matrix[0, 2] = norm_tx
    T_matrix[1, 2] = norm_ty
    T_matrix[2, 2] = 1.0
    # print(T_matrix)
    return T_matrix

def joint(dataset, model, target, patch, sf, tx_min, tx_max, ty_min, ty_max, epochs, lr, device):

    patch_t = patch.clone().to(device).requires_grad_(True)

    sf_t = torch.tensor(sf, device=device, dtype=torch.float32, requires_grad=False)
    tx_t = torch.FloatTensor(1).uniform_(0, 10).to(device).requires_grad_(True)
    ty_t = torch.FloatTensor(1).uniform_(0, 10).to(device).requires_grad_(True)


    optimizer = torch.optim.Adam([patch_t, tx_t, ty_t], lr=lr, eps=1e-4)
    best_loss = np.inf
    best_patch = initial_patch.clone().detach() * 255.
    best_T = construct_T_matrix(sf_t.clone(), tx_t.clone(), ty_t.clone(), tx_min, tx_max, ty_min, ty_max, noise=False).detach()

    for epoch in trange(epochs):
        epoch_loss = torch.tensor(0., device=device)

        for data in dataset:
            
            batch, _ = data

            batch = batch.to(device) / 255.
            
            batch_T = torch.stack([construct_T_matrix(sf_t, tx_t, ty_t, tx_min, tx_max, ty_min, ty_max) for _ in range(batch.size(0))], dim=0)
            batch_patch = patch_t.expand(batch.size(0), -1, -1, -1)
            # print(batch.shape, batch_T.shape, batch_patch.shape)

            # print(batch.dtype, batch_T.dtype, batch_patch.dtype)
            manipulated_images = torch.stack([project_patch(batch_patch[i], batch_T[i], batch[i]) for i in range(batch.size(0))], dim=0).squeeze(1)
            # print(manipulated_images.shape, manipulated_images[0].grad_fn)
            # plt.imshow(manipulated_images[0].detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.show()
            #new_poses = torch.stack([sim.sim_new_pose(img*255.) for img in manipulated_images], dim=0)

            x, y, z, yaw = model(manipulated_images)
            prediction = torch.stack([x, y, z, yaw])
            prediction = prediction.squeeze(2).mT

            l2_distance = torch.stack([torch.linalg.norm((pred[:3] - target[:3]), ord=2) for pred in prediction])
            angular_loss = torch.stack([1 - torch.cos(pred[3] - target[3]) for pred in prediction])

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
            
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.item()}")
        if epoch_loss.item() < 0.1:
            print("Early stopping due to low loss")
            break
        if epoch_loss.item() < best_loss:
            best_patch = patch_t.clone().detach() * 255.
            best_T = best_T = construct_T_matrix(sf_t.clone(), tx_t.clone(), ty_t.clone(), tx_min, tx_max, ty_min, ty_max, noise=False).detach()
            best_loss = epoch_loss.item()

    
    return best_patch, best_T


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_set = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, num_workers=0, IMRC=True)
    # print("DATASET SIZE: ", len(train_set.dataset))
    
    test_set = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, train=False, num_workers=0, IMRC=True)


    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()

    initial_patch = torch.rand((1, 1, 80, 80), device=device, dtype=torch.float32)

    sf, tx_min, tx_max, ty_min, ty_max = gen_random_monitor_space(0.5, 1.5, camera_image_size=(160, 96), patch_size=(80, 80))

    epochs = 250
    lr = 3e-3

    target = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)  # Example target pose

    patch, T = joint(train_set, model, target, initial_patch, sf, tx_min, tx_max, ty_min, ty_max, epochs, lr, device)