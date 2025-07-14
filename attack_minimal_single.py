import torch
import numpy as np

import os

from tqdm import trange


from util import load_model, load_dataset

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

def norm_transformation(sf, tx, ty, scale_min, scale_max, tx_min, tx_max, ty_min, ty_max):
    # new patch placement implementation might need different tx, ty limits!:
    #sf_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min
    #tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    #ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min

    sf_norm = single_norm(sf, scale_min, scale_max)
    tx_norm = single_norm(tx, tx_min, tx_max)
    ty_norm = single_norm(ty, ty_min, ty_max)



    return sf_norm, tx_norm, ty_norm

def noisy_transformations(sf, tx, ty):
    sf_n = sf + np.random.normal(0.0, 0.1)
    tx_n = tx + np.random.normal(0.0, 0.1)
    ty_n = ty + np.random.normal(0.0, 0.1)
    return sf_n, tx_n, ty_n

def construct_T_matrix(sf, tx, ty, scale_min=0.2, scale_max=0.575, tx_min=48., tx_max=128., ty_min=20., ty_max=66., noise=True):
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
    norm_scale, norm_tx, norm_ty = norm_transformation(sf=sf, tx=tx, ty=ty, scale_min=scale_min, scale_max=scale_max, tx_min=tx_min, tx_max=tx_max, ty_min=ty_min, ty_max=ty_max)
    # print(norm_tx, norm_ty)

    T_matrix = torch.zeros((3, 3), device=sf.device)
    scale_T = torch.eye(2, device=sf.device) * norm_scale
    T_matrix[:2, :2] = scale_T
    T_matrix[0, 2] = norm_tx
    T_matrix[1, 2] = norm_ty
    T_matrix[2, 2] = 1.0
    # print(T_matrix)
    return T_matrix

def single_norm(value, min_value, max_value):
    norm_value = (max_value - min_value) * (torch.tanh(value) + 1) * 0.5 + min_value
    return norm_value

def inverse_norm(norm_value, min_value, max_value):
    inverse_norm = torch.atanh(2 * (norm_value - min_value) / (max_value - min_value) - 1)
    return inverse_norm


def calc_heading_vec(radius, angle):
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    zero = torch.zeros_like(x, device=angle.device, dtype=torch.float32)
    return torch.stack((x, y, zero), dim=-1)


def T_matrix(pose):
    T = torch.eye(4, dtype=torch.float32).to(pose.device)
    normalized_yaw = normalize_yaw_t(pose[3])

    sin_yaw = torch.sin(normalized_yaw).to(torch.float32)
    cos_yaw = torch.cos(normalized_yaw).to(torch.float32)
    zero = torch.zeros_like(sin_yaw, device=pose.device, dtype=torch.float32)

    rotation_matrix_row1 = torch.stack([cos_yaw, -sin_yaw, zero], dim=-1)
    rotation_matrix_row2 = torch.stack([sin_yaw, cos_yaw, zero], dim=-1)
    rotation_matrix_row3 = torch.tensor([0., 0., 1.], device=pose.device, dtype=torch.float32)
    R = torch.stack((rotation_matrix_row1, rotation_matrix_row2, rotation_matrix_row3), dim=0)

    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T



os.makedirs('test_single/', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("pulp-frontnet/PyTorch/Models/Frontnet160x32.pt", device, config="160x32")
model.eval()


dataset = load_dataset("pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle", batch_size = 32, shuffle = False, drop_last = True, num_workers = 1, train=True, train_set_size=0.9, IMRC=True)

print(len(dataset))

# img = torch.ones((1, 1, 96, 160), device=device, dtype=torch.float32) * 0.5  # gray image
img = dataset.dataset[0][0].to(device).unsqueeze(0) / 255.0
print(img.shape)

all_drone_poses = []


T_drone_in_world = torch.eye(4, device=device, dtype=torch.float32)
T_drone_in_world[:3, 3] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)  # Initial position

all_drone_poses.append([*T_drone_in_world[:3, 3].clone().detach().cpu().numpy(), 0.0])

t = np.linspace(0, 2 * np.pi, 20)
x = 0.5 * np.sin(2 * t)  # Horizontal figure 8
y = 1.5 * np.sin(t)  # Vertical figure 8
z = np.ones_like(t)  # Constant height at 1
yaw = np.zeros_like(t)  # Constant yaw
target_trajectory = np.column_stack((x, y, z, yaw))
target_trajectory = torch.tensor(target_trajectory, dtype=torch.float32, device=device)


for target_idx in trange(1, len(target_trajectory)):

    img = dataset.dataset[0][0].to(device).unsqueeze(0) / 255.0

    target = target_trajectory[target_idx]

    patch = torch.rand((1, 1, 80, 80), device=device, dtype=torch.float32, requires_grad=True)  # random patch

    sf = torch.tensor(1.0, device=device, dtype=torch.float32)  # scale factor
    tx = torch.tensor(-1.0, device=device, dtype=torch.float32, requires_grad=True)  # translation x
    ty = torch.tensor(-0.8, device=device, dtype=torch.float32, requires_grad=True)  # translation y


    opt = torch.optim.Adam([{'params': [tx, ty], 'lr': 0.03},
                       {'params': [patch], 'lr': 1e-2}], lr=1e-3)


    # # calculate target
    # T_setpoint_world = torch.eye(4, device=device, dtype=torch.float32)
    # T_setpoint_world[:3, 3] = target_trajectory[1, :3]
    # print("Setpoint world transformation matrix:")
    # print(T_setpoint_world)

    # T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
    # T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_trajectory[1, 3]-np.pi)).to(device)
    # print("Direction in world: ")
    # print(T_direction_world)


    # T_pred_in_world = torch.inverse(T_direction_world) @ T_setpoint_world
    # print("Predicted transformation matrix in world frame:")
    # print(T_pred_in_world)

    

    # T_pred_in_drone = torch.inverse(T_drone_in_world) @ T_pred_in_world
    # print("Predicted transformation matrix in drone frame:")
    # print(T_pred_in_drone)
    # # print(T)

    # target = torch.stack([*T_pred_in_drone[:3, 3], target_trajectory[0, 3]]).to(device) # target values
    # print("Target values:", target)

    loss = torch.inf
    i = 0

    best_loss = torch.inf
    best_patch = patch.clone()
    best_sf = sf.clone()
    best_tx = tx.clone()
    best_ty = ty.clone()
    best_setpoint = None

    while loss > 0.01 and i < 5000:
        opt.zero_grad()


        T = construct_T_matrix(
        sf=sf, tx=tx, ty=ty, 
        scale_min=0.2, scale_max=0.6, 
        tx_min=0., tx_max=160., 
        ty_min=-0., ty_max=96.,
        noise=False
        )

        manipulated_image = project_patch(
            patches=patch, 
            T_matrices=T.unsqueeze(0),  # add batch dimension
            images=img
        )



        # print("Manipulated image shape:", manipulated_image.shape)



        # print(manipulated_image.min(), manipulated_image.max())


        x, y, z, yaw = model(manipulated_image*255.)
        # print("x, y, z, yaw:", x, y, z, yaw)
        prediction = torch.stack([x, y, z, yaw])
        prediction = prediction.squeeze(2).mT

        T_pred_in_drone = T_matrix(prediction[0])
        T_pred_in_world = T_drone_in_world @ T_pred_in_drone
        # print("T_pred_in_world within loop:")
        # print(T_pred_in_world)

        target_yaw = normalize_yaw_t(prediction[0, 3])
        T_direction_world = torch.eye(4, device=device, dtype=torch.float32)
        T_direction_world[:3, 3] = calc_heading_vec(1., normalize_yaw_t(target_yaw - torch.pi)).to(device)
        # print("Direction in world within loop:")
        # print(T_direction_world)

        T_setpoint_world = T_direction_world @ T_pred_in_world
        # print("Setpoint world transformation matrix within loop:")
        # print(T_setpoint_world)
        # target = torch.stack([*T_setpoint_world[:3, 3], target_yaw]).to(device)
        

        prediction = torch.stack([*T_setpoint_world[:3, 3], target_yaw]).to(device).unsqueeze(0)  # prediction values



        # print("Prediction: ", prediction)

        # target = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)  # target values

        distance = torch.norm(prediction[0, :3] - target[:3], p=2)
        angular_loss = 1 - torch.cos(normalize_yaw_t(prediction[0, 3]) - normalize_yaw_t(target[3]))

        loss = distance + angular_loss

        # if i % 25 == 0:
        #     # print(f"Iteration {i}:")
        #     # print("Scale factor: ", sf.item())
        #     # print("Translation x: ", tx.item())
        #     # print("Translation y: ", ty.item())

        #     print("Prediction: ", prediction.detach().cpu().numpy())
        #     print("Loss: ", loss.detach().cpu().item())
            # print('tx, ty:', tx.item(), ty.item())
        # print("Loss: ", loss.detach().cpu().item())

        loss.backward()
        opt.step()

        patch.data.clamp_(0., 1.)
        i += 1

        if loss < best_loss:
            best_loss = loss.detach().detach().clone()
            best_patch = patch.detach().clone()
            best_sf = sf.detach().clone()
            best_tx = tx.detach().clone()
            best_ty = ty.detach().clone()
            best_setpoint = prediction[0].detach().clone()


    np.save(f'test_single/patch_{target_idx}.npy', best_patch.detach().cpu().numpy())
    T = construct_T_matrix(
        sf=best_sf, tx=best_tx, ty=best_ty, 
        scale_min=0.2, scale_max=0.6, 
        tx_min=0., tx_max=160., 
        ty_min=-0., ty_max=96.,
        noise=False
    )
    np.save(f'test_single/T_{target_idx}.npy', T.detach().cpu().numpy())

    with torch.no_grad():
        manipulated_image = project_patch(
            patches=best_patch, 
            T_matrices=T.unsqueeze(0),  # add batch dimension
            images=img
        )

    T_drone_in_world = T_matrix(best_setpoint)
    all_drone_poses.append(best_setpoint.numpy())
    print(all_drone_poses)

    print("Current drone pose: ", best_setpoint)
    print("Target pose that was to be reached: ", target)
    

    print("Iterations needed: ", i)
    import matplotlib.pyplot as plt
    plt.imshow(manipulated_image[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.savefig('test_single/manipulated_image_{}.png'.format(target_idx))
    plt.close()

all_drone_poses = np.array(all_drone_poses)
print(all_drone_poses.shape)

fig, ax = plt.subplots(1, 1)
ax.plot(target_trajectory[:, 0].detach().numpy(), target_trajectory[:, 1].detach().numpy(), 'r--')
ax.plot(all_drone_poses[:, 0], all_drone_poses[:, 1])
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('test_single/trajectory.png')
plt.close()