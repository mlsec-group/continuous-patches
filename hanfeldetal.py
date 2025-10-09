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


def _perspective_grid(coeffs, w, h, ow, oh, dtype, device, center=None):
    batch_size = coeffs.shape[0]
    theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)

    theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0).reshape(batch_size, 2, 3)

    d = 0.5
    base_grid = torch.empty(batch_size, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)

    base_grid[..., 0].copy_(x_grid)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], device=device, dtype=dtype))
    shape = (batch_size, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    center = torch.as_tensor(center if center is not None else 1.0, dtype=dtype, device=device)
    output_grid = output_grid1.div_(output_grid2).sub_(center)

    return output_grid.view(batch_size, oh, ow, 2)


def project_patch(patches, T_matrices, images):
    device = images.device
    batch_size, _, p_height, p_width = patches.shape
    i_height, i_width = images.shape[-2:]

    while images.dim() < 4:
        images = images.unsqueeze(0)

    masks = torch.ones_like(patches, dtype=torch.float32, device=device)

    inv_T_matrices = torch.inverse(T_matrices)
    coeffs = inv_T_matrices.reshape(batch_size, -1)

    grids = _perspective_grid(coeffs, p_width, p_height, i_width, i_height, torch.float32, device, center=[1., 1.])

    transformed_patches = torch.nn.functional.grid_sample(
        patches, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    )
    bit_masks = torch.nn.functional.grid_sample(
        masks, grids, mode='bilinear', align_corners=False, padding_mode='zeros'
    ).bool()

    manipulated_images = images * ~bit_masks + transformed_patches
    return manipulated_images


def norm_transformation(sf, tx, ty, sf_min, sf_max, tx_min, tx_max, ty_min, ty_max):
    sf_norm = (sf_max - sf_min) * (torch.tanh(sf) + 1) * 0.5 + sf_min
    tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min
    return sf_norm, tx_norm, ty_norm


def construct_T_matrix_batch(sf, tx, ty, sf_min=0.5, sf_max=1.2, tx_min=0., tx_max=160.,
                             ty_min=0., ty_max=96., noise=True):
    if noise:
        noise_std = 0.1
        sf = sf + torch.randn_like(sf) * noise_std
        tx = tx + torch.randn_like(tx) * noise_std
        ty = ty + torch.randn_like(ty) * noise_std

    norm_sf, norm_tx, norm_ty = norm_transformation(sf, tx, ty, sf_min, sf_max, tx_min, tx_max, ty_min, ty_max)

    N = norm_sf.shape[0]
    T = torch.zeros(N, 3, 3, device=norm_sf.device, dtype=norm_sf.dtype)
    T[:, 0, 0] = norm_sf
    T[:, 1, 1] = norm_sf
    T[:, 0, 2] = norm_tx
    T[:, 1, 2] = norm_ty
    T[:, 2, 2] = 1.0
    return T


def calc_eval_loss(test_set, model, patch, T, target):
    device = patch.device
    patch = patch.to(device)
    T = T.to(device)
    total_loss = 0.0
    with torch.no_grad():
        for batch, _ in test_set:
            batch = batch.to(device, non_blocking=True).float() / 255.0
            batch_patch = patch.expand(batch.size(0), -1, -1, -1)
            batch_T = T.expand(batch.size(0), -1, -1)

            manipulated_images = project_patch(batch_patch, batch_T, batch).clamp_(0., 1.)
            x, y, z, yaw = model(manipulated_images * 255.)
            prediction = torch.stack([x, y, z, yaw]).squeeze(2).mT

            l2_distance = torch.linalg.norm(prediction[:, :3] - target[:3], dim=1)
            angular_loss = 1 - torch.cos(normalize_yaw_t(prediction[:, 3]) - normalize_yaw_t(target[3]))
            loss = (l2_distance + angular_loss).mean()
            total_loss += loss.item()

    return total_loss / len(test_set)


def joint(train_set, test_set, model, targets, patches, positions, assignment,
          sf_min, sf_max, tx_min, tx_max, ty_min, ty_max, epochs, lr, device, prob_weight=5):

    patches_t = torch.stack([p.clone().detach() for p in patches], dim=0).to(device).requires_grad_(True)
    positions_t = positions.clone().detach().to(device).requires_grad_(True)
    assignment_t = torch.as_tensor(assignment, device=device, dtype=torch.bool)

    T_dim, P_dim, R_dim = len(targets), *positions_t.shape[:2]
    assert assignment_t.shape == (T_dim, P_dim, R_dim)

    optimizer = torch.optim.Adam([patches_t, positions_t], lr=lr)

    best_patches, best_positions, best_assignments = [None] * T_dim, [None] * T_dim, [-1] * T_dim
    best_losses = [float('inf')] * T_dim
    train_losses, eval_losses, stats_history, best_assignments_history = [], [], [], []

    for epoch in trange(epochs):
        epoch_loss = torch.tensor(0., device=device)
        stats_sum = torch.zeros(T_dim, P_dim, R_dim, device=device)
        stats_count = torch.zeros_like(stats_sum)

        for batch, _ in train_set:
            batch = batch.to(device, non_blocking=True).float() / 255.0
            B = batch.shape[0]
            target_terms = []

            for t_idx, target in enumerate(targets):
                active_pairs = torch.nonzero(assignment_t[t_idx], as_tuple=False)
                if active_pairs.numel() == 0:
                    continue

                p_idx, r_idx = active_pairs[:, 0], active_pairs[:, 1]
                k = p_idx.shape[0]

                patch_batches = patches_t.index_select(0, p_idx).unsqueeze(1).expand(-1, B, -1, -1, -1).reshape(k * B, *patches_t.shape[1:])
                pos_kb3 = positions_t[p_idx, r_idx].unsqueeze(1).expand(k, B, 3).reshape(k * B, 3)

                T_kb = construct_T_matrix_batch(pos_kb3[:, 0], pos_kb3[:, 1], pos_kb3[:, 2],
                                                sf_min, sf_max, tx_min, tx_max, ty_min, ty_max, noise=True)
                batch_multi = batch.repeat(k, 1, 1, 1)

                mod_img = project_patch(patch_batches, T_kb, batch_multi).clamp_(0., 1.)
                x, y, z, yaw = model(mod_img * 255.)
                pred = torch.stack([x, y, z, yaw]).squeeze(2).mT

                l2 = torch.linalg.norm(pred[:, :3] - target[:3], dim=1)
                ang = 1 - torch.cos(normalize_yaw_t(pred[:, 3]) - normalize_yaw_t(target[3]))
                loss_per_pair = (l2 + ang).view(k, B).mean(dim=1)

                stats_sum[t_idx, p_idx, r_idx] += loss_per_pair.detach() * B
                stats_count[t_idx, p_idx, r_idx] += B

                probs = torch.nn.functional.softmin(loss_per_pair * prob_weight, dim=0)
                target_terms.append((probs * loss_per_pair).sum())

                min_idx = torch.argmin(loss_per_pair)
                if (val := loss_per_pair[min_idx].item()) < best_losses[t_idx]:
                    best_losses[t_idx] = val
                    best_assignments[t_idx] = p_idx[min_idx].item()
                    best_patches[t_idx] = patches_t[best_assignments[t_idx]].detach().clone()
                    best_positions[t_idx] = positions_t[p_idx[min_idx], r_idx[min_idx]].detach().clone()

            if target_terms:
                loss = torch.stack(target_terms).sum()
                epoch_loss += loss.detach()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        patches_t.data.clamp_(0., 1.)
        epoch_loss /= len(train_set)
        train_losses.append(epoch_loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.item():.6f}")

        with torch.no_grad():
            avg_loss_matrix = torch.where(stats_count > 0, stats_sum / stats_count, torch.full_like(stats_sum, float('nan')))
        stats_history.append(avg_loss_matrix.cpu().numpy())

        print("Per-(target, patch, position) avg loss this epoch:")
        for t in range(T_dim):
            print(f"  Target {t}:")
            print(avg_loss_matrix[t].detach().cpu().numpy())



        best_assignments_history.append(np.array(best_assignments, dtype=np.int64))

    return best_patches, best_positions, best_assignments, train_losses, eval_losses, stats_history, best_assignments_history


if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='results/minimal_test/', help='Path to save results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate for the optimizer')
    args = parser.parse_args()


    results_path = Path(args.results_path)
    os.makedirs(results_path, exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    # If load_dataset supports pin_memory/num_workers, enable them for GPU throughput
    train_set = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False,
                             num_workers=2 if device.type == "cuda" else 0, IMRC=True)
    test_set = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=False, train=False,
                            num_workers=2 if device.type == "cuda" else 0, IMRC=True)

    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()

    # Initialize two patches
    patch_size = (1, 45, 80)
    patches = [
        torch.rand(patch_size, device=device, dtype=torch.float32),
        torch.rand(patch_size, device=device, dtype=torch.float32)
    ]

    # Define two targets
    targets = [
        torch.tensor([0.5, 0., 0., 0.], device=device, dtype=torch.float32),
        torch.tensor([1., 0., 0., 0.], device=device, dtype=torch.float32),
        torch.tensor([1.5, 0., 0., 0.], device=device, dtype=torch.float32),
        torch.tensor([0., 1., 0., 0.], device=device, dtype=torch.float32),
        torch.tensor([0., -1., 0., 0.], device=device, dtype=torch.float32)
    ]

    # Randomly sample positions for each patch (shared across targets)
    num_positions = len(targets)  # R
    positions = torch.FloatTensor(len(patches), num_positions, 3).uniform_(-1., 1.).to(device)
    # 3D mapping: assignment[target_idx, patch_idx, pos_idx]
    assignment = np.ones((len(targets), len(patches), num_positions), dtype=np.bool_)

    # Training settings
    sf_min, sf_max = 0.5, 1.1
    tx_min, tx_max = 0., 160.
    ty_min, ty_max = 0., 96.
    epochs = args.epochs
    lr = args.lr

    # Save settings
    settings = {
        'results_path': str(results_path),
        'seed': seed,
        'patch_size': list(patch_size[-2:]),
        'sf_min': sf_min,
        'sf_max': sf_max,
        'tx_min': tx_min,
        'tx_max': tx_max,
        'ty_min': ty_min,
        'ty_max': ty_max,
        'epochs': epochs,
        'lr': lr,
        'targets': [target.cpu().numpy().tolist() for target in targets]
        # 'positions': positions.detach().cpu().numpy().tolist()
    }
    with open(results_path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)

    # Optimize patches
    best_patches, best_positions, best_assignments, train_losses, eval_losses, stats_history, best_assignments_history = joint(
        train_set, test_set, model, targets, patches, positions, assignment,
        sf_min=sf_min, sf_max=sf_max, tx_min=tx_min, tx_max=tx_max, ty_min=ty_min, ty_max=ty_max,
        epochs=epochs, lr=lr, device=device
    )

    # Print results
    for target_idx, patch_idx in enumerate(best_assignments):
        print(f"Target {target_idx} is best matched with Patch {patch_idx}")
        if patch_idx != -1:  # Ensure there is a valid assignment
            optimal_position = best_positions[target_idx]
            print(f"Optimal position for Target {target_idx} and Patch {patch_idx}: {optimal_position.cpu().numpy()}")

    # Save results
    for i, patch in enumerate(best_patches):
        np.save(results_path / f'patch_{i}.npy', patch.cpu().numpy())

    # Save best positions (sf, tx, ty) per target and the patch->target mapping
    np.save(results_path / 'best_positions.npy',
            np.stack([pos.detach().cpu().numpy() for pos in best_positions]))
    np.save(results_path / 'best_assignments.npy', np.array(best_assignments, dtype=np.int64))
    # Optional: human-readable mapping
    with open(results_path / 'best_mapping.yaml', 'w') as f:
        yaml.dump({f"target_{i}": int(idx) for i, idx in enumerate(best_assignments)}, f)
    np.save(results_path / 'train_losses.npy', np.array(train_losses))
    np.save(results_path / 'eval_losses.npy', np.array(eval_losses))

    # Plot losses
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_path / 'losses.png')
    plt.close()