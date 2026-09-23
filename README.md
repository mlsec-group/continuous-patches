# Come with me: Controlling Autonomous Vehicles using Continuous Adversarial Patches

This repository contains the code and artifacts required to reproduce the results of our paper:

> Pia Hanfeld, Erik Imgrund, Felix Weißberg, Thorsten Eisenhofer, Wolfgang Hönig, and Konrad Rieck. 2026.
> *Controlling Autonomous Vehicles using Continuous Adversarial Patches.*
> In 19th Workshop on Artificial Intelligence and Security (AISec '26), November 15–19, 2026, The Hague, Netherlands.
> ACM, New York, NY, USA, 12 pages. [https://doi.org/10.1145/3847352.3848104](https://doi.org/10.1145/3847352.3848104)

```bibtex
@inproceedings{hanfeld2026controlling,
  author    = {Hanfeld, Pia and Imgrund, Erik and Wei{\ss}berg, Felix and Eisenhofer, Thorsten and H{\"o}nig, Wolfgang and Rieck, Konrad},
  title     = {Controlling Autonomous Vehicles using Continuous Adversarial Patches},
  booktitle = {19th Workshop on Artificial Intelligence and Security (AISec '26)},
  year      = {2026},
  doi       = {10.1145/3847352.3848104}
}
```

A simulated drone follows a target trajectory using only visual pose estimation (PULP-Frontnet or YOLOv5); an attacker renders an adversarial patch — continuously adapted to the drone's state via a diffusion model — onto a wall monitor to hijack the drone's perception and steer it along adversary-defined trajectories (slingshot and freestyle maneuvers). All experiments are conducted in the custom simulation environment released in this repository.

## Repository layout

```
├── src/                      # Core source code (the `src` package)
│   ├── main.py               # Experiment entrypoint (python -m src.main)
│   ├── simulation.py         # Drone physics, monitor geometry, P controller
│   ├── attacks.py            # Patch projection, pose recovery, attacker
│   ├── camera.py             # Camera calibration, box -> 3D lifting
│   ├── yolo_bounding.py      # Differentiable YOLOv5 wrapper
│   ├── util.py               # Datasets, model loading, trajectories
│   └── diffusion/            # Diffusion model for patch generation
│       ├── diffusion_model.py    # UNet + EDM sampling
│       ├── diffusion_overfit.py  # Training script
│       └── legacy/           # Deprecated experimental scripts (reference only)
├── scripts/
│   ├── run_experiments_frontnet.sh  # SLURM sweep (frontnet)
│   ├── run_experiments_yolov5.sh    # SLURM sweep (yolov5)
│   └── diffusion_inference.py       # Diffusion inference sanity check
├── evaluation/
│   ├── compute_metrics.py    # Fréchet/DTW scores -> aggregated CSV
│   ├── generate_plots.py     # LaTeX tables and figures from the CSV
│   └── analyze_timing.py     # Anytime-loss / compute-time tables
├── examples/                 # Small standalone examples
├── configs/                  # Configuration files (camera calibration)
├── misc/                     # Extra dataset (gitignored, must be supplied)
├── pulp-frontnet/            # PULP-Frontnet (git submodule)
└── paper_results/            # Experiment outputs (gitignored)
```

All commands below are run from the repository root.

## Quickstart

### Option A: Apptainer container (recommended)

We package all dependencies using an [apptainer](https://apptainer.org/) container for ease of reproduction. Make sure to run
```bash
apptainer build container.sif container.def
```
before proceeding. You can start an interactive session (with support for Nvidia CUDA) using
```bash
apptainer run --nv container.sif bash
```

### Option B: Python virtual environment

For development or machines without Apptainer, install the system dependencies first, then create a venv (Python 3.12):
```bash
sudo apt-get install -y libgl1 git build-essential libnlopt-dev libnlopt-cxx-dev \
  libgoogle-glog-dev cmake libeigen3-dev libboost-all-dev

python -m venv .venv
source .venv/bin/activate
pip install -r apptainer_requirements.txt
```

Note: the first `yolov5` run downloads the `yolov5s` weights via `torch.hub` and therefore requires network access.

## Reproducing the AISEC'26 results

We evaluate two attack scenarios, both against PULP-Frontnet and YOLOv5:

- **Slingshot attacks** (Section 5.1, Table 1): aggressively accelerate the drone in a target direction (`forward`, `backward`, `left`, `right`). Metric: displacement projected onto the target direction in meters (higher is better).
- **Freestyle attacks** (Section 5.3, Table 2): guide the drone along complex trajectories (`triangle`, Fig. 6; `figure8` — the paper's infinity symbol, Fig. 8), each defined by 25 waypoints. Metric: Fréchet distance to the target trajectory in meters (lower is better).

Each configuration is repeated across three random seeds (`--seed 0 1 2`); results are averaged and reported with standard deviations.

### Prerequisites

1. Initialize the Frontnet submodule: `git submodule update --init`
2. Place the dataset files:
   - `pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle` (expected SHA256 in `Data/checksums.txt`)
   - `misc/IMRC_images.pickle` (dataset extension used by `src/util.py`)
3. Train the diffusion model (Step 1 below). Trained weights are **not released** — see Open Science note at the bottom.
4. FAP baseline: TODO docu

### Step 1: Train the diffusion model

The attack model is a diffusion model (EDM-style UNet) conditioned on the display constraint `[sf, tx, ty]` and the target pose `[x, y, z, yaw]`, trained on a corpus of 1000 optimal patches (one per random condition):

```bash
python -m src.diffusion.diffusion_overfit --model frontnet --corpus_size 1000 --batch_size 64
# for YOLOv5:
python -m src.diffusion.diffusion_overfit --model yolov5 --corpus_size 1000 --batch_size 64
```

- Output: `flipped_diffusion/{model}/diffusion_model_{model}_1000.pth` (plus loss curves and sample grids in the same directory).
- The paper reports training on an NVIDIA A100 for under 4 hours (1000 epochs, batch size 32, Adam lr 1e-4). The script currently trains for 2000 epochs with a fixed Adam lr of 1e-3 — adjust in `src/diffusion/diffusion_overfit.py` if you want the exact paper hyperparameters.
- The script loads `{model}/corpus_{model}.npz` if present
- **TODO**: FAP integration documentation to generate `{model}/corpus_{model}.npz`

Sanity-check the trained model:
```bash
python scripts/diffusion_inference.py -m frontnet --steps 25 --candidates 5
```

### Step 2: Run experiments

#### Full sweep on a SLURM cluster

```bash
bash scripts/run_experiments_frontnet.sh   # submits a job array to partition gpu-9m (80 GB constraint)
bash scripts/run_experiments_yolov5.sh
```

Edit the parameter arrays at the top of the scripts (`PATCH_MODES`, `TRAJECTORIES`, `DISPLAY_SIZES`, `TIMEOUT_VALUES`, `SEED_VALUES`) to select the sweep. For the paper's grids: `TRAJECTORIES` including `slingshot_backward` (and the other slingshot directions for Fig. 5/7) plus `triangle`/`figure8` for freestyle; `DISPLAY_SIZES=(60 80 100 120)`; `TIMEOUT_VALUES=(10)` for the optimization modes; seeds 0, 1, 2.

#### Single experiment (without a cluster)

Inside the Apptainer container or an activated venv, from the repository root:

```bash
# Slingshot (paper's focus direction), our diffusion attack, 10 Hz control loop
python -m src.main -m frontnet -t slingshot_backward --patch_mode diffusion \
  --display_size 60 --seed 0 --pic_mode idx --img_idx 1860 \
  --corpus_size 1000 --timeout 10 --temperature warm

# Freestyle (infinity symbol), same settings
python -m src.main -m frontnet -t figure8 --patch_mode diffusion \
  --display_size 60 --seed 0 --pic_mode idx --img_idx 1860 \
  --corpus_size 1000 --timeout 10 --temperature warm
```

Each run writes one PNG per step plus `all_drone_poses.npy`, `time_per_step.npy`, and `all_velocity_commands.npy` under `paper_results/{model}/{patch_mode}/[...]/image_{idx}/{seed}/`.

| Parameter | Values | Meaning |
|---|---|---|
| `-m` | `frontnet`, `yolov5` | victim detection model |
| `-t` | `slingshot_forward`, `slingshot_backward`, `slingshot_left`, `slingshot_right`, `figure8` (infinity), `triangle`, `square`, `circle`, `line_x`, `line_y`, `diagonal_line`, `c`, `s`, `u` | target trajectory (25 waypoints) |
| `--patch_mode` | `diffusion`, `fap`, `corpus`, `interpolation`, `timeout`, `velo`, `black`, `white`, `random`, `none`, `optimal` | attack method (see baseline table below) |
| `--display_size` | e.g. `60 80 100 120` | monitor diagonal in inches (16:9) |
| `--timeout` | `10`, `0` | `10` = paper's 10 Hz control loop (optimization capped at 1/10 s per step); `0` = unlimited optimization, dt = 1/30 s |
| `--temperature` | `warm`, `cold`, `none` | restart patch optimization from previous patch (`warm`) or from random (`cold`); only for optimization modes |
| `--corpus_size` | `1000`, `2000`, `3000` | corpus size for `diffusion`/`corpus`/`interpolation` |
| `--pic_mode` / `--img_idx` | `idx` / `1860 4861 5431`, or `random` | background image selection (paper images: 1860, 4861, 5431) |
| `--seed` | `0 1 2` | random seed (3 seeds per configuration in the paper) |

### Step 3: Compute metrics

```bash
python evaluation/compute_metrics.py
```

The script scans `paper_results/` for `all_drone_poses.npy` files, computes per-run scores against the corresponding target trajectory, and aggregates everything into `paper_results/all_results_frontnet.csv` (plus a pickle). It provides the metrics used in the paper:

- `compute_frechet_distance(target_traj, actual_traj)` — discrete Fréchet distance, the primary freestyle metric (Table 2)
- `compute_dtw_distance(target_traj, actual_traj)` — normalized DTW distance, the alternative trajectory-similarity metric

Note the parameter grids (trajectories, display sizes, patch modes) are fixed at the top of the script — adapt them to the sweep you ran. For slingshot displacement (Table 1), project `all_drone_poses` onto the target direction and normalize by the displacement of the `optimal` run.

### Step 4: Aggregate and plot

```bash
python evaluation/generate_plots.py            # LaTeX tables (paper_results/tables/) grouped by trajectory
python evaluation/analyze_timing.py            # anytime-loss / compute-time tables
```

### Baselines (Tables 1 and 2)

All baselines share the same trajectory/display/seed settings; only `--patch_mode` (and a few flags) change:

| Paper row | Command (same flags as Step 2, except:) |
|---|---|
| None (unaltered input) | `--patch_mode none --temperature none` |
| FAP [13] (static patches) | `--patch_mode fap --temperature none` (needs `{model}/fap/` artifacts) |
| PGD (10 Hz upper bound, direct camera access) | `--patch_mode timeout --timeout 10` |
| Interpolation (2-NN) | `--patch_mode interpolation --corpus_size 1000` |
| Corpus (nearest neighbor) | `--patch_mode corpus --corpus_size 1000` |
| Ours (continuous patches) | `--patch_mode diffusion --corpus_size 1000` |

`--patch_mode optimal` is the idealized upper bound (perceived pose is set to the target pose) used for normalization; static baselines additionally have `black`/`white`/`random` variants.

### Simulation setup (paper Section 4.1)

- Confined 3D space; run aborts when the drone leaves x/y ∈ [-2, 2] m, z ∈ [0.1, 2] m.
- Proportional controller (DJI-Neo-like): position gain 1.5 with velocity clamped at 15 m/s, yaw-rate gain 3.0 clamped at 3 rad/s — set in `DroneSimulation` (`src/simulation.py`).
- The paper's 10 Hz simulation loop corresponds to `--timeout 10`.
- Monitor centered 2 m in front of the drone at height 1 m, facing the drone; size via `--display_size`.
- Camera images drawn from the extended Frontnet indoor dataset; the dataset is mirrored left-to-right at load time (doubled) to correct the original right-side bias.
- Controller/monitor parameters are defined in `src/simulation.py` and `configs/camera_calibration.yaml`.

## Dependencies

**System** (installed by `container.def`; install manually for the venv option): `libgl1`, `git`, `build-essential`, `libnlopt-dev`, `libnlopt-cxx-dev`, `libgoogle-glog-dev`, `cmake`, `libeigen3-dev`, `libboost-all-dev`.

**Python** (Python 3.12, see `apptainer_requirements.txt`): `torch`, `torchvision`, `torchaudio`, `torchsummary`, `numpy`, `tqdm`, `rowan`, `pandas`, `opencv-python`, `PyYAML`, `scikit-learn`, `setuptools==80.4.0`, `matplotlib`, `requests`, `py-cpuinfo`, `psutil`, `seaborn`, `ultralytics-thop`, `ultralytics`, `gitpython`.

## Legacy code

`src/diffusion/legacy/` contains experimental scripts from earlier development phases that are not part of the main code and may not run against the current codebase; see the README in that directory.

## Third-party code

We used the original implementation to train [Flying Adversarial Patches](https://github.com/IMRCLab/flying_adversarial_patch/) (the FAP baseline, Tables 1 and 2) and our own implementation for all other baselines. The PULP-Frontnet model is provided as the `pulp-frontnet/` git submodule.

## Open Science

In line with the paper, we do **not** release the trained diffusion model weights: they could be readily misused to generate continuous adversarial patches in practice. We instead provide the training script (`src/diffusion/diffusion_overfit.py`) so the models can be trained given sufficient hardware. All experiments are simulation-only; physical validation is future work.
