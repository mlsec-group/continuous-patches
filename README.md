# Come with me: Controlling Autonomous Vehicles using Continuous Adversarial Patches

This repository contains the anoymized code and artifacts required to reproduce the results for our submission.


## Quickstart

We package all dependencies using an [apptainer](https://apptainer.org/) container for ease of reproduction. Make sure to run
```bash
apptainer build container.sif container.def
```
before proceeding. You can start an interactive session (with support for Nvidia CUDA) using
```bash
apptainer run --nv container.sif bash
```

Our experiments assume that you are using a SLURM cluster to run them.

### Third-party code

We used the original implementation to train [Flying Adversarial Patches](https://github.com/IMRCLab/flying_adversarial_patch/) and our own implementation for all other baselines.