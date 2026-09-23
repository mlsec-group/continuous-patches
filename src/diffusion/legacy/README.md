# Legacy diffusion experiments

This directory contains experimental scripts from earlier development
phases (USenix artifact preparation and early diffusion-model experiments).
They are **not part of the released AISec'26 artifact** and are not
maintained: several of them reference modules or workflows (e.g.,
`create_dataset`, `patch_placement`, an external
`flying_adversarial_patch` checkout) that are no longer present in this
repository.

They are kept for reference only. The current, supported workflows are:

- Training the patch diffusion model: `python -m src.diffusion.diffusion_overfit`
- Inference sanity check: `python scripts/diffusion_inference.py`
