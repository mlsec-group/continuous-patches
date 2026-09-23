"""Core source code for the continuous adversarial patch drone simulation.

Modules:
    main          -- experiment entrypoint (run with ``python -m src.main``)
    simulation    -- drone physics, monitor geometry, and P controller
    attacks       -- patch projection, pose recovery, and the Attacker
    camera        -- camera calibration and box-to-3D lifting
    yolo_bounding -- differentiable YOLOv5 wrapper
    util          -- dataset loading, model loading, and trajectories
    diffusion     -- diffusion model for patch generation
"""
