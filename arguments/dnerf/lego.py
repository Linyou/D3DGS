FlowParams = dict(
    xyz_traj_feat_dim = 1,
    xyz_trajectory_type = 'fft_poly',
    rot_traj_feat_dim = 2,
    rot_trajectory_type = 'fft_poly',
    # feature_traj_feat_dim = 2,
    feature_traj_feat_dim = 1,
    feature_trajectory_type = 'fft_poly',
    traj_init = 'zero',
    poly_base_factor = 1,
    Hz_base_factor = 1,
)


OptimizationParams = dict(
    iterations = 30_000,
    position_lr_init = 0.000001,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 30_000,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.001,
    rotation_lr = 0.000001,
    percent_dense = 0.01,
    lambda_dssim = 0.2,
    densification_interval = 100,
    opacity_reset_interval = 10000000,
    densify_from_iter = 100,
    densify_until_iter = 15_000,
    densify_grad_threshold = 0.0005,
    min_opacity = 0.005,
    batch_size=1,
)