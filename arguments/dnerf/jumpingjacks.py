_base_ = './dnerf_default.py'

FlowParams = dict(
    xyz_traj_feat_dim = 8,
    xyz_trajectory_type = 'fft_poly',
    rot_traj_feat_dim = 8,
    rot_trajectory_type = 'fft_poly',
    scale_traj_feat_dim = 2,
    scale_trajectory_type = 'fft_poly',
    # feature_traj_feat_dim = 2,
    feature_traj_feat_dim = 4,
    feature_dc_trajectory_type = 'fft_poly',
    traj_init = 'zero',
    poly_base_factor = 1,
    Hz_base_factor = 1,
    
    # regularization
    random_noise = True,
)


OptimizationParams = dict(
    iterations = 20_000,
    position_lr_init =  0.00008,
    position_lr_final = 0.0000008,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 20_000,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.005,
    percent_dense = 0.01,
    lambda_dssim = 0.01,
    densification_interval = 100,
    prune_interval = 100,
    opacity_reset_interval = 1000,
    densify_from_iter = 500,
    densify_until_iter = 5_000,
    densify_grad_threshold = 0.00005,
    min_opacity = 0.001,
    batch_size=8,
    no_deform_from_iter=0,
    detach_base_iter=10_000,
    # knn_loss = True,
    factor_t=True,
    factor_t_value=0.5,
)