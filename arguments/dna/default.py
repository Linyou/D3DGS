ModelParams = dict(
    use_extra_cam_info=True,
    white_background=True,
)

PipelineParams = dict(
    load_img_factor=0.5,
    # real_dynamic=True,
)

FlowParams = dict(
    xyz_traj_feat_dim = 16,
    xyz_trajectory_type = 'fft_poly',
    rot_traj_feat_dim = 32,
    rot_trajectory_type = 'fft_poly',
    # scale_traj_feat_dim = 2,
    # scale_trajectory_type = 'fft_poly',
    # opc_traj_feat_dim = 2,
    # opc_trajectory_type = 'fft_poly',
    # feature_traj_feat_dim = 2,
    feature_traj_feat_dim = 16,
    feature_trajectory_type = 'fft_poly',
    traj_init = 'zero',
    poly_base_factor = 1,
    Hz_base_factor = 1,
    normliaze = False,
    
    #training
    get_smooth_loss=False,
    use_interpolation=False,
    random_noise=False,
    get_moving_loss=False,
    masked=False,
    
    #lr
    # xyz_lr = 0.001,
    # rot_lr = 0.001,
    # scale_lr = 0.001,
    # opc_lr = 0.001,
    # feature_lr = 0.001,
)

OptimizationParams = dict(
    iterations = 30_000,
    position_lr_init =  0.0008,
    position_lr_final = 0.0000008,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 30_000,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.005,
    percent_dense = 0.01,
    lambda_dssim = 0.2,
    densification_interval = 100,
    opacity_reset_interval = 3000,
    densify_from_iter = 500,
    densify_until_iter = 15_000,
    densify_grad_threshold = 0.0002,
    min_opacity = 0.001,
    batch_size=1,
    dataloader=True,
    knn_loss=False,
    scale_loss=False,
    # knn_selec_thresh=1.0,
    no_deform_from_iter=0,
)