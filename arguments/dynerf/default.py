# ModelParams = dict(
#     use_extra_cam_info=True,
# )

# PipelineParams = dict(
#     # load_img_factor=0.5,
#     real_dynamic=True,
# )

FlowParams = dict(
    xyz_traj_feat_dim = 16,
    xyz_trajectory_type = 'fft_poly',
    rot_traj_feat_dim = 16,
    rot_trajectory_type = 'fft_poly',
    # feature_traj_feat_dim = 2,
    feature_traj_feat_dim = 8,
    feature_trajectory_type = 'fft_poly',
    traj_init = 'zero',
    poly_base_factor = 1,
    Hz_base_factor = 1,
    
    #training
    get_smooth_loss=False,
    use_interpolation=False,
    random_noise=False,
    get_moving_loss=False,
    masked=False,
)

OptimizationParams = dict(
    iterations = 10_000,
    position_lr_init  = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 10_000,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.005,
    percent_dense = 0.01,
    lambda_dssim = 0.2,
    densification_interval = 100,
    prune_interval = 100,
    opacity_reset_interval = 3000,
    densify_from_iter = 500,
    densify_until_iter = 5_000,
    densify_grad_threshold = 0.00005,
    min_opacity = 0.005,
    batch_size=8,
    dataloader=True,
    knn_loss=False,
    scale_loss=False,
    no_deform_from_iter=0,
    # knn_selec_thresh=1.0,
)