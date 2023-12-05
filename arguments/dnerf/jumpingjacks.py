_base_ = './dnerf_default.py'

FlowParams = dict(
    xyz_traj_feat_dim = 3,
    xyz_trajectory_type = 'fft_poly',
    rot_traj_feat_dim = 2,
    rot_trajectory_type = 'fft_poly',
    # feature_traj_feat_dim = 2,
    feature_traj_feat_dim = 2,
    feature_trajectory_type = 'fft_poly',
    traj_init = 'zero',
    poly_base_factor = 1,
    Hz_base_factor = 1,
    
    # regularization
    random_noise = True,
)