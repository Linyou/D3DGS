from .poly_taichi import Polynomial_taichi
from .fft_taichi import FFT_taichi
from .fft_poly_taichi import FFTPloy_taichi

def get_fit_model(type_name, feat_dim, poly_factor=1, Hz_factor=1):
    if type_name == "fft":
        trajectory_func = FFT_taichi(
            feat_dim, 
            Hz_base_factor=Hz_factor
        ) 
    elif type_name == "fft_poly":
        trajectory_func = FFTPloy_taichi(
            feat_dim, 
            poly_base_factor=poly_factor,
            Hz_base_factor=Hz_factor
        ) 
    elif type_name == "poly":
        trajectory_func = Polynomial_taichi(
            feat_dim,
            poly_base_factor=poly_factor,
        )
    else:
        raise ValueError(f"Unknown module name: {type_name}")
    
    return trajectory_func