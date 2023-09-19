import torch
import taichi as ti
from taichi.math import vec2
import numpy as np

pi = np.pi
half_pi = np.pi / 2
quit_pi = np.pi / 4
double_pi = np.pi * 2

@ti.kernel
def fft_kernel_fwd(factors: ti.types.ndarray(dtype=vec2, ndim=3), t: float, out: ti.types.ndarray()):
    for pid, dim_id, f_id in ti.ndrange(factors.shape[0], factors.shape[1], factors.shape[2]):
        fac_vec = factors[pid, dim_id, f_id]
        current_w = f_id * quit_pi * t
        out[pid, dim_id] += (fac_vec[0] * ti.sin(current_w) + fac_vec[1] * ti.cos(current_w))
        
@ti.kernel
def fft_kernel_bwd(d_factors: ti.types.ndarray(), t: float, d_out: ti.types.ndarray()):
    for pid, dim_id, f_id in ti.ndrange(d_factors.shape[0], d_factors.shape[1], d_factors.shape[2]):
        current_w = f_id * quit_pi * t
        gradient = d_out[pid, dim_id]
        d_factors[pid, dim_id, f_id, 0] = gradient * ti.sin(current_w)
        d_factors[pid, dim_id, f_id, 1] = gradient * ti.cos(current_w)
        
        
class _fft_taichi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factors, t):
        ctx.save_for_backward(factors)
        ctx.t = t
        out = torch.empty(
            (factors.shape[0], factors.shape[2]), 
            dtype=torch.float32, 
            device=factors.device
        )
        fft_kernel_fwd(factors, t, out)
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        factors, = ctx.saved_tensors
        t = ctx.t
        d_factors = torch.empty_like(factors)
        fft_kernel_bwd(d_factors, t, d_out)
        return d_factors, None
        
        
class FFT_taichi(torch.nn.Module):
    def __init__(self, num_points, dim, max_degree):
        super(FFT_taichi,self).__init__()
        self.dim = dim
        self.max_degree = max_degree
        self.factors = torch.nn.Parameter(
            torch.randn(
                (num_points, dim, max_degree, 2),
                dtype=torch.float32,
            ).requires_grad_(True)
        )
        
    def forward(self, x):
        return _fft_taichi.apply(
            self.factors.contiguous(), 
            x.contiguous(), 
        )