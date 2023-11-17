import torch
import taichi as ti
from taichi.math import vec2, pi, vec3
               
class FFTPloy_taichi(torch.nn.Module):
    def __init__(self, max_degree, poly_base_factor=1., Hz_base_factor=1.):
        super(FFTPloy_taichi,self).__init__()
        self.max_degree = max_degree
        Hz_base = Hz_base_factor * pi
        @ti.kernel
        def _fft_poly_kernel_fwd(
            # factors: ti.types.ndarray(dtype=vec3, ndim=3), 
            factors: ti.types.ndarray(dtype=vec3, ndim=3),
            # t: ti.types.ndarray(), 
            t: float,
            out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id, f_id in ti.ndrange(
                factors.shape[0], 
                factors.shape[2],
                max_degree
            ):
                
                # for f_id in ti.static(range(max_degree)):
                    if f_id < degree:
                        x = poly_base_factor * t
                        f = factors[pid, f_id, dim_id]
                        poly = f[0] * ti.pow(x, f_id)
                        x = (f_id) * 2 * Hz_base * t
                        sin = f[1] * ti.sin(x)
                        cos = f[2] * ti.cos(x)
                        out[pid, dim_id] += poly + sin + cos
                        
        self._fft_poly_kernel_fwd = _fft_poly_kernel_fwd
                
        @ti.kernel
        def _fft_poly_kernel_bwd(
            d_factors: ti.types.ndarray(),
            factors: ti.types.ndarray(dtype=vec3, ndim=3),
            # t: ti.types.ndarray(), 
            t: float,
            d_out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id, f_id in ti.ndrange(
                d_factors.shape[0], 
                d_factors.shape[2],
                max_degree
            ):
                # for f_id in ti.static(range(max_degree)):
                if f_id < degree:
                    # f = factors[pid, f_id, dim_id]
                    x = poly_base_factor * t #* f[1] + f[2]
                    d_o = d_out[pid, dim_id]
                    d_factors[pid, f_id, dim_id, 0] = d_o * ti.pow(x, f_id)
                    x = (f_id) * 2 * Hz_base * t
                    d_factors[pid, f_id, dim_id, 1] = d_o*ti.sin(x)
                    d_factors[pid, f_id, dim_id, 2] = d_o*ti.cos(x)
                else:
                    d_factors[pid, f_id, dim_id, 0] = 0.0
                    d_factors[pid, f_id, dim_id, 1] = 0.0
                    d_factors[pid, f_id, dim_id, 2] = 0.0
        
        self._fft_poly_kernel_bwd = _fft_poly_kernel_bwd
        class _fft_ploy_taichi(torch.autograd.Function):
            @staticmethod
            def forward(ctx, factors, t, degree=1):
                ctx.save_for_backward(factors)
                ctx.t = t
                ctx.degree = degree
                out = torch.zeros(
                    (factors.shape[0], factors.shape[2]), 
                    dtype=torch.float32, 
                    device=factors.device
                )
                self._fft_poly_kernel_fwd(factors, t, out, degree)
                return out
            
            @staticmethod
            def backward(ctx, d_out):
                factors, = ctx.saved_tensors
                t = ctx.t
                degree = ctx.degree
                d_factors = torch.empty_like(factors)
                d_out = d_out.contiguous()
                self._fft_poly_kernel_bwd(d_factors, factors, t, d_out, degree)
                return d_factors, None, None
                    
        self._module_function = _fft_ploy_taichi.apply
        
    def forward(self, factors, timestamp, degree):
        return self._module_function(
            factors.contiguous(), 
            timestamp,
            degree,
        )