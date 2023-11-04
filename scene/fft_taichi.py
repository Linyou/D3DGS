import torch
import taichi as ti
from taichi.math import vec2, pi

Hz_base = pi
        
        
class FFT_taichi(torch.nn.Module):
    def __init__(self, max_degree):
        super(FFT_taichi,self).__init__()
        self.max_degree = max_degree
        @ti.kernel
        def fft_kernel_fwd(factors: ti.types.ndarray(dtype=vec2, ndim=3), t: float, out: ti.types.ndarray(), degree: int):
            for pid, dim_id in ti.ndrange(
                factors.shape[0], 
                factors.shape[2]
            ):
                for f_id in ti.static(range(max_degree)):
                    if f_id < degree:
                        fac_vec = factors[pid, f_id, dim_id]
                        # noise_vec = noise[pid, dim_id, f_id]
                        current_w = (f_id+1) * 2 * Hz_base * t
                        out[pid, dim_id] += (fac_vec[0] * (ti.sin(current_w)) + fac_vec[1] * (ti.cos(current_w)))
                        
                
        @ti.kernel
        def fft_kernel_bwd(d_factors: ti.types.ndarray(), t: float, d_out: ti.types.ndarray(), degree: int):
            for pid, dim_id in ti.ndrange(
                d_factors.shape[0], 
                d_factors.shape[2]
            ):
                # if f_id < degree:
                for f_id in ti.static(range(max_degree)):
                    if f_id < degree:
                        current_w = (f_id+1) * 2 * Hz_base * t
                        gradient = d_out[pid, dim_id]
                        # noise_vec = noise[pid, dim_id, f_id]
                        d_factors[pid, f_id, dim_id, 0] = gradient * (ti.sin(current_w))
                        d_factors[pid, f_id, dim_id, 1] = gradient * (ti.cos(current_w))
                    else:
                        d_factors[pid, f_id, dim_id, 0] = 0.0
                        d_factors[pid, f_id, dim_id, 1] = 0.0
                    
        class _fft_taichi(torch.autograd.Function):
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
                fft_kernel_fwd(
                    factors, 
                    t, 
                    out, 
                    # noise.contiguous(), 
                    degree
                )
                return out
            
            @staticmethod
            def backward(ctx, d_out):
                factors, = ctx.saved_tensors
                t = ctx.t
                degree = ctx.degree
                d_factors = torch.empty_like(factors)
                # make sure contiguous 
                d_out = d_out.contiguous()
                fft_kernel_bwd(
                    d_factors, 
                    t, 
                    d_out,
                    # noise, 
                    degree
                )
                return d_factors, None, None
                
        self._module_function = _fft_taichi.apply
        
    def forward(self, factors, timestamp, degree):
        return self._module_function(
            factors.contiguous(), 
            timestamp, 
            # noise,
            degree,
        )