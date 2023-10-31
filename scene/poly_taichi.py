import torch
import taichi as ti
               
class Polynomial_taichi(torch.nn.Module):
    def __init__(self, max_degree):
        super(Polynomial_taichi,self).__init__()
        self.max_degree = max_degree
        @ti.kernel
        def poly_kernel_fwd(
            factors: ti.types.ndarray(), 
            t: float, 
            out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id in ti.ndrange(
                factors.shape[0], 
                factors.shape[2]
            ):
                for f_id in ti.static(range(max_degree)):
                    if f_id < degree:
                        out[pid, dim_id] += factors[pid, f_id, dim_id] * (t ** (f_id+1))
                
        @ti.kernel
        def poly_kernel_bwd(
            d_factors: ti.types.ndarray(), 
            t: float, 
            d_out: ti.types.ndarray(), 
            degree: int
        ):
            for pid, dim_id in ti.ndrange(
                d_factors.shape[0], 
                d_factors.shape[2]
            ):
                for f_id in ti.static(range(max_degree)):
                    if f_id < degree:
                        d_factors[pid, f_id, dim_id] = d_out[pid, dim_id] * (t ** (f_id+1))
                    else:
                        d_factors[pid, f_id, dim_id] = 0.0
                    
        class _polynomial_taichi(torch.autograd.Function):
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
                poly_kernel_fwd(factors, t, out, degree)
                return out
            
            @staticmethod
            def backward(ctx, d_out):
                factors, = ctx.saved_tensors
                t = ctx.t
                degree = ctx.degree
                d_factors = torch.empty_like(factors)
                d_out = d_out.contiguous()
                poly_kernel_bwd(d_factors, t, d_out, degree)
                return d_factors, None, None
                    
        self._module_function = _polynomial_taichi.apply
        
    def forward(self, factors, timestamp, degree):
        return self._module_function(
            factors.contiguous(), 
            timestamp,
            degree,
        )