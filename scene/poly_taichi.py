import torch
import taichi as ti
ti.init(arch=ti.cuda)

@ti.kernel
def poly_kernel_fwd(factors: ti.types.ndarray(), t: int, out: ti.types.ndarray()):
    for pid, dim_id, f_id in ti.ndrange(factors.shape[0], factors.shape[1], factors.shape[2]):
        out[pid, dim_id] += factors[pid, dim_id, f_id] * (t ** (f_id+1))
        
@ti.kernel
def poly_kernel_bwd(d_factors: ti.types.ndarray(), t: int, d_out: ti.types.ndarray()):
    for pid, dim_id, f_id in ti.ndrange(d_factors.shape[0], d_factors.shape[1], d_factors.shape[2]):
        d_factors[pid, dim_id, f_id] = d_out[pid, dim_id] * (t ** (f_id+1))
        
class _polynomial_taichi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factors, t):
        ctx.save_for_backward(factors)
        ctx.t = t
        out = torch.empty(
            (factors.shape[0], factors.shape[2]), 
            dtype=torch.float32, 
            device=factors.device
        )
        poly_kernel_fwd(factors, t, out)
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        factors, = ctx.saved_tensors
        t = ctx.t
        d_factors = torch.empty_like(factors)
        poly_kernel_bwd(d_factors, t, d_out)
        return d_factors, None
        
        
class Polynomial_taichi(torch.nn.Module):
    def __init__(self, num_points, dim, max_degree):
        super(Polynomial_taichi,self).__init__()
        self.dim = dim
        self.max_degree = max_degree
        self.factors = torch.nn.Parameter(
            torch.randn(
                (num_points, dim, max_degree),
                dtype=torch.float32,
            ).requires_grad_(True)
        )
        
    def forward(self, x):
        return _polynomial_taichi.apply(self.factors, x)