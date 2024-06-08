import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_add_densification_stats_and_max_radii2D(
    N: int,

    update_filter: torch.Tensor,	# [N], bool

    radii: torch.Tensor,	# [N]
    max_radii2D: torch.Tensor,	# [N]

    viewspace_point_tensor_grad: torch.Tensor,	# [N, 2]
    xyz_gradient_accum: torch.Tensor,	# [N, 2]

    denom: torch.Tensor,	# [N]

    viewspace_point_tensor_grad_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    tid = tl.program_id(0)
    point_offsets = (tid*BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
    border_mask = point_offsets < N

    cur_update_filter = tl.load(update_filter + point_offsets, mask=border_mask, other=0) == 1

    # Update max_radii2D
    cur_max_radii2D = tl.load(max_radii2D + point_offsets, mask=border_mask)
    cur_radii = tl.load(radii + point_offsets, mask=border_mask)
    cur_max_radii2D = tl.maximum(cur_max_radii2D, cur_radii)
    tl.store(max_radii2D + point_offsets, cur_max_radii2D, mask=cur_update_filter)

    # Update xyz_gradient_accum
    grad_x = tl.load(viewspace_point_tensor_grad + point_offsets*viewspace_point_tensor_grad_stride, mask=border_mask)
    grad_y = tl.load(viewspace_point_tensor_grad + (point_offsets*viewspace_point_tensor_grad_stride+1), mask=border_mask)
    grad_norm = tl.sqrt(grad_x*grad_x + grad_y*grad_y)
    cur_xyz_gradient_accum = tl.load(xyz_gradient_accum + point_offsets, mask=border_mask)
    cur_xyz_gradient_accum += grad_norm
    tl.store(xyz_gradient_accum + point_offsets, cur_xyz_gradient_accum, mask=cur_update_filter)

    # Update denom
    cur_denom = tl.load(denom + point_offsets, mask=border_mask)
    cur_denom += 1
    tl.store(denom + point_offsets, cur_denom, mask=cur_update_filter)

def add_densification_stats_and_max_radii2D_kernel(
    update_filter: torch.Tensor,
    radii: torch.Tensor,
    max_radii2D: torch.Tensor,
    viewspace_point_tensor_grad: torch.Tensor,
    xyz_gradient_accum: torch.Tensor,
    denom: torch.Tensor
):
    BLOCK_SIZE = 512
    N = update_filter.size(0)
    viewspace_point_tensor_grad_stride = viewspace_point_tensor_grad.stride(0)

    GRID_SHAPE = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _fwd_add_densification_stats_and_max_radii2D[GRID_SHAPE](
        N,

        update_filter,

        radii,
        max_radii2D,

        viewspace_point_tensor_grad,
        xyz_gradient_accum,

        denom,

        viewspace_point_tensor_grad_stride,
        BLOCK_SIZE
    )

