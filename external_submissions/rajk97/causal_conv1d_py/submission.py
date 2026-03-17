from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (autotuned, tensor_descriptor stripped for KernelBot)
    (1, 64, 64, 4): helion.Config(block_sizes=[32, 8], load_eviction_policies=['last', '', ''], num_stages=1, num_warps=8, pid_type='flat', range_unroll_factors=[0, 1]),
    (2, 128, 128, 4): helion.Config(block_sizes=[8, 32], load_eviction_policies=['last', '', ''], loop_orders=[[0, 2, 1]], num_stages=1, num_warps=16, pid_type='flat', range_warp_specializes=[None, True]),
    (1, 256, 256, 3): helion.Config(block_sizes=[16, 64], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None, False]),
    (1, 128, 64, 8): helion.Config(block_sizes=[8, 32], load_eviction_policies=['', '', 'last'], num_stages=1, num_warps=2, pid_type='flat'),
    (4, 64, 128, 4): helion.Config(block_sizes=[16, 8], num_stages=1, num_warps=2, pid_type='flat', static_ranges=[True]),
    # Benchmark shapes (autotuned, tensor_descriptor stripped for KernelBot)
    (1, 768, 512, 4): helion.Config(block_sizes=[16, 64], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None, True]),
    (1, 768, 2048, 4): helion.Config(block_sizes=[16, 32], num_stages=2, num_warps=8, pid_type='flat', range_flattens=[None, False]),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[1, 512], num_stages=1, num_warps=1, pid_type='flat'),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[1, 512], num_stages=1, num_warps=1, indexing='block_ptr', pid_type='flat'),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[1, 512], num_stages=1, num_warps=1, indexing=['pointer', 'pointer', 'pointer', 'block_ptr'], pid_type='flat'),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,   # (B, D, S) original input — NO padding needed
        w: torch.Tensor,    # (D, W) filter coefficients
        b: torch.Tensor,    # (D,) additive offset
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                idx = rs.index + j - (W - 1)
                safe_idx = idx.clamp(min=0)
                xv = hl.load(x, [bi, rd, safe_idx]).to(torch.float32)
                valid = (idx >= 0).to(torch.float32)
                acc = acc + xv * c[:, None] * valid[None, :]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNEL_CACHE: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    key = (B, D, S, W)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _make_kernel(SHAPE_CONFIGS[key])
    return _KERNEL_CACHE[key](x, weight, bias)
