# import os
# os.environ["ENABLE_TILE"] = "1"
# os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import base64
import hashlib
from pathlib import Path
import tempfile

import torch
import helion
import helion.language as hl


_EMBEDDED_ACF_B64 = (
    "dxWiJZMOCeaAKfxShG5sR8Slu8WAqGs0RB67rBVVo+kdOdke6UMONB3JWMOw9UvEqsYpy4W+94MEmPFzk2iwLpALAE9cWgenpNDQdGoimXe2F6hQvH+Pf3CBBhVp8EzLrabqO0z4muux4QWpa49J3gDT3Vqx5NlwE1m1suPewqz+evY9acuh9pyt+jvYGNRrDA4t9IWoyvL/ZHzpL+nguKFtVFlhK5fLSUcxPJ4x97VlG21QYPdmXch0KUE+U82Rc3/z1QFgASQvZ9oI2xoNqEIVuP7pgMmW+xmXlwTPZD6A1I8Ijwgip8Ws1MAV5Q=="
)
_EMBEDDED_ACF_SHA256 = "7526e4568a661f1035efa937525117cff154119c302f58aba1712ad851efbb46"


def _materialize_advanced_controls_file() -> str:
    payload = base64.b64decode(_EMBEDDED_ACF_B64, validate=True)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != _EMBEDDED_ACF_SHA256:
        raise RuntimeError("Embedded advanced controls file failed checksum validation")

    candidate_dirs = [
        Path(__file__).resolve().parent / ".embedded_assets",
        Path(tempfile.gettempdir()) / "helion_embedded_assets" / "causal_conv1d_py",
    ]
    last_error = None

    for base_dir in candidate_dirs:
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            acf_path = base_dir / "causal_conv_2.acf"
            if acf_path.exists():
                existing_sha256 = hashlib.sha256(acf_path.read_bytes()).hexdigest()
                if existing_sha256 == _EMBEDDED_ACF_SHA256:
                    return str(acf_path)

            acf_path.write_bytes(payload)
            return str(acf_path)
        except OSError as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to materialize embedded advanced controls file: {last_error}"
    )


_ADVANCED_CONTROLS_FILE = "/opt/booster_pack/causal_conv_2.acf" # _materialize_advanced_controls_file()


# Per-shape configs: map (B, D, S, W) to helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(block_sizes=[64], num_warps=2, num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    (1, 256, 256, 3): helion.Config(block_sizes=[256], num_warps=4, num_stages=2),
    (1, 128, 64, 8): helion.Config(block_sizes=[64], num_warps=2, num_stages=2),
    (4, 64, 128, 4): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 1536, 2048, 4): helion.Config(
        block_sizes=[256],
        num_warps=2,
        num_stages=2,
        advanced_controls_file=_ADVANCED_CONTROLS_FILE,
    ),
    (1, 2560, 2048, 4): helion.Config(
        block_sizes=[512],
        num_warps=4,
        num_stages=2,
        advanced_controls_file=_ADVANCED_CONTROLS_FILE,
    ),
    (1, 2560, 4096, 4): helion.Config(
        block_sizes=[512],
        num_warps=4,
        num_stages=2,
        advanced_controls_file=_ADVANCED_CONTROLS_FILE,
    ),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,  # (B, D, S) input
        w: torch.Tensor,  # (D, W) filter coefficients
        b: torch.Tensor,  # (D,) additive offset
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty_like(x)

        block_s = hl.register_block_size(32, S)
        for rbd, rs in hl.tile([B * D, S], block_size=[1, block_s]):
            flat_idx = rbd.begin
            b_idx = flat_idx // D
            d_idx = flat_idx % D

            acc = hl.zeros([rs], dtype=torch.float32)
            base_idx = rs.index - (W - 1)
            for j in range(W):
                x_idx = base_idx + j
                x_vals = hl.load(x, [b_idx, d_idx, x_idx], extra_mask=x_idx >= 0).to(
                    torch.float32
                )
                acc = acc + x_vals * w[d_idx, j].to(torch.float32)

            y[b_idx, d_idx, rs] = (acc + b[d_idx].to(torch.float32)).to(y.dtype)

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    return kernel(x, weight, bias)
