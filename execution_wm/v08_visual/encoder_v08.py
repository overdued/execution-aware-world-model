"""V-JEPA2 冻结 encoder 封装（B1 审计口径）。

- 权重：facebook/vjepa2-vitl-fpc64-256（snapshot b3c1679，sha256 见 audit/encoder_and_checkpoint.json）；
- 只前向 encoder 路径，全部参数 requires_grad_(False) + eval()；
- 因果性：encoder 在 clip 内双向 attention（非时间因果），调用方必须做 clip 级
  分离——context 与 target 各自独立调用 encode_frames，不得混合。
"""
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
SNAPSHOT = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"

_CACHE = {}


def load_encoder(device="cuda"):
    if "model" not in _CACHE:
        model = AutoModel.from_pretrained(MODEL_ID, revision=SNAPSHOT,
                                          dtype=torch.float32)
        model = model.to(device).eval()
        model.requires_grad_(False)
        n_frozen = sum(p.numel() for p in model.parameters())
        assert not any(p.requires_grad for p in model.parameters())
        proc = AutoVideoProcessor.from_pretrained(MODEL_ID, revision=SNAPSHOT)
        _CACHE.update(model=model, proc=proc, device=device, n_frozen=n_frozen)
    return _CACHE["model"], _CACHE["proc"]


@torch.no_grad()
def encode_frames(frames: np.ndarray, device="cuda") -> torch.Tensor:
    """frames uint8 [T,H,W,3]（T 偶数）-> tokens float32 [T//2, 1024]（CPU，已 detach）。

    空间 256 token 做 mean-pool，只保留时域 token 级表示（pilot 主终点口径）。
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3
    T = frames.shape[0]
    assert T % 2 == 0, "tubelet_size=2 要求偶数帧"
    model, proc = load_encoder(device)
    inputs = proc(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs).last_hidden_state           # [1, (T/2)*256, 1024]
    n_tok = T // 2
    tok = out.reshape(1, n_tok, 256, -1).mean(dim=2)  # 空间 pool -> [1, T/2, 1024]
    return tok[0].float().cpu()
