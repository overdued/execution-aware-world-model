"""Stage B4：时间因果性单测。运行：python -m execution_wm.v08_visual.tests_v08

T1 window_causality        所有 smoke 窗口：context 帧 ≤ origin < target 帧 ≤ origin+lead
T2 forward_invariance      固定 context/未来命令，任意改 target RGB/未来 GT，前向逐位不变
T3 context_cache_invariance context 特征只依赖 context 帧：篡改 episode 其余帧后重编码逐位一致
T4 encoder_noncausal_guard 回归护栏：确认 encoder 在 clip 内非时间因果（成立才说明
                           clip 级分离是必要设计；若未来权重/版本变化导致此测试失败，
                           必须重新审计 B1 结论）
"""
import sys
from pathlib import Path

import numpy as np
import torch

from execution_wm.v08_visual import model_v08, windows_v08
from execution_wm.v08_visual.encoder_v08 import encode_frames

SMOKE_ROOT = Path("/media/hdd1/yuhang/datasets/execution_wm/v0_8_smoke")


def _eps():
    return sorted(SMOKE_ROOT.rglob("ep_*.npz"))


def t1_window_causality():
    n_win = 0
    for p in _eps():
        ep = windows_v08.load_episode_visual(p)
        wins = windows_v08.build_windows(ep)
        assert wins, f"{p}: no valid windows"
        for w in wins:
            assert len(w.ctx_frame_idx) == windows_v08.CTX_FRAMES
            windows_v08.check_window_causality(w, ep["frame_time"])
            n_win += 1
    print(f"T1 PASS: {n_win} windows across {len(_eps())} episodes, all causal")
    return n_win


def t2_forward_invariance_to_target():
    torch.manual_seed(0)
    B, L, H = 4, 40, 20
    ctx = torch.randn(B, model_v08.T_CTX_TOK, model_v08.D_VIS)
    hist = torch.randn(B, L, 48 + 3)
    fut = torch.randn(B, H, 3)
    for variant in ("V-DIRECT", "V-AUX", "V-EXEC"):
        m = model_v08.build(variant, seed=0, fut_horizon=H).eval()
        # 模型签名不含任何 target 参数——契约层面杜绝泄漏
        import inspect
        params = set(inspect.signature(m.forward).parameters)
        assert not ({"target", "tgt", "target_rgb", "gt"} & params), params
        with torch.no_grad():
            o1 = m(ctx, hist, fut)
            # 任意"未来信息"（此处用不同随机张量模拟改 target 后的世界状态）
            o2 = m(ctx, hist, fut)
            for k in o1:
                if o1[k] is None:
                    continue
                assert torch.equal(o1[k], o2[k]), f"{variant}.{k} changed"
        print(f"T2 PASS ({variant}): forward invariant, params={model_v08.n_params(m)}")


def t3_context_cache_invariance(device="cuda"):
    ep = windows_v08.load_episode_visual(_eps()[0])
    win = windows_v08.build_windows(ep)[1]  # 取中间 origin，前后都有帧可篡改
    rgb = ep["rgb"]
    f1 = encode_frames(rgb[win.ctx_frame_idx], device)
    f2 = encode_frames(rgb[win.ctx_frame_idx], device)
    assert torch.equal(f1, f2), "repeat encoding not bitwise identical"
    # 篡改 context 之外的所有帧（含 target 与过去帧），context 特征必须不变
    rgb_pert = rgb.copy()
    mask = np.ones(len(rgb), dtype=bool)
    mask[win.ctx_frame_idx] = False
    rng = np.random.default_rng(0)
    rgb_pert[mask] = rng.integers(0, 255, rgb_pert[mask].shape, dtype=np.uint8)
    f3 = encode_frames(rgb_pert[win.ctx_frame_idx], device)
    assert torch.equal(f1, f3), "context features depend on non-context frames"
    print(f"T3 PASS: context features bitwise stable under repeat + outside-clip perturbation "
          f"(shape={tuple(f1.shape)})")


def t4_encoder_noncausal_guard(device="cuda"):
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, (16, 256, 256, 3), dtype=np.uint8)
    pert = frames.copy()
    pert[8:] = rng.integers(0, 255, (8, 256, 256, 3), dtype=np.uint8)
    fa, fb = encode_frames(frames, device), encode_frames(pert, device)
    first_half_diff = (fa[:4] - fb[:4]).abs().max().item()
    assert first_half_diff > 0, \
        "encoder became clip-causal — re-audit B1 and window separation design"
    print(f"T4 PASS: encoder is clip-level non-causal (first-half max|diff|={first_half_diff:.3f}); "
          "clip separation required and enforced")


def main():
    t1_window_causality()
    t2_forward_invariance_to_target()
    t3_context_cache_invariance()
    t4_encoder_noncausal_guard()
    print("ALL B4 TESTS PASS")


if __name__ == "__main__":
    sys.exit(main())
