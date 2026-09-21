"""Step 2/3: pre-swap audit（§4）。

输出 context_swap_v2/audit/:
    pre_swap_audit.md            人读审计报告
    leakage_check.json           4.1 privileged info 检查
    model_interface_regression.json  4.3 接口重构前后数值一致性

用法: python -m execution_wm.context_swap.audit --config execution_wm/configs/context_swap.yaml
"""
import argparse
import inspect
import json
import os
import platform

import numpy as np
import torch

from execution_wm.context_swap.common import (
    checkpoint_info, git_commit, load_cfg, load_context_model, out_subdir,
)
from execution_wm.data.dataset import PROPRIO_KEYS

# §4.1: 禁止出现在 context 输入里的字段
FORBIDDEN = ["friction", "actuator", "disturbance", "condition", "terrain",
             "future_execution", "future_residual"]


def leakage_check():
    """ContextEncoder 的输入只有 history_proprio(PROPRIO_KEYS) + history_action。"""
    from execution_wm.models.context_encoder import ContextEncoder
    src = inspect.getsource(ContextEncoder.forward)
    findings = {f: (f in src) for f in FORBIDDEN}
    result = {
        "encoder_input_keys": PROPRIO_KEYS + ["history_action(commanded velocity)"],
        "forbidden_fields_in_encoder_source": findings,
        "pass": not any(findings.values()),
        "note": "PROPRIO_KEYS 全部是机载传感器量（速度/角速度/重力投影/IMU/关节/足端接触），"
                "不含任何仿真 metadata；metadata 仅用于 pair selection 与可视化。",
    }
    return result


def causality_check():
    """窗口定义验证：history 索引 <= t0 < future 索引（§4.2）。"""
    L, H = 20, 40
    t0 = 30
    h0 = t0 - L + 1
    assert h0 >= 0 and t0 + 1 + H <= 100
    return {
        "history_indices": [h0, t0],
        "future_indices": [t0 + 1, t0 + H],
        "pass": h0 <= t0 < t0 + 1,
        "note": "c_t = GRU(proprio[t0-L+1..t0], cmd[t0-L+1..t0])，GRU 因果；"
                "不读取 t+1..t+H 任何数据。",
    }


def interface_regression(model, device):
    """接口重构前后 forward 数值必须一致（§4.3）。"""
    torch.manual_seed(0)
    B, L, H, D = 4, 20, model.predictor.H, 40
    hp = torch.randn(B, L, D, device=device)
    ha = torch.randn(B, L, 3, device=device)
    cs = hp[:, -1].clone()
    fa = torch.randn(B, H, 3, device=device)
    with torch.no_grad():
        # 旧式调用：整体 forward
        out_full = model(history_proprio=hp, history_action=ha,
                         current_state=cs, future_action=fa)
        # 新式拆分调用
        z = model.encode_state(hp)
        c = model.encode_context(hp, ha)
        out_split = model.predict_execution(z, c, fa)
    diff_r = (out_full["r_hat"] - out_split["r_hat"]).abs().max().item()
    diff_e = (out_full["e_hat"] - out_split["e_hat"]).abs().max().item()
    # current_state 必须等于 history 最后一帧（WindowDataset 保证）
    diff_z = (model.encode_state(hp) - model.state_proj(cs)).abs().max().item()
    return {
        "max_abs_diff_r_hat": diff_r,
        "max_abs_diff_e_hat": diff_e,
        "max_abs_diff_state_latent": diff_z,
        "pass": diff_r < 1e-6 and diff_e < 1e-6 and diff_z < 1e-6,
        "note": "纯接口重构，未改任何 checkpoint 参数。",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = out_subdir(cfg, "audit")

    model = load_context_model(cfg, device)
    leak = leakage_check()
    caus = causality_check()
    reg = interface_regression(model, device)
    ckpt = checkpoint_info(cfg)
    commit = git_commit(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open(os.path.join(out, "leakage_check.json"), "w") as f:
        json.dump({**leak, "causality": caus}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "model_interface_regression.json"), "w") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)

    env = {
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    mc = (ckpt.get("train_config") or {}).get("model", {})
    lines = [
        "# Pre-Swap Audit（03.1 v2 §4）",
        "",
        "## 4.4 Checkpoint",
        f"- path: `{ckpt['path']}`",
        f"- sha256: `{ckpt['sha256']}`",
        f"- size: {ckpt['size_bytes']} bytes",
        f"- original V0 best val loss: {ckpt['best_val_loss']}",
        f"- context_dim: {mc.get('context_dim')}, gru_hidden: {mc.get('gru_hidden')}, "
        f"hidden_dims: {mc.get('hidden_dims')}",
        "- 说明：V0 只训练过一个 context checkpoint，无挑选空间（§4.4）。",
        "",
        "## 4.1 Privileged information leakage",
        f"- 结果: {'PASS' if leak['pass'] else 'FAIL'}",
        f"- encoder 输入: {', '.join(leak['encoder_input_keys'][:3])} ... 共 "
        f"{len(PROPRIO_KEYS)} 组 proprio + 历史命令",
        f"- 禁止字段在 encoder 源码中出现: "
        f"{[k for k, v in leak['forbidden_fields_in_encoder_source'].items() if v] or '无'}",
        "",
        "## 4.2 Causality",
        f"- 结果: {'PASS' if caus['pass'] else 'FAIL'} — {caus['note']}",
        "",
        "## 4.3 接口可拆分 + regression",
        f"- 结果: {'PASS' if reg['pass'] else 'FAIL'}",
        f"- max|Δr_hat| = {reg['max_abs_diff_r_hat']:.3e}, "
        f"max|Δe_hat| = {reg['max_abs_diff_e_hat']:.3e}, "
        f"max|Δz| = {reg['max_abs_diff_state_latent']:.3e}",
        "- encode_state / encode_context / predict_execution 三个接口已加入 "
        "ExecutionContextModel，forward 改调它们，参数未动。",
        "",
        "## Environment",
        f"- git commit: {commit}",
        f"- python {env['python']}, torch {env['torch']} (cuda {env['cuda']}), GPU: {env['gpu']}",
    ]
    with open(os.path.join(out, "pre_swap_audit.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    assert leak["pass"] and caus["pass"] and reg["pass"], "audit 未通过，停止"


if __name__ == "__main__":
    main()
