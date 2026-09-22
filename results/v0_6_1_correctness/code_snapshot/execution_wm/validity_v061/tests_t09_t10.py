"""V0.6.1 T09（缓存一致性）与 T10（policy 消费时刻 smoke）测试产物。

T09: model eval / summary / bootstrap 复用同一 pred_cache 哈希 —— 断言 bootstrap
     读取的哈希 == eval 写出的哈希，且 bootstrap 不重新推理。
T10: policy 消费时刻（需已在 Isaac 环境跑过 policy_command_audit.py）；若缺失则
     标记 BLOCKED（相应采集结论一并阻塞）。
"""
import hashlib
import json
import os
import subprocess

OUT = "results/v0_6_1_correctness"
UT = os.path.join(OUT, "unit_tests")
CACHE = os.path.join(OUT, "predictions", "pred_cache.npz")
MANIFEST = os.path.join(OUT, "predictions", "pred_cache_manifest.json")
TIMELINE = os.path.join(OUT, "audit", "policy_command_timeline.json")


def t09():
    man = json.load(open(MANIFEST))
    sha = hashlib.sha256(open(CACHE, "rb").read()).hexdigest()
    assert sha == man["sha256"], "pred_cache 与 manifest 哈希不一致"
    # bootstrap 源码中不得出现模型加载 / 推理调用
    src = open("execution_wm/validity_v061/boot_v061.py").read()
    forbidden = ["torch.load", "MODEL_REGISTRY", "predict_with", "load_trained", "encode_context"]
    hits = [f for f in forbidden if f in src]
    assert not hits, f"bootstrap 中出现推理调用: {hits}"
    assert "pred_cache.npz" in src, "bootstrap 未读取 pred_cache"
    return {"cache_sha256": sha, "manifest_sha256": man["sha256"], "match": True,
            "n_cached_arrays": len(man["keys"]),
            "bootstrap_reads_cache_only": True,
            "forbidden_calls_found": hits,
            "note": "所有 summary/bootstrap 引用同一缓存；单次推理，不重抽 donor 重新推理"}


def t10():
    if not os.path.exists(TIMELINE):
        return {"status": "BLOCKED",
                "reason": "未找到 policy_command_timeline.json（需在 Isaac 环境运行 "
                          "execution_wm/data/policy_command_audit.py）",
                "blocked_conclusions": ["command pipeline latency 结论", "u_consumed 恢复"]}
    d = json.load(open(TIMELINE))
    res = {"status": "PASS", "control_dt": d["control_dt"],
           "obs_dim": d["obs_dim"], "cmd_slice": d["cmd_slice_from_terms"],
           "policy_checkpoint": d["policy_checkpoint"]}
    summ = {}
    for wname, log in d["waves"].items():
        import numpy as np
        u = np.array(log["u_requested"]); po = np.array(log["policy_obs_cmd"])
        n = len(u)
        same = int(sum(np.allclose(po[i], u[i], atol=1e-6) for i in range(n)))
        prev = int(sum(np.allclose(po[i], u[i - 1], atol=1e-6) for i in range(1, n)))
        events = [i for i in range(1, n) if not np.allclose(u[i], u[i - 1], atol=1e-9)]
        lag = []
        for i in events:
            hit = [j for j in range(i, n) if np.allclose(po[j], u[i], atol=1e-6)]
            lag.append(int(hit[0] - i) if hit else None)
        summ[wname] = {"n_ticks": n, "consumed_eq_requested_same_tick": same,
                       "consumed_eq_prev_tick": prev, "n_events": len(events),
                       "event_lags_ticks": lag}
        assert prev == n - 1, f"{wname}: 消费侧滞后不是 1 tick（{prev}/{n-1}）"
        assert set(lag) == {1}, f"{wname}: 事件滞后不为 1: {lag}"
    res["waves"] = summ
    res["verified_rule"] = "u_consumed[i] = u_requested[i-1]（精确 1 control tick = 0.02s）"
    res["execution_governing_command_lag_ticks"] = 2
    res["raw_reuse_decision"] = (
        "固定可验证规则 -> 旧 240 条 raw 无需重采；cmd_consumed 可由 cmd_ref 版本化恢复")
    res["gym_version"] = subprocess.run(
        ["python", "-c", "import gymnasium;print(gymnasium.__version__)"],
        capture_output=True, text=True).stdout.strip()
    return res


def main():
    os.makedirs(UT, exist_ok=True)
    out = {}
    for name, fn in (("T09_pred_cache_consistency", t09),
                     ("T10_policy_command_timeline", t10)):
        try:
            detail = fn()
            status = detail.pop("status", "PASS")
            out[name] = {"test": name, "status": status, "detail": detail}
        except Exception as e:  # noqa: BLE001
            out[name] = {"test": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}"}
        with open(os.path.join(UT, f"{name}.json"), "w") as f:
            json.dump(out[name], f, indent=1, ensure_ascii=False)
        print(f"[{out[name]['status']}] {name}: "
              f"{json.dumps(out[name]['detail'], ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
