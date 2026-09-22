"""V0.6 阶段A1：可独立复核的 node-level manifest。

episode_uid 含数据版本（d0=v0 主数据集, d1=v0_controlled_probes），防止跨版本碰撞。
reset_anchor_group 重构依据（代码审计，详见 audit/causal_time_frame_audit.md / reset_and_injection_audit.md）:
  - d0: 同一 condition 轮次内 env 不 reset（schedule_end 后直接接下一 episode）；
        仅 termination_reason=='terminated'（摔倒 done）后 env 自动 reset。
        anchor = d0:{condition}:env{e}:chain{k}，k 在每次 terminated 后 +1。
  - d1: 每条 probe episode 前 env.reset()（collect_controlled_probes.py），每条独立 anchor。
seed: d0 未记录 -> null；d1 有 seed 字段。

输出:
  manifests/episodes.csv
  manifests/full_pair_manifest.csv     （accepted + rejected，含 A1 全部列）
  manifests/duplicate_and_reuse.csv
  manifests/split_manifest.csv
  manifests/support_query_manifest.csv （schema-only：阶段A 无 support/query 数据）

用法: python -m execution_wm.validity_v06.manifest_build --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

from execution_wm.data.dataset import discover_episodes, split_episodes


def sha1_arr(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def build_episodes(cfg):
    """两个数据集的 episode 级总表。"""
    rows = []
    for vi, key in ((0, "dataset_v0"), (1, "dataset_probes")):
        root = cfg[key]
        idx_path = os.path.join(root, "index.json")
        if os.path.exists(idx_path):
            eps = json.load(open(idx_path))["episodes"]
        else:  # d1 无 index.json，从 per-episode json 发现
            eps = []
            for e in discover_episodes(root):
                m = e["meta"]
                eps.append({"file": e["path"], "meta_file": e["meta_path"],
                            "steps_20hz": None, **m})
        for e in eps:
            uid = f"d{vi}_ep{int(e['episode_id']):05d}"
            rows.append({
                "episode_uid": uid,
                "dataset_version": f"d{vi}",
                "origin_index": int(e["episode_id"]),
                "file": e.get("file"),
                "condition": e["condition"],
                "episode_type": e["episode_type"],
                "probe_name": e.get("probe_name"),
                "friction": e.get("friction"),
                "actuator_scale": e.get("actuator_scale"),
                "disturbance": e.get("disturbance"),
                "termination_reason": e.get("termination_reason"),
                "duration_s": e.get("duration_s"),
                "env_id": e.get("env_id"),
                "seed": e.get("seed"),          # d0 -> None（缺失，明确 null）
                "reset_state": e.get("reset_state"),
                "controlled": bool(e.get("controlled", False)),
            })
    df = pd.DataFrame(rows)

    # ---- reset_anchor_group ----
    anchor = {}
    for (ver, cond, env), g in df[df.dataset_version == "d0"].groupby(
            ["dataset_version", "condition", "env_id"]):
        chain = 0
        for _, r in g.sort_values("origin_index").iterrows():
            anchor[r.episode_uid] = f"{ver}:{cond}:env{int(env)}:chain{chain}"
            if r.termination_reason == "terminated":
                chain += 1
    for _, r in df[df.dataset_version == "d1"].iterrows():
        anchor[r.episode_uid] = f"d1:{r.episode_uid}"
    df["reset_anchor_group"] = df.episode_uid.map(anchor)

    # ---- train/val/test role（与训练同协议同 seed）----
    tc = yaml.safe_load(open(cfg["train_config"]))["dataset"]
    d0_eps = discover_episodes(cfg["dataset_v0"])
    splits = split_episodes(d0_eps, tc["val_fraction"], tc["test_id_fraction"],
                            tc["ood_conditions"], tc["exclude_probe_from_train"], 42)
    role = {}
    for name, lst in splits.items():
        for e in lst:
            m = e["meta"]
            role[f"d0_ep{int(m['episode_id']):05d}"] = name
    df["train_val_test_role"] = [
        role.get(u, "probe_eval" if u.startswith("d1") else "unknown") for u in df.episode_uid]
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg["out_dir"], "manifests")
    os.makedirs(out, exist_ok=True)
    hz = cfg["hz"]
    L = int(round(cfg["history_s"] * hz))
    H = int(round(cfg["horizon_s"] * hz))

    ep = build_episodes(cfg)
    ep.to_csv(os.path.join(out, "episodes.csv"), index=False)
    epm = ep.set_index("episode_uid")

    # ---- anchor 跨 split 泄漏检查（同链 episode 落到不同 split）----
    d0 = ep[ep.dataset_version == "d0"]
    chain_roles = d0.groupby("reset_anchor_group")["train_val_test_role"].nunique()
    n_mixed = int((chain_roles > 1).sum())

    # ---- full pair manifest（accepted + rejected）----
    acc = pd.read_csv(cfg["pair_manifest"])
    rej = pd.read_csv(cfg["pair_manifest"].replace("pair_manifest.csv", "rejected_pairs.csv"))
    acc["accepted"] = True
    rej["accepted"] = False
    man = pd.concat([acc, rej], ignore_index=True)
    man.insert(0, "pair_id", [f"p{i:06d}" for i in range(len(man))])

    def side_cols(prefix):
        uid = man[f"ep_{prefix}"]
        return pd.DataFrame({
            f"{prefix}_episode_uid": uid,
            f"{prefix}_origin_index": uid.map(epm.origin_index),
            f"{prefix}_origin_time": man.t0 / hz,
            f"{prefix}_seed": uid.map(epm.seed),
            f"{prefix}_dataset_version": uid.map(epm.dataset_version),
            f"{prefix}_condition": uid.map(epm.condition),
            f"{prefix}_anchor_group": uid.map(epm.reset_anchor_group),
            f"{prefix}_split_role": uid.map(epm.train_val_test_role),
        })
    # 显式命名（A1 要求的列名）
    src = side_cols("A")
    src.columns = [c.replace("A_", "source_").replace("A", "source") for c in src.columns]
    tgt = side_cols("B")
    tgt.columns = [c.replace("B_", "target_").replace("B", "target") for c in tgt.columns]
    full = pd.concat([man, src, tgt], axis=1)
    full["probe_or_command_family"] = man.probe
    full["shared_prefix_group"] = np.where(man.probe.notna(), man.probe, None)
    full["support_query_group"] = None
    full["history_start"] = (man.t0 - L + 1) / hz
    full["history_end"] = man.t0 / hz
    full["prediction_origin"] = man.t0 / hz
    full["future_start"] = (man.t0 + 1) / hz
    full["future_end"] = (man.t0 + H) / hz
    full["sample_dt"] = 1.0 / hz
    full["reject_reason"] = np.where(
        full.accepted, "",
        np.where(man.cmd_mismatch > 0, "cmd_mismatch>0", "D_state>threshold"))

    # hash 列 + reuse（只对有缓存的 accepted 行计算 hash）
    cache = dict(np.load(cfg["pred_cache"]))
    n_acc = len(acc)
    full["command_hash"] = ""
    full["source_state_hash"] = ""
    full["target_state_hash"] = ""
    full.loc[:n_acc - 1, "command_hash"] = [sha1_arr(a) for a in cache["u"]]
    full.loc[:n_acc - 1, "source_state_hash"] = [sha1_arr(a) for a in cache["e_a"]]
    full.loc[:n_acc - 1, "target_state_hash"] = [sha1_arr(a) for a in cache["e_b"]]
    src_cnt = full[full.accepted].source_episode_uid.value_counts()
    tgt_cnt = full[full.accepted].target_episode_uid.value_counts()
    full["source_reuse_count"] = full.source_episode_uid.map(src_cnt).fillna(0).astype(int)
    full["target_reuse_count"] = full.target_episode_uid.map(tgt_cnt).fillna(0).astype(int)
    full.to_csv(os.path.join(out, "full_pair_manifest.csv"), index=False)

    # ---- duplicate_and_reuse ----
    fut_hash = pd.Series([sha1_arr(a) for a in cache["e_a"]])
    fut_b = pd.Series([sha1_arr(a) for a in cache["e_b"]])
    dup = pd.DataFrame({"episode_uid": ep.episode_uid})
    dup["reuse_as_source"] = dup.episode_uid.map(src_cnt).fillna(0).astype(int)
    dup["reuse_as_target"] = dup.episode_uid.map(tgt_cnt).fillna(0).astype(int)
    accm = full[full.accepted]
    uniq_src = accm.assign(h=fut_hash.values).groupby("source_episode_uid").h.nunique()
    uniq_tgt = accm.assign(h=fut_b.values).groupby("target_episode_uid").h.nunique()
    dup["unique_source_windows"] = dup.episode_uid.map(uniq_src).fillna(0).astype(int)
    dup["unique_target_windows"] = dup.episode_uid.map(uniq_tgt).fillna(0).astype(int)
    dup["max_single_episode_pair_share"] = (
        pd.concat([src_cnt, tgt_cnt]).groupby(level=0).sum() / len(accm)).reindex(
        dup.episode_uid).fillna(0.0).round(4).values
    dup.to_csv(os.path.join(out, "duplicate_and_reuse.csv"), index=False)

    # ---- split manifest（episode + anchor + role）----
    split = ep[["episode_uid", "dataset_version", "condition", "episode_type",
                "reset_anchor_group", "train_val_test_role", "seed"]].copy()
    split["split_note"] = np.where(
        split.dataset_version == "d0",
        "episode-level split（非 anchor-level；同链 episode 可跨 split——见审计报告）",
        "d1 全部 probe_eval，不进训练")
    split.to_csv(os.path.join(out, "split_manifest.csv"), index=False)

    # ---- support/query（schema-only）----
    sq = pd.DataFrame(columns=[
        "sq_id", "context_session_id", "support_episode_uid", "query_anchor_id",
        "query_episode_uid", "query_command_family", "condition_support",
        "condition_query", "swap_type", "anchor_restoration", "notes"])
    sq.to_csv(os.path.join(out, "support_query_manifest.csv"), index=False)

    print(f"[manifest] episodes={len(ep)} (d0={len(d0)}, d1={len(ep)-len(d0)})")
    print(f"[manifest] anchors: d0={d0.reset_anchor_group.nunique()} "
          f"d1={ep[ep.dataset_version=='d1'].reset_anchor_group.nunique()}")
    print(f"[manifest] pairs: accepted={int(full.accepted.sum())} rejected={int((~full.accepted).sum())}")
    print(f"[manifest] unique source episodes={accm.source_episode_uid.nunique()} "
          f"target={accm.target_episode_uid.nunique()} "
          f"anchor_groups_src={accm.source_anchor_group.nunique()} "
          f"tgt={accm.target_anchor_group.nunique()}")
    print(f"[manifest] 同 anchor 跨 split 的链数（d0, 排除 probe_eval）: "
          f"{n_mixed} / {len(chain_roles)}")
    print(f"[manifest] max single-episode pair share: {dup.max_single_episode_pair_share.max():.3f}")
    print(f"[manifest] -> {out}")


if __name__ == "__main__":
    main()
