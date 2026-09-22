"""V0.6.1 B5：显式独立 population（旧 held_anchor 未排除 Q4 -> 混入 96/384 窗）。

三个主 split（互不重叠的 window 集合）:
    A unseen_anchor_seen_family   = AQ6/AQ7 x Q1-Q3
    B seen_anchor_unseen_family   = train AQ(AQ0-4) x Q4
    C unseen_anchor_unseen_family = AQ6/AQ7 x Q4
另外:
    supp_AQ5xQ4 = AQ5 x Q4   （evaluation-only supplementary，显式单列，不并入主表）
    legacy_held_anchor / legacy_held_family = 旧口径（含 overlap），仅历史兼容，不当作独立确认
"""
import json
import os

TRAIN_ANCHORS = [0, 1, 2, 3, 4]
VAL_ANCHORS = [5]
TEST_ANCHORS = [6, 7]
FAMILIES = ["Q1_straight", "Q2_lateral", "Q3_turn"]
HELD_OUT_FAMILY = "Q4_combo"


def load_index(sq_dir):
    return json.load(open(os.path.join(sq_dir, "index.json")))["episodes"]


def attach_derived_paths(entries, derived_root, raw_root=None):
    """为 index.json 条目附加派生/原始路径（不修改原 dict）。"""
    import os as _os
    out = []
    for e in entries:
        e = dict(e)
        rel = _os.path.join(e["condition"], _os.path.basename(e["file"]).replace(".npz", ".20hz.npz"))
        e["path20"] = _os.path.join(derived_root, rel)
        e["path_raw"] = e["file"] if raw_root is None else _os.path.join(raw_root, rel.replace(".20hz.npz", ".npz"))
        out.append(e)
    return out


def classify(entries):
    """按 episode metadata -> split 名。返回 dict[split] = [episode]。"""
    out = {"train": [], "val": [], "A_unseen_anchor_seen_family": [],
           "B_seen_anchor_unseen_family": [], "C_unseen_anchor_unseen_family": [],
           "supp_AQ5xQ4": [], "support": []}
    for e in entries:
        if e["episode_type"] == "support":
            out["support"].append(e)
            continue
        ag = int(e["anchor_group"][2:])
        q4 = (e["command_family"] == HELD_OUT_FAMILY)
        if ag in TEST_ANCHORS:
            out["C_unseen_anchor_unseen_family" if q4 else "A_unseen_anchor_seen_family"].append(e)
        elif ag in TRAIN_ANCHORS:
            (out["B_seen_anchor_unseen_family"] if q4 else out["train"]).append(e)
        elif ag in VAL_ANCHORS:
            (out["supp_AQ5xQ4"] if q4 else out["val"]).append(e)
        else:
            raise ValueError(f"未知 anchor group: {e['anchor_group']}")
    return out


def assert_split_integrity(splits):
    """B5 断言：family whitelist / anchor whitelist / episode 去重 / 交集。"""
    rep = {}
    for name, entries in splits.items():
        if name in ("train", "val", "support"):
            continue
        anchors = sorted({e["anchor_group"] for e in entries})
        fams = sorted({e["command_family"] for e in entries})
        ids = [e["episode_id"] for e in entries]
        assert len(ids) == len(set(ids)), f"{name} 内部 episode 重复"
        rep[name] = {"n_episodes": len(ids), "anchors": anchors, "families": fams}
        if name == "A_unseen_anchor_seen_family":
            assert set(anchors) <= {"AQ6", "AQ7"}, anchors
            assert set(fams) <= set(FAMILIES), fams
            assert HELD_OUT_FAMILY not in fams
        if name == "B_seen_anchor_unseen_family":
            assert set(anchors) <= {f"AQ{a}" for a in TRAIN_ANCHORS}, anchors
            assert fams == [HELD_OUT_FAMILY], fams
        if name == "C_unseen_anchor_unseen_family":
            assert set(anchors) <= {"AQ6", "AQ7"}, anchors
            assert fams == [HELD_OUT_FAMILY], fams
        if name == "supp_AQ5xQ4":
            assert set(anchors) <= {"AQ5"}, anchors
            assert fams == [HELD_OUT_FAMILY], fams
    # 主 split 两两无交集
    main = ["A_unseen_anchor_seen_family", "B_seen_anchor_unseen_family",
            "C_unseen_anchor_unseen_family"]
    idsets = {n: {e["episode_id"] for e in splits[n]} for n in main}
    for i, a in enumerate(main):
        for b in main[i + 1:]:
            inter = idsets[a] & idsets[b]
            assert not inter, f"{a} ∩ {b} = {inter}"
            rep.setdefault("pairwise_empty", []).append(f"{a}∩{b}=∅")
    # train/val 与主 split 无交集
    tv = {e["episode_id"] for e in splits["train"]} | {e["episode_id"] for e in splits["val"]}
    for n in main + ["supp_AQ5xQ4"]:
        inter = tv & {e["episode_id"] for e in splits[n]}
        assert not inter, f"train/val 与 {n} 交集: {inter}"
    rep["train_val_vs_eval"] = "all empty"
    return rep


def legacy_splits(entries, test_anchors=TEST_ANCHORS):
    """旧口径（含 overlap），仅用于历史兼容对照，标注不能当独立确认。"""
    held_anchor = [e for e in entries if e["episode_type"] == "query"
                   and int(e["anchor_group"][2:]) in test_anchors]
    held_family_ = [e for e in entries if e["episode_type"] == "query" and e.get("held_out_family")]
    inter = {e["episode_id"] for e in held_anchor} & {e["episode_id"] for e in held_family_}
    return {"legacy_held_anchor": held_anchor, "legacy_held_family": held_family_,
            "legacy_overlap_n_episodes": len(inter)}
