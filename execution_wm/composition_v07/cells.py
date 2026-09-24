"""V0.7 命令单元（cell）定义与 cell 级 split —— 采集/训练/评价共用同一份真源。

值集（每轴独立，来自 Stage 0 实测 controller 范围 ±1.0）:
    A = 0.35 * limit, B = 0.65 * limit        ->  vx: ±0.35/±0.65 m/s
    VALUES = [-B, -A, 0, +A, +B]              ->  索引 0..4（索引 2 == 0）
cell_id = c{ix}{iy}{iw}，例 c202 = (vx=+A, vy=0, wz=-B)

cell 类型: zero / single / double(二元) / triple(三元)
regime:  R0 = zero + 全部 single；R1 = R0 + 每类 double 的 10 个 train cells。

泛化层级（任务书 §4.2，互不混同）:
    P0 seen-cell control      训练见过的 cell，新 anchor / 新时序模板
    P1 held-out pair cells    二元轴类型在训练出现，具体符号/幅值元组从未出现；
                              每个单轴取值都在训练出现过
    P2 triple composition     三轴同时作用未训练；其三个二元投影都在 R1 train 出现
"""
import itertools
import json
import os

LIMIT = 1.0                     # Stage 0 实测 controller 范围（三轴均为 ±1.0）
A_MAG, B_MAG = 0.35, 0.65
VALUES = [-B_MAG, -A_MAG, 0.0, A_MAG, B_MAG]      # 索引 0..4，2 == 0
NZ_IDX = (0, 1, 3, 4)                             # 非零值索引
ZERO_IDX = 2
AXES = ("vx", "vy", "wz")

# 时序模板的第 i 个轴命名的编号（cell id 中第 i 位对应 AXES[i]）
def cell_id(ix, iy, iw):
    return f"c{ix}{iy}{iw}"


def parse(cid):
    return int(cid[1]), int(cid[2]), int(cid[3])


def cell_values(cid):
    ix, iy, iw = parse(cid)
    return [VALUES[ix], VALUES[iy], VALUES[iw]]


def cell_type(cid):
    nz = sum(1 for i in parse(cid) if i != ZERO_IDX)
    return {0: "zero", 1: "single", 2: "double", 3: "triple"}[nz]


def cell_axes(cid):
    """返回非零轴名的元组（有序）。"""
    return tuple(AXES[i] for i, v in enumerate(parse(cid)) if v != ZERO_IDX)


def all_cells():
    return [cell_id(*t) for t in itertools.product(range(5), repeat=3)]


PAIR_NAME = {("vx", "vy"): "XY", ("vx", "wz"): "XW", ("vy", "wz"): "YW"}


def split_doubles(min_balance=True):
    """每类 double 的 16 个非零 cell -> 10 train / 3 val / 3 test（穷举最小不平衡）。

    pair 类型 = 两个非零轴名的组合（XY = vx&vy / XW = vx&wz / YW = vy&wz）。
    目标：每轴每个值在 val(3 cell) 与 test(3 cell) 中尽量均匀出现。
    """
    out = {}
    for pair in (("vx", "vy"), ("vx", "wz"), ("vy", "wz")):     # XY / XW / YW
        a, b = pair
        ai, bi = AXES.index(a), AXES.index(b)
        cells = []
        for i in NZ_IDX:
            for j in NZ_IDX:
                idx = [ZERO_IDX, ZERO_IDX, ZERO_IDX]
                idx[ai], idx[bi] = i, j
                cells.append(cell_id(*idx))
        assert len(cells) == 16, len(cells)
        best, best_score = None, None
        for val in itertools.combinations(range(16), 3):
            rest = [k for k in range(16) if k not in val]
            for test in itertools.combinations(rest, 3):
                score = 0.0
                for grp in (val, test):
                    for ax in (ai, bi):
                        cnt = {}
                        for k in grp:
                            v = parse(cells[k])[ax]
                            cnt[v] = cnt.get(v, 0) + 1
                        for v in NZ_IDX:
                            score += abs(cnt.get(v, 0) - 3 / 4)
                if best_score is None or score < best_score:
                    best_score, best = score, (val, test)
        val, test = best
        train = [k for k in range(16) if k not in val and k not in test]
        assert len(train) == 10
        out[PAIR_NAME[pair]] = {
            "pair": pair, "train": [cells[k] for k in train],
            "val": [cells[k] for k in val], "test": [cells[k] for k in test],
            "balance_score": best_score}
    return out


def pick_triples(doubles, n_val=3, n_test=3):
    """P2：三轴同时作用。要求三个二元投影都在 R1 train 的 double 里。"""
    train_doubles = set()
    for k, v in doubles.items():
        train_doubles |= set(v["train"])
    ok = []
    for t in itertools.product(NZ_IDX, repeat=3):
        cid = cell_id(*t)
        ix, iy, iw = t
        proj = [cell_id(ix, iy, ZERO_IDX), cell_id(ix, ZERO_IDX, iw),
                cell_id(ZERO_IDX, iy, iw)]
        if all(p in train_doubles for p in proj):
            ok.append((cid, proj))
    assert len(ok) >= n_val + n_test, f"可用 triple 太少: {len(ok)}"
    # 确定性分配：按符号模式分组交错排序（使 val/test 的正负号构成相近），
    # 再交错取 -- 保证 val 与 test 的符号/幅值分布接近。
    def sign_key(cid):
        ix, iy, iw = parse(cid)
        return (tuple(0 if i < 2 else 1 for i in (ix, iy, iw)),
                tuple(abs(VALUES[i]) for i in (ix, iy, iw)), parse(cid))
    ok_sorted = sorted(ok, key=lambda x: sign_key(x[0]))
    interleaved = []
    lo, hi = 0, len(ok_sorted) - 1
    while lo <= hi:
        interleaved.append(ok_sorted[lo]); lo += 1
        if lo <= hi:
            interleaved.append(ok_sorted[hi]); hi -= 1
    val_idx = list(range(0, min(2 * n_val, len(interleaved)), 2))
    val = [interleaved[i][0] for i in val_idx]
    test = [x[0] for i, x in enumerate(interleaved) if i not in set(val_idx)][:n_test]
    return {"available": [c for c, _ in ok], "n_available": len(ok),
            "val": val, "test": test,
            "val_projections": [p for c, p in ok if c in val],
            "test_projections": [p for c, p in ok if c in test]}


# ---------------- 时序模板 ----------------
# 每个模板使用**同一个 dwell 多重集**（只是顺序不同）-> 任意 split 子集的驻留分布
# 与切换次数完全相同，避免"训练短驻留、测试长驻留"（任务书 §4.3）。
DWELL_MULTISET = [0.6, 0.6, 0.9, 0.9, 1.2, 1.2, 1.5, 1.5,
                  0.6, 0.9, 1.2, 1.5]                      # 12 段，合计 12.6 s

_TEMPLATE_ORDERS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11],
    [5, 11, 4, 10, 3, 9, 2, 8, 1, 7, 0, 6],
    [3, 0, 9, 6, 1, 10, 7, 4, 11, 8, 5, 2],
    [8, 5, 2, 11, 7, 4, 1, 10, 6, 3, 0, 9],
]


def timing_templates():
    base = DWELL_MULTISET
    return {f"T{i}": [base[k] for k in order] for i, order in enumerate(_TEMPLATE_ORDERS)}


TEMPLATE_SPLIT = {"train": ["T0", "T1", "T2"], "val": ["T3"], "test": ["T4", "T5"]}


def build_schedules(n_scripts=12, cells_for_script=None, templates=None, seed=0):
    """为一条 group 生成 n_scripts 条脚本（cell 序列 + 时序模板）。

    cells_for_script: callable(script_idx) -> list[cell_id]（长度 == 12）
    """
    import numpy as np
    tpl = templates or timing_templates()
    tnames = list(tpl.keys())
    rng = np.random.default_rng(seed)
    scripts = []
    for s in range(n_scripts):
        tname = tnames[s % len(tnames)]
        dwells = tpl[tname]
        cells = cells_for_script(s, rng)
        assert len(cells) == len(dwells), (len(cells), len(dwells))
        scripts.append({"script_id": f"sc{s:02d}", "timing_template": tname,
                        "segments": [{"cell": c, "duration_s": float(d)}
                                     for c, d in zip(cells, dwells)]})
    return scripts


def regime_of_script(cells):
    """脚本所属 regime = 其 cell 集合能由哪个 regime 覆盖。"""
    types = {cell_type(c) for c in cells}
    return "R1" if "double" in types else "R0"
