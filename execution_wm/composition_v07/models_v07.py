"""V0.7 Stage 3：两个主架构 D（Direct）与 I（Interaction，零锚定、每因素输出完整 [H,3]）。

D: r̂ = F(history, U)                      GRU history encoder + 完整未来命令
I: Ê = b(h) + Σ_i f_i(h,U_i) + Σ_{i<j} f_ij(h,U_i,U_j)
   f_i(h,U_i)      = φ_i(h,U_i) − φ_i(h,0)
   f_ij(h,U_i,U_j) = φ_ij(h,U_i,U_j) − φ_ij(h,U_i,0) − φ_ij(h,0,U_j) + φ_ij(h,0,0)
   0 = **整条未来该轴命令序列置零**；h 不置零（惯性与历史作用保留）
   每个 φ 输出完整 [H,3]（不是只输出自己的轴），否则预先禁止了 cross-axis 效应。

不加 context / oracle 摩擦 / 概率头 / 语言模型；D 与 I 参数量控制在 ±10%。
"""
import numpy as np
import torch
import torch.nn as nn

from execution_wm.validity_v061.schema import PROPRIO_SCHEMA

H = 40
D_IN = PROPRIO_SCHEMA.dim


def mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    return nn.Sequential(*layers[:-1])


class HistoryEncoder(nn.Module):
    """D 与 I 共用同一 history encoder（公平性）。"""

    def __init__(self, proprio_dim=D_IN, gru_hidden=128):
        super().__init__()
        self.gru = nn.GRU(proprio_dim + 3, gru_hidden, batch_first=True)
        self.out_dim = gru_hidden

    def forward(self, hp, ha):
        _, hn = self.gru(torch.cat([hp, ha], dim=-1))
        return hn[-1]


class DirectD(nn.Module):
    name = "D"

    def __init__(self, gru_hidden=128, hidden=(256, 256)):
        super().__init__()
        self.enc = HistoryEncoder(gru_hidden=gru_hidden)
        self.net = mlp([gru_hidden + 3 * H] + list(hidden) + [3 * H])

    def forward(self, hp, ha, fa):
        h = self.enc(hp, ha)
        return self.net(torch.cat([h, fa.reshape(len(fa), -1)], -1)).reshape(-1, H, 3)


def _zero_axis(fa, axis):
    z = fa.clone()
    z[:, :, axis] = 0.0
    return z


class InteractionI(nn.Module):
    name = "I"

    def __init__(self, gru_hidden=128, phi_hidden=65, pair_hidden=64, base_hidden=72):
        super().__init__()
        self.enc = HistoryEncoder(gru_hidden=gru_hidden)
        g = gru_hidden
        self.base = mlp([g, base_hidden, 3 * H])
        # 单因素 φ_i(h, U_i)
        self.phis = nn.ModuleList([mlp([g + H, phi_hidden, 3 * H]) for _ in range(3)])
        # 二元因素 φ_ij(h, U_i, U_j)
        self.pairs = nn.ModuleList([mlp([g + 2 * H, pair_hidden, 3 * H])
                                    for _ in range(3)])
        self.pair_idx = ((0, 1), (0, 2), (1, 2))

    def _phi_i_batch(self, i, h, fa_a, fa_b):
        """一次前向同时算 φ_i(h,U_i) 与 φ_i(h,0)，减少小模块调用次数（延迟）。"""
        x = torch.cat([torch.cat([h, fa_a[:, :, i]], -1),
                       torch.cat([h, fa_b[:, :, i]], -1)], dim=0)
        out = self.phis[i](x)
        return out[:len(h)].reshape(-1, H, 3), out[len(h):].reshape(-1, H, 3)

    def _phi_ij_batch(self, k, h, fa_list):
        """一次前向算 4 个零锚定项。"""
        a, b = self.pair_idx[k]
        xs = [torch.cat([h, f[:, :, a], f[:, :, b]], -1) for f in fa_list]
        out = self.pairs[k](torch.cat(xs, dim=0))
        n = len(h)
        return [out[i * n:(i + 1) * n].reshape(-1, H, 3) for i in range(len(fa_list))]

    def forward(self, hp, ha, fa):
        h = self.enc(hp, ha)
        zero = torch.zeros_like(fa)
        out = self.base(h).reshape(-1, H, 3)
        # 单因素零锚定（每次前向合并两项）
        for i in range(3):
            pi, pz = self._phi_i_batch(i, h, fa, zero)
            out = out + pi - pz
        # 二元零锚定（每次前向合并四项）
        for k, (a, b) in enumerate(self.pair_idx):
            fa_i0, fa_j0 = _zero_axis(fa, a), _zero_axis(fa, b)
            p_full, p_i0, p_j0, p_00 = self._phi_ij_batch(k, h, [fa, fa_i0, fa_j0, zero])
            out = out + p_full - p_i0 - p_j0 + p_00
        return out


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(model_name, seed=0, **kw):
    torch.manual_seed(seed)
    if model_name == "D":
        return DirectD()
    if model_name == "I":
        return InteractionI(**kw)
    raise ValueError(model_name)


if __name__ == "__main__":
    hp = torch.randn(4, 20, D_IN); ha = torch.randn(4, 20, 3); fa = torch.randn(4, H, 3)
    d = DirectD(); i = InteractionI()
    print("D params", n_params(d), "I params", n_params(i),
          "ratio", round(n_params(i) / n_params(d), 4))
    print("D out", d(hp, ha, fa).shape, "I out", i(hp, ha, fa).shape)
    # 零锚定性：U 全零时 I 应退化为 b(h)
    z = torch.zeros_like(fa)
    h = i.enc(hp, ha)
    assert torch.allclose(i(hp, ha, z), i.base(h).reshape(-1, H, 3), atol=1e-5)
    print("zero-anchoring OK: I(h, U=0) == b(h)")
