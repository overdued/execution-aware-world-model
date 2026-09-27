"""V0.8 视觉预测模型（Stage C pilot 版；接口与 B4 单测契约兼容）。

三种变体（任务书 §C3）：
- V-DIRECT：h = R(H^p,U)；Ẑ = P(Z_t, h)
- V-AUX：   h = R(H^p,U)；Ê = G(h)；Ẑ = P(Z_t, h)          （L_exec 只经 R 反传）
- V-EXEC：  h = R(H^p,U)；Ê = G(h)；Ẑ = P(Z_t, h, stop_gradient(Ê))

共享结构 R/G/P：各 run 独立训练，不共享参数。R 的 GRU 与 G 可按预注册规则
从 V0.7 D/R1 seed 匹配 checkpoint 初始化（形状严格一致才允许）。
所有变体前向不接受任何 target/未来 GT（B4 T2 契约）。

目标表示：future latent = 时域 token × 4×4 空间 mean-pool token × 1024，
1s -> 8×16，2s -> 16×16。两个 lead 各一个 factored decoder head。
"""
import torch
import torch.nn as nn

D_VIS = 1024
T_CTX_TOK = 8                  # context clip 16 帧 / tubelet 2
S_TOK = 16                     # 4×4 空间 mean-pool
T_TGT_TOK = {1.0: 8, 2.0: 16}
D_TOK = 128                    # factored decoder 的 token embedding 维度
PROPRIO_DIM = 40               # deployable-candidate 本体（PROPRIO_SCHEMA.dim，privileged 已屏蔽）
CMD_DIM = 3
L_HIST, H_FUT = 20, 40         # 20Hz：过去 1s / 未来 2s


def mlp(dims, act=nn.GELU):
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), act()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class FactoredHead(nn.Module):
    """h -> [B, n_tok, D_VIS]：Linear(d_h, n_tok*D_TOK) + 共享 Linear(D_TOK, D_VIS)。"""

    def __init__(self, d_in, n_tok, d_tok=D_TOK, d_vis=D_VIS):
        super().__init__()
        self.n_tok, self.d_tok = n_tok, d_tok
        self.token_table = nn.Linear(d_in, n_tok * d_tok)
        self.channel_head = nn.Linear(d_tok, d_vis)

    def forward(self, h):
        B = h.shape[0]
        t = self.token_table(h).reshape(B, self.n_tok, self.d_tok)
        return self.channel_head(t)


class VisualPredictor(nn.Module):
    def __init__(self, variant="V-DIRECT", d_vis=D_VIS, hidden=256, d_trunk=512):
        super().__init__()
        assert variant in ("V-DIRECT", "V-AUX", "V-EXEC")
        self.variant = variant
        # Z_t：context token 池化投影
        self.z_pool = mlp([d_vis, hidden, hidden // 2])          # 1024->256->128
        # R：共享时序表示（历史 GRU + 未来命令编码）
        self.hist_gru = nn.GRU(PROPRIO_DIM + CMD_DIM, 128, batch_first=True)
        self.u_enc = mlp([H_FUT * CMD_DIM, 128, 128])
        self.r_trunk = mlp([128 + 128, hidden, hidden])          # -> h (256)
        # G：execution 头（V-DIRECT 无）
        self.exec_head = mlp([hidden, 128, H_FUT * CMD_DIM]) if variant != "V-DIRECT" else None
        # P：视觉 decoder（V-EXEC 额外吃 stop_gradient(Ê) 的 16 维压缩）
        d_p = hidden + hidden // 2 + (16 if variant == "V-EXEC" else 0)
        if variant == "V-EXEC":
            self.e_cond = mlp([H_FUT * CMD_DIM, 32, 16])
        self.p_trunk = mlp([d_p, d_trunk, d_trunk])
        self.head_1s = FactoredHead(d_trunk, T_TGT_TOK[1.0] * S_TOK)
        self.head_2s = FactoredHead(d_trunk, T_TGT_TOK[2.0] * S_TOK)

    def _pool_ctx(self, ctx_tokens):
        """ctx [B,8,1024] 或 [B,8,S,1024] -> [B,1024]（时域+空间 mean-pool）。"""
        if ctx_tokens.dim() == 4:
            ctx_tokens = ctx_tokens.mean(dim=2)
        return ctx_tokens.mean(dim=1)

    def forward(self, ctx_tokens, hist_phys, fut_cmd):
        """ctx_tokens [B,8,(S,)1024]（冻结 encoder 特征，调用方已 detach）；
        hist_phys [B,20,48+3]（masked proprio + 过去命令）；
        fut_cmd [B,40,3]（公开候选命令）。**不接受任何 target 信息。**
        返回 dict(pred_1s [B,8*16,1024], pred_2s [B,16*16,1024], e_hat [B,120] or None)。
        """
        B = ctx_tokens.shape[0]
        z = self.z_pool(self._pool_ctx(ctx_tokens))
        h_hist, _ = self.hist_gru(hist_phys)
        u = self.u_enc(fut_cmd.reshape(B, -1))
        h = self.r_trunk(torch.cat([h_hist[:, -1], u], dim=-1))
        e_hat = self.exec_head(h) if self.exec_head is not None else None
        p_in = torch.cat([h, z], dim=-1)
        if self.variant == "V-EXEC":
            p_in = torch.cat([p_in, self.e_cond(e_hat.detach())], dim=-1)  # stop_gradient(Ê)
        p = self.p_trunk(p_in)
        return {"pred_1s": self.head_1s(p), "pred_2s": self.head_2s(p), "e_hat": e_hat}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(variant, seed=0, **kw):
    torch.manual_seed(seed)
    return VisualPredictor(variant=variant, **kw)
