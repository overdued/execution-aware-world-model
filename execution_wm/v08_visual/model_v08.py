"""V0.8 视觉预测模型骨架（Stage B 因果性测试用；Stage C 训练沿用同一接口）。

三种变体（与任务书 §C 主比较一致）：
- V-DIRECT：context 视觉 token + 物理历史 + 未来命令 -> 直接预测 target 视觉 token；
- V-AUX：同上，额外辅助监督 execution（预测 Ê 只作为训练目标，不进解码路径）；
- V-EXEC：显式把预测的 Ê（stop_gradient）拼入解码输入。

所有变体的前向**只接受 context 与未来命令**，target RGB/GT 仅以监督标签出现；
B4 单测据此验证：任意改 target，前向输出必须逐位不变。
"""
import torch
import torch.nn as nn

D_VIS = 1024          # V-JEPA2 token 维度
T_CTX_TOK = 8         # context clip 16 帧 / tubelet 2
T_TGT_TOK = {1.0: 8, 2.0: 16}   # target clip 时域 token 数（16/32 帧）


def mlp(dims, act=nn.GELU):
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), act()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class VisualPredictor(nn.Module):
    """variant ∈ {V-DIRECT, V-AUX, V-EXEC}。"""

    def __init__(self, variant="V-DIRECT", lead_s=1.0, d_vis=D_VIS,
                 proprio_dim=48, cmd_dim=3, fut_horizon=20, hidden=512, n_tgt_tok=None):
        super().__init__()
        assert variant in ("V-DIRECT", "V-AUX", "V-EXEC")
        self.variant = variant
        self.lead_s = float(lead_s)
        self.fut_horizon = int(fut_horizon)
        self.n_tgt_tok = n_tgt_tok or T_TGT_TOK[self.lead_s]
        self.ctx_pool = mlp([d_vis, hidden, hidden])
        self.hist_enc = nn.GRU(proprio_dim + cmd_dim, 128, batch_first=True)
        trunk_out = hidden + 128 + self.fut_horizon * cmd_dim
        dec_in = trunk_out + (3 if variant == "V-EXEC" else 0)  # stop_gradient(Ê)
        self.decoder = mlp([dec_in, hidden, self.n_tgt_tok * d_vis])
        self.exec_head = mlp([trunk_out, 128, 3]) if variant != "V-DIRECT" else None

    def forward(self, ctx_tokens, hist_phys, fut_cmd):
        """ctx_tokens [B,T_CTX_TOK,D_VIS]（冻结 encoder 输出，已 detach）；
        hist_phys [B,L,proprio_dim+cmd_dim]；fut_cmd [B,H,cmd_dim]。
        **不接受任何 target 信息**（B4 测试锚点）。
        返回 dict(pred_tokens [B,n_tgt_tok,D_VIS], e_hat [B,3] or None)。
        """
        B = ctx_tokens.shape[0]
        h_vis = self.ctx_pool(ctx_tokens.mean(dim=1))            # [B,hidden]
        h_hist, _ = self.hist_enc(torch.cat([hist_phys], dim=-1))
        h = torch.cat([h_vis, h_hist[:, -1], fut_cmd.reshape(B, -1)], dim=-1)
        e_hat = self.exec_head(h) if self.exec_head is not None else None
        if self.variant == "V-EXEC":
            h = torch.cat([h, e_hat.detach()], dim=-1)           # stop_gradient(Ê)
        pred = self.decoder(h).reshape(B, self.n_tgt_tok, -1)
        return {"pred_tokens": pred, "e_hat": e_hat}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(variant, seed=0, **kw):
    torch.manual_seed(seed)
    return VisualPredictor(variant=variant, **kw)
