"""Baseline 0: Action-only（first_work.md §8）。

输入只有 future commanded action，预测 future execution residual。
目标：证明 command alone 无法充分解释不同 physical condition 下的 execution。
"""
import torch
import torch.nn as nn


def build_mlp(in_dim, hidden_dims, out_dim, act=nn.ELU):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ActionOnlyBaseline(nn.Module):
    def __init__(self, horizon, hidden_dims=(256, 256), predict_uncertainty=False):
        super().__init__()
        self.H = horizon
        self.predict_uncertainty = predict_uncertainty
        self.net = build_mlp(3 * horizon, list(hidden_dims), 3 * horizon)
        if predict_uncertainty:
            self.logvar_net = build_mlp(3 * horizon, list(hidden_dims), 3 * horizon)

    def forward(self, future_action, **kwargs):
        B = future_action.shape[0]
        x = future_action.reshape(B, -1)
        r_hat = self.net(x).reshape(B, self.H, 3)
        out = {"r_hat": r_hat}
        if self.predict_uncertainty:
            out["logvar_r"] = self.logvar_net(x).reshape(B, self.H, 3)
        return out
