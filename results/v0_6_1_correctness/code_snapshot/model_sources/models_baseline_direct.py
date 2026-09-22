"""Baseline 1: Direct history predictor（first_work.md §9）。

r_hat = F(history_proprio, history_action, future_action)
普通 direct predictor，无显式 execution context。
"""
import torch
import torch.nn as nn

from .baseline_action import build_mlp


class DirectPredictor(nn.Module):
    def __init__(self, proprio_dim, horizon, gru_hidden=128,
                 hidden_dims=(256, 256), predict_uncertainty=False):
        super().__init__()
        self.H = horizon
        self.predict_uncertainty = predict_uncertainty
        self.encoder = nn.GRU(proprio_dim + 3, gru_hidden, batch_first=True)
        in_dim = gru_hidden + 3 * horizon
        self.net = build_mlp(in_dim, list(hidden_dims), 3 * horizon)
        if predict_uncertainty:
            self.logvar_net = build_mlp(in_dim, list(hidden_dims), 3 * horizon)

    def forward(self, history_proprio, history_action, future_action, **kwargs):
        B = history_proprio.shape[0]
        h_in = torch.cat([history_proprio, history_action], dim=-1)
        _, h_n = self.encoder(h_in)
        h = h_n[-1]                                   # [B, gru_hidden]
        x = torch.cat([h, future_action.reshape(B, -1)], dim=-1)
        r_hat = self.net(x).reshape(B, self.H, 3)
        out = {"r_hat": r_hat}
        if self.predict_uncertainty:
            out["logvar_r"] = self.logvar_net(x).reshape(B, self.H, 3)
        return out
