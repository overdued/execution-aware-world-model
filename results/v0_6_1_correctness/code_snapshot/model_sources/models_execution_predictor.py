"""ExecutionPredictor（first_work.md §10）。

r_hat = G_theta(current_state_latent, c_t, future_command)
e_hat = future_command + r_hat
§13: 预留 uncertainty 输出（mu_r, logvar_r），V0 默认关闭。
"""
import torch
import torch.nn as nn

from .baseline_action import build_mlp


class ExecutionPredictor(nn.Module):
    def __init__(self, state_dim, context_dim, horizon,
                 hidden_dims=(256, 256), predict_uncertainty=False):
        super().__init__()
        self.H = horizon
        self.predict_uncertainty = predict_uncertainty
        in_dim = state_dim + context_dim + 3 * horizon
        self.net = build_mlp(in_dim, list(hidden_dims), 3 * horizon)
        if predict_uncertainty:
            self.logvar_net = build_mlp(in_dim, list(hidden_dims), 3 * horizon)

    def forward(self, state_latent, context, future_command):
        B = state_latent.shape[0]
        x = torch.cat([state_latent, context, future_command.reshape(B, -1)], dim=-1)
        r_hat = self.net(x).reshape(B, self.H, 3)
        out = {"r_hat": r_hat, "e_hat": future_command + r_hat}
        if self.predict_uncertainty:
            out["logvar_r"] = self.logvar_net(x).reshape(B, self.H, 3)
        return out
