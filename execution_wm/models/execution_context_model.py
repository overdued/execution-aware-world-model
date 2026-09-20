"""Ours V0: Execution Context Model（first_work.md §10）。

组合 ContextEncoder + ExecutionPredictor:
    c_t = C_phi(history)
    r_hat = G_theta(current_state_latent, c_t, future_command)
    e_hat = future_command + r_hat
V0 使用 deterministic c_t。
"""
import torch
import torch.nn as nn

from .context_encoder import ContextEncoder
from .execution_predictor import ExecutionPredictor


class ExecutionContextModel(nn.Module):
    def __init__(self, proprio_dim, horizon, context_dim=8, gru_hidden=128,
                 hidden_dims=(256, 256), predict_uncertainty=False):
        super().__init__()
        self.context_encoder = ContextEncoder(proprio_dim, context_dim, gru_hidden)
        # state latent: 当前 proprio 压一层
        self.state_proj = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ELU())
        self.predictor = ExecutionPredictor(
            64, context_dim, horizon, hidden_dims, predict_uncertainty)

    def forward(self, history_proprio, history_action, current_state, future_action, **kwargs):
        c = self.context_encoder(history_proprio, history_action)
        s = self.state_proj(current_state)
        out = self.predictor(s, c, future_action)
        out["context"] = c
        return out
