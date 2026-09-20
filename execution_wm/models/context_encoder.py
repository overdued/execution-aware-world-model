"""ContextEncoder（first_work.md §11）。

c_t = C_phi(history)，GRU 编码过去 1 秒 history -> R^context_dim。
history 只用 proprioception + past command，不含任何仿真 metadata
（friction / actuator strength / terrain ID 禁止输入，§11）。
"""
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(self, proprio_dim, context_dim=8, gru_hidden=128):
        super().__init__()
        self.context_dim = context_dim
        self.gru = nn.GRU(proprio_dim + 3, gru_hidden, batch_first=True)
        self.proj = nn.Linear(gru_hidden, context_dim)

    def forward(self, history_proprio, history_action):
        h_in = torch.cat([history_proprio, history_action], dim=-1)
        _, h_n = self.gru(h_in)
        c = self.proj(h_n[-1])                        # [B, context_dim]
        return c
