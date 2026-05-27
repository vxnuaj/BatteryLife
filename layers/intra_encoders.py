import torch
import torch.nn as nn


class ConvIntraEncoder(nn.Module):
    """Structure-aware intra-cycle encoder (Hypothesis H1).

    Replaces the CyclePatch ``flatten -> Linear`` intra-cycle embedding. Each cycle
    arrives as a ``[num_var, fixed_len]`` signal (num_var=3 channels: voltage,
    current, capacity). A small 1-D CNN runs over the ``fixed_len`` (time) axis with
    the variables as input channels, so per-timestep cross-variable interaction and
    local temporal structure are *preserved* instead of being scrambled by a flatten.

    Input :  ``[B, S, C, L]``   (S = number of cycle tokens, C = num_var, L = fixed_len)
    Output:  ``[B, S, d_model]``

    Kept deliberately small: with the defaults below the parameter count
    (~37k at d_model=128) stays well under the linear encoder it replaces
    (``Linear(900 -> 128)`` = 115,328), so any performance change is attributable to
    structure rather than added capacity.
    """

    def __init__(self, num_var, fixed_len, d_model, hidden_channels=(16, 32),
                 kernel_size=7, pool_len=8, dropout=0.0):
        super().__init__()
        self.num_var = num_var
        self.fixed_len = fixed_len

        layers = []
        in_ch = num_var
        for out_ch in hidden_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_len)   # fixes output width regardless of L
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_ch * pool_len, d_model)

    def forward(self, x):
        # x: [B, S, C, L]
        B, S, C, L = x.shape
        x = x.reshape(B * S, C, L)        # treat every cycle token independently
        x = self.conv(x)                  # [B*S, C', L]
        x = self.pool(x)                  # [B*S, C', pool_len]
        x = x.flatten(start_dim=1)        # [B*S, C'*pool_len]
        x = self.dropout(x)
        x = self.proj(x)                  # [B*S, d_model]
        return x.reshape(B, S, -1)        # [B, S, d_model]
