import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    """Domain-conditional FiLM modulation (Hypothesis H3).

    Learns a per-domain (gamma, beta) and applies a feature-wise affine transform
    ``x * (1 + gamma) + beta`` to a representation, modulating a single shared network
    per domain instead of training separate weight sets.

    Accepts ``x`` of shape ``[B, S, d_model]`` (modulation broadcast over the S cycle
    tokens) or ``[B, d_model]``. ``domain_id`` is a ``[B]`` long tensor in ``[0, num_domains)``.

    The embedding is **zero-initialized**, so at the start of training gamma = beta = 0 and
    the conditioner is the identity -- it does not perturb the (already strong) base model;
    domain specialization is learned from there.
    """

    def __init__(self, num_domains, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(num_domains, 2 * d_model)
        nn.init.zeros_(self.embed.weight)   # identity at init

    def forward(self, x, domain_id):
        gb = self.embed(domain_id)                       # [B, 2*d_model]
        gamma, beta = gb[:, :self.d_model], gb[:, self.d_model:]
        if x.dim() == 3:                                 # [B, S, d_model]
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        return x * (1 + gamma) + beta
