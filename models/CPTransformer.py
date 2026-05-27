import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding
from layers.intra_encoders import ConvIntraEncoder
from layers.domain_conditioning import FiLMConditioner
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        # [H2] number of input channels per cycle (3 = base [V,I,Q]; >3 with physics features)
        self.enc_in = getattr(configs, 'enc_in', 3)
        # [H1] intra-cycle encoder: 'linear' (original flatten->Linear) or 'conv' (1-D CNN)
        self.intra_encoder = getattr(configs, 'intra_encoder', 'linear')
        if self.intra_encoder == 'conv':
            self.conv_intra = ConvIntraEncoder(num_var=self.enc_in, fixed_len=self.charge_discharge_length,
                                               d_model=self.d_model, dropout=self.drop_rate)
        else:
            self.intra_flatten = nn.Flatten(start_dim=2)
            self.intra_embed = nn.Linear(self.charge_discharge_length*self.enc_in, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])

        self.pe = PositionalEmbedding(self.d_model)
        self.inter_TransformerEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ]
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.inter_flatten = nn.Flatten(start_dim=1)
        self.projection = nn.Linear(configs.d_model * self.early_cycle_threshold, configs.output_num)

        # [H3] optional domain-conditioning (FiLM) on the cycle-token embeddings
        self.multidomain = getattr(configs, 'multidomain', 'off')
        if self.multidomain == 'on':
            self.film = FiLMConditioner(getattr(configs, 'num_domains', 4), self.d_model)

    def forward(self, cycle_curve_data, curve_attn_mask, domain_id=None, return_embedding=False):
        '''
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        domain_id: [B] long tensor of domain ids (H3); None disables conditioning
        '''
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

        if self.intra_encoder == 'conv':
            cycle_curve_data = self.conv_intra(cycle_curve_data) # [B, early_cycle, d_model]
        else:
            cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, early_cycle, fixed_len * num_var]
            cycle_curve_data = self.intra_embed(cycle_curve_data)
        for i in range(self.e_layers):
            cycle_curve_data = self.intra_MLP[i](cycle_curve_data) # [B, early_cycle, d_model]

        if self.multidomain == 'on' and domain_id is not None:
            cycle_curve_data = self.film(cycle_curve_data, domain_id) # [B, early_cycle, d_model]

        cycle_curve_data = self.pe(cycle_curve_data) + cycle_curve_data
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, curve_attn_mask.shape[-1], dim=1) # [B, L, L]
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L, L]
        curve_attn_mask = curve_attn_mask==0 # set True to mask
        output, attns = self.inter_TransformerEncoder(cycle_curve_data, attn_mask=curve_attn_mask)

        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, L * d_model)
        preds = self.projection(output)  # (batch_size, num_classes)
        if return_embedding:
            return preds, output
        return preds
