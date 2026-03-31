"""models.py — All 7 model architectures for ICU Mortality Prediction
===================================================================
Traditional (sklearn / lightgbm):
  1. LogisticRegressionModel
  2. LightGBMModel

Neural (PyTorch):
  3. LSTMModel
  4. StandardTransformer

Advanced (PyTorch):
  5. ChronoFormer      — multi scale temporal attention
  6. PatchTST           — patch based time series transformer
  7. PretrainedTST      — self supervised pretrain -> fine tune"""

import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F 
import Task4.config as config 

#Shared Positional Encoding

class SinusoidalPE(nn.Module):
    #Standard sinusoidal positional encoding.

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]
    
#LSTM
class LSTMModel(nn.Module):
    """
    Bidirectional LSTM -> last hidden → MLP -> binary prediction.
    Input includes both features and missingness mask: (B, T, 2F).
    """

    def __init__(self, c: config.LSTMConfig = None):
        super().__init__()
        c = c or config.LSTMConfig()
        input_dim = config.NumberOfFeatures * 2 # features + mask
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=c.hidden_dim,
            num_layers=c.num_layers,
            batch_first=True,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
            bidirectional=c.bidirectional,
        )
        h_out = c.hidden_dim * (2 if c.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(h_out),
            nn.Dropout(c.dropout),
            nn.Linear(h_out, 1),
        )

    def forward(self, x, mask):
        # x: (B, T, F), mask: (B, T, F)
        inp = torch.cat([x, mask], dim=-1) # (B, T, 2F)
        out, _ = self.lstm(inp) # (B, T, H)
        last = out[:, -1, :] # (B, H)
        return self.head(last).squeeze(-1) # (B,)
    
#Studio Transformer
class StandardTransformer(nn.Module):
    """
    Vanilla Transformer encoder for time‑series classification.
    Projects (features + mask) → d_model, adds positional encoding,
    passes through TransformerEncoder, pools via [CLS] token or mean,
    then a linear head.
    """

    def __init__(self, c: config.TransformerConfig = None):
        super().__init__()
        c = c or config.TransformerConfig()
        input_dim = config.NumberOfFeatures * 2
        self.proj = nn.Linear(input_dim, c.d_model)
        self.pe   = SinusoidalPE(c.d_model, max_len=config.ObservionWindow + 1)

        # learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(c.d_model),
            nn.Linear(c.d_model, 1),
        )

    def forward(self, x, mask):
        B = x.size(0)
        inp = torch.cat([x, mask], dim=-1) # (B, T, 2F)
        h = self.proj(inp)# (B, T, d)
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, d)
        h = torch.cat([cls, h], dim=1) # (B, T+1, d)
        h = self.pe(h)
        h = self.encoder(h) # (B, T+1, d)
        cls_out = h[:, 0, :]   # (B, d)
        return self.head(cls_out).squeeze(-1)



#  5.  ChronoFormer  (multi‑scale temporal transformer)


class MultiScaleTemporalBlock(nn.Module):
    #Scales:  [1, 2, 4, ...]  (i.e. 1h, 2h, 4h bins)

    def __init__(self, d_model, nhead, ff_dim, dropout, scales):
        super().__init__()
        self.scales = scales
        self.attns = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=ff_dim, dropout=dropout,
                batch_first=True, activation="gelu",
            )
            for _ in scales
        ])
        # fusion: concat → project
        self.fuse = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h):
        # h: (B, T, d)
        outs = []
        for scale, attn in zip(self.scales, self.attns):
            if scale == 1:
                hs = attn(h) # (B, T, d)
                outs.append(hs[:, -1, :]) # take last step
            else:
                # average‑pool in time to reduce resolution
                B, T, D = h.shape
                T_new = T // scale
                h_ds = h[:, :T_new * scale, :].reshape(B, T_new, scale, D).mean(dim=2)
                hs = attn(h_ds) # (B, T_new, d)
                outs.append(hs[:, -1, :])

        fused = torch.cat(outs, dim=-1)  # (B, d * num_scales)
        return self.fuse(fused)  # (B, d)


class ChronoFormer(nn.Module):
    """
    1. Project input -> d_model
    2. Stack temporal blocks
    3. Classification head
    """

    def __init__(self, c: config.ChronoFormerConfig = None):
        super().__init__()
        c = c or config.ChronoFormerConfig()
        input_dim = config.NumberOfFeatures * 2
        self.proj = nn.Linear(input_dim, c.d_model)
        self.pe   = SinusoidalPE(c.d_model)

        scales = [2 ** i for i in range(c.num_scales)] # [1, 2, 4]

        self.blocks = nn.ModuleList([
            MultiScaleTemporalBlock(c.d_model, c.nhead, c.dim_feedforward,
                                    c.dropout, scales)
            for _ in range(c.num_layers)
        ])
        # residual connection between blocks needs input dim = output dim
        self.head = nn.Sequential(
            nn.LayerNorm(c.d_model),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, 1),
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=-1)
        h = self.pe(self.proj(inp)) # (B, T, d)
        # Stack blocks, each block returns (B, d) from multi‑scale fusion
        # We re‑broadcast back to sequence for next block
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                # last block: just get the pooled output
                pooled = block(h)# (B, d)
            else:
                pooled = block(h) # (B, d)
                h = h + pooled.unsqueeze(1) # show (B, 1, d)
        return self.head(pooled).squeeze(-1)


#  6.  PatchTST
class PatchTST(nn.Module):
    """
    1.  Split the time series into overlapping patches along the time axis
    2.  Linearly project each patch -> d_model
    3.  Add positional encoding
    4.  Transformer encoder
    5.  Pool -> classification head
    """

    def __init__(self, c: config.PatchTSTConfig = None, channel_independent: bool = False):
        super().__init__()
        c = c or config.PatchTSTConfig()
        self.channel_independent = channel_independent
        input_dim = config.NumberOfFeatures * 2   # features + mask
        self.patch_len = c.patch_len
        self.stride    = c.stride

        # number of patches
        num_patches = (config.ObservionWindow - c.patch_len) // c.stride + 1
        patch_input_dim = c.patch_len * (1 if channel_independent else input_dim)

        self.patch_proj = nn.Linear(patch_input_dim, c.d_model)
        self.pe = SinusoidalPE(c.d_model, max_len=num_patches + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout, batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.num_layers)

        if channel_independent:
            head_in = c.d_model * input_dim # one output per channel
        else:
            head_in = c.d_model

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(c.dropout),
            nn.Linear(head_in, 1),
        )
        self.num_patches = num_patches
        self.input_dim = input_dim

    def _patchify(self, x):
        #x: (B, T, D) -> (B, num_patches, patch_len * D)
        B, T, D = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)    # (B, P, D, patch_len)
        patches = patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        return patches # (B, P, patch_len * D)

    def _encode_patches(self, patches):
        #patches: (B, P, patch_dim) -> (B, d_model)
        B = patches.size(0)
        h = self.patch_proj(patches) # (B, P, d)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) # (B, P+1, d)
        h = self.pe(h)
        h = self.encoder(h)
        return h[:, 0, :] # CLS output (B, d)

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=-1) # (B, T, 2F)

        if not self.channel_independent:
            patches = self._patchify(inp) # (B, P, patch_len * 2F)
            cls_out = self._encode_patches(patches) # (B, d)
            return self.head(cls_out).squeeze(-1)
        else:
            # channel‑independent: process each channel separately
            outs = []
            for ch in range(self.input_dim):
                ch_data = inp[:, :, ch : ch + 1]               # (B, T, 1)
                patches = ch_data.unfold(1, self.patch_len, self.stride)
                patches = patches.reshape(inp.size(0), self.num_patches, self.patch_len)
                outs.append(self._encode_patches(patches))
            all_cls = torch.cat(outs, dim=-1)                  # (B, d * 2F)
            return self.head(all_cls).squeeze(-1)



# 7.Pretrained TST  (self‑supervised pretrain → fine‑tune)


class PretrainedTSTEncoder(nn.Module):
 
    #Shared encoder used in both pretraining and fine‑tuning.
 

    def __init__(self, c: config.PretrainedTSTConfig):
        super().__init__()
        input_dim = config.NumberOfFeatures * 2
        self.proj = nn.Linear(input_dim, c.d_model)
        self.pe   = SinusoidalPE(c.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout, batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.num_layers)

    def forward(self, x):
        h = self.pe(self.proj(x))
        return self.encoder(h)


class PretrainedTSTForPretraining(nn.Module):
    """
    Pretraining task: Masked Reconstruction.
    Randomly mask `mask_ratio` fraction of time steps,
    replace with learnable [MASK] token, reconstruct original.
    """

    def __init__(self, c: config.PretrainedTSTConfig = None):
        super().__init__()
        c = c or config.PretrainedTSTConfig()
        self.c = c
        input_dim = config.NumberOfFeatures * 2
        self.encoder = PretrainedTSTEncoder(c)
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.decoder = nn.Linear(c.d_model, input_dim)

    def forward(self, x, mask):
        """
        x:    (B, T, F)
        mask: (B, T, F)
        Returns: loss (scalar)
        """
        inp = torch.cat([x, mask], dim=-1)          # (B, T, 2F)
        B, T, D = inp.shape

        # generate random time‑step mask
        num_mask = int(T * self.c.mask_ratio)
        rand_idx = torch.rand(B, T, device=inp.device).argsort(dim=1)
        mask_idx = rand_idx[:, :num_mask]            # (B, num_mask)
        keep_idx = rand_idx[:, num_mask:]

        # build masked input
        inp_masked = inp.clone()
        batch_idx = torch.arange(B, device=inp.device).unsqueeze(1).expand(-1, num_mask)
        inp_masked[batch_idx, mask_idx] = self.mask_token.squeeze(0)

        # encode → decode
        h = self.encoder(inp_masked)                 # (B, T, d)
        recon = self.decoder(h)                      # (B, T, 2F)

        # loss only on masked positions
        target = inp[batch_idx, mask_idx]            # (B, num_mask, 2F)
        pred   = recon[batch_idx, mask_idx]
        loss   = F.mse_loss(pred, target)
        return loss


class PretrainedTST(nn.Module):
    #Fine tuning wrapper: frozen/unfrozen pretrained encoder + classification head.

    def __init__(self, pretrained_encoder: PretrainedTSTEncoder,
                 c: config.PretrainedTSTConfig = None):
        super().__init__()
        c = c or config.PretrainedTSTConfig()
        self.encoder = pretrained_encoder
        self.head = nn.Sequential(
            nn.LayerNorm(c.d_model),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, 1),
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=-1)
        h = self.encoder(inp) # (B, T, d)
        pooled = h.mean(dim=1) # (B, d)  mean‑pool over time
        return self.head(pooled).squeeze(-1)



#  Model registry (for the experiment runner)


def get_model(name: str, device: torch.device = None):
    """
    Factory function.  Returns (model, is_pretrained_tst).
    For PretrainedTST, we return the pretraining wrapper first;
    the experiment runner handles the two‑phase training.
    """
    device = device or torch.device("cpu")
    name_lower = name.lower().replace("-", "").replace("_", "")

    if name_lower == "lstm":
        return LSTMModel().to(device), False
    elif name_lower in ("transformer", "standardtransformer"):
        return StandardTransformer().to(device), False
    elif name_lower == "chronoformer":
        return ChronoFormer().to(device), False
    elif name_lower == "patchtst":
        return PatchTST(channel_independent=False).to(device), False
    elif name_lower in ("pretrainedtst", "pretrainedtimeseriestransformer"):
        # return pretraining model; caller must handle 2‑phase
        return PretrainedTSTForPretraining().to(device), True
    else:
        raise ValueError(f"Unknown model: {name}")
