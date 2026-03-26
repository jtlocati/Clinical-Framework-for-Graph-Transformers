#Initalize temporal graph transformer
"""
Ovgerview:
    1.) Embedding layer

"""

import math
import torch
import ClinGroups

#This class initalizes Standard sinsodal positional encoding. Adding time-step information to each node so edges can be drawn in Graph.py Temporal Agacent
#(sets up connection for same mimic stay_id nodes through time)
#*
class SinusoidalPE(torch.nn.Module):
    def __init__(self, Model, max_length = 48):
        super().__init__()
        #format: (48 and model)
        pe = torch.zeros(max_length, Model)
        #format: (48 and 1)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, Model, 2).float() * (-math.log(10000))/Model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #Format: (1, 48, Model)
        self.register_buffer("pe", pe.unsqueeze(0))

    #inject the time data into node.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]
    

#Transformer masked by adjacency (each node connects to neighbor relative to time (1 -> 48), no connections drawn to other nodes)
class GraphTransformer(torch.nn.Module):
    def __init__(self, Model, nhead, feedforward, dropout):
        super().__init__()

        #multi-head attention:
        self.attn = torch.nn.MultiheadAttention(embed_dim=Model, num_heads=nhead, dropout=dropout, batch_first=True)

        self.norm1 = torch.nn.LayerNorm(Model)
        self.norm2 = torch.nn.LayerNorm(Model)

        #Set the feedforward network to connect back to ___
        self.ffn = torch.nn.Sequential(torch.nn.Linear(Model, feedforward), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(feedforward, Model), torch.nn.Dropout(dropout))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attnMSK: torch.Tensor) -> torch.Tensor:
        #Initalize the muliti-head attention with graph mask.
        # resid + x is the residual connection.
        resid = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attnMSK)
        x = self.dropout(x)
        x=resid + x

        #feedforwarf
        resid = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = resid + x

        return x


#Full model initalization (end-to-end)

class TemporalGraph(torch.nn.Module):
    def __init__(self, config: ClinGroups.TGTModelConfig = None):
        config = config or ClinGroups.TGTModelConfig()

        super().__init__()
        self.embedding = torch.nn.Sequential(torch.nn.Linear(config.input_dim, config.d_model), torch.nn.LayerNorm(config.d_model), torch.nn.GELU(), torch.nn.Dropout(config.dropout))

        self.PositionalEncoder  = SinusoidalPE(config.d_model, max_length=config.seq_len)

        self.layers = torch.nn.ModuleList([GraphTransformer(config.d_model, config.nhead,config.feedforward, config.dropout,) for _ in range(config.layers)])

        #build prediction head
        self.classifer = torch.nn.Sequential(torch.nn.LayerNorm(config.d_model), torch.nn.Linear(config.d_model, 1))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.PositionalEncoder(x)
        for layer in self.layers:
            x=layer(x, attn_mask)
        
        #pool mean: patient_embedding -> mean(node_embedings) (batch, Model)
        x = x.mean(dim=1)

        return self.classifer(x).squeeze(-1)

#Final pararm output while running:
if __name__ == "__main__":
    model = TemporalGraph()
    ParameterNumber = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TemporalGraphTransformer  — {ParameterNumber:,} trainable parameters")
    # Dummy forward pass
    x = torch.randn(4, 48, 50)
    mask = torch.zeros(48, 48)
    out = model(x, mask)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (should be [4])")


