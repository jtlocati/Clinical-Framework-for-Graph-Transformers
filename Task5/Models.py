#Initalize temporal graph transformer
"""
Ovgerview:
    1.) Embedding layer

"""

import math
import torch
import torch.nn

#This class initalizes Standard sinsodal positional encoding. Adding time-step information to each node so edges can be drawn in Graph.py Temporal Agacent
#(Connects same mimic stay_id nodes through time)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]