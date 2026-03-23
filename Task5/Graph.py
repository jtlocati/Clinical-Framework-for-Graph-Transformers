#Build graph that masks for all graph varients (temporal, clinical, and full)
import torch
import numpy
import ClinGroups

#Adjacency builders:
#TemporalAdj -> Chain graph (each node connects to neighbor bidirectionaly) + self-loops for slef attention
#*
def TemporalAdj(seq_length = 48) -> torch.BoolTensor:   
    adj = torch.zeros(seq_length, seq_length, dtype=torch.bool)

    for temp in range(seq_length - 1):
        adj[temp, temp + 1] = True
        adj[temp + 1, temp] = True
    adj.fill_diagonal_(True)
    return adj

#ClinicalAdj -> connections driven by clinical conncections driven by clinical measuremtn patterns and influenced by TimeSkips + same self attention loops as temporal
#*
def ClinicalAdj(seq_length = 48, TSkip: list = None) -> torch.BoolTensor:
    if TSkip == None:
        TSkip = ClinGroups.TIME_SKIP
    
    adj = torch.zeros(seq_length, seq_length, dtype=torch.bool)
    for K in TSkip:
        for T in range(seq_length):
            if T+K < seq_length:
                adj[T, T+1] = True
                adj[T+1, T] = True
    adj.fill_diagonal_(True)
    return adj

def FullAdj(seq_length = 48, TSkip: list = None):
    Temp = TemporalAdj(seq_length)
    Clin = ClinicalAdj(seq_length, TSkip)
    return Temp | Clin


#Select Table varints as set in Task 5 config and link to corresponding builder function

def AdjType(buildee, seq_length = 48) -> torch.BoolTensor:
    builder = {"temporal_solo": TemporalAdj, "clinical_solo": ClinicalAdj, "full": FullAdj}
    if buildee not in builder:
        raise ValueError(f"{buildee} not valid for current builder types:\n{ClinGroups.GRAPH_TYPES}")
    
    return builder[buildee](seq_length)


def ADJ2AtMask(adj: torch.BoolTensor):
    mask = torch.zeros(adj.shape, dtype=torch.float32)
    mask[~adj] = float("-inf")
    return mask

#Build node features for graph by combineing featuere values and missinness mask from task4

def PrepInput(X: numpy.ndarray, M: numpy.ndarray) -> numpy.ndarray:
    #Returns in (N, hours, features)
    return numpy.concatenate([X, M], axis=1).astype(numpy.float32)

if __name__ == "__main__":
    for name in ["temporal_solo", "clinical_solo", "full"]:
        adj = AdjType(name, seq_length=48)
        NumEdges = adj.sum().item() - 48
        density = adj.float().mean().item()
        print(f"{name:20s}  edges={NumEdges:5d} density={density:.3f}  shape={tuple(adj.shape)}")
