#Training loop for Temporal Graph Trasformer simmalarly to task 4

import math
import numpy
import torch
from sklearn.metrics import roc_auc_score , average_precision_score
import ClinGroups
import Graph
import Models

#simple DS that returns (node feature, labels) ((hours, features), (Mortality (1 or 0)))
class GraphData(torch.utils.data.Dataset):
    def __init__ (self, X: numpy.ndarray, Y: numpy.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__ (self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#*    
def Warmup(optimiser, warmup_epochs, total_epoch):
    def LearnLambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epoch - warmup_epochs, 1)
        return .5 * (1 + math.cos(math.pi * progress))
    return LearnLambda(optimiser, LearnLambda)

#Optimiser: compute AUROC and AUPRC on loader
def Optimiser(model, loader, attention_mask, device):
    model.eval()
    all_y = []
    all_p = []
    with torch.no_grad():
        for x, y, in loader:
            x = x.to(device)
            logits = model(x, attention_mask)
            prblms = torch.sigmoid(logits).cpu().numpy()
            all_y.append(y.numpy())
            all_p.append(prblms)
    YTrue = numpy.concatenate(all_y)
    PTrue = numpy.concatenate(all_p)
    return{
        "auroc": roc_auc_score(YTrue, PTrue),
        "auprc": average_precision_score(YTrue, PTrue)
    }
        
        