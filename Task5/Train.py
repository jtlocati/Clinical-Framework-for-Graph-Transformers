#Training loop for Temporal Graph Trasformer simmalarly to task 4

import math
import numpy
import torch
from sklearn.metrics import roc_auc_score , average_precision_score
import ClinGroups
import Graph
import Models
from Task4.config import TrainConfig

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
#*
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
        
"""
    Train one Temporal Graph Transformer variant on one seed.

    Parameters
    ----------
    variant : "temporal_only" | "clinical_only" | "full_graph"
    splits  : dict from data_pipeline.prepare_for_seed()
    seed    : random seed for reproducibility
    device  : "cuda" or "cpu"

    Returns
    -------
    result  : dict with "val" and "test" sub-dicts (each has auroc, auprc)
    log     : dict with per-epoch training history (for convergence plots)
"""       

def FullModel(var: str, splits: dict, seed: int, device: str = "cuda") -> tuple:
        ModelConfig = ClinGroups.TGTModelConfig()
        TrainConfigure = TrainConfig()
        #If CUDA is avalable then implement, while working localy leve device = "cpu"
        #device =  torch.device(device if torch.cuda.is_available() else "cpu")
        device = "cpu"

        #Define Grpah Nodes to plot on all model types
        x_train = Graph.PrepInput(splits["train"]["X"], splits["train"]["X"])
        x_valid = Graph.PrepInput(splits["val"]["X"], splits["val"]["M"])
        x_test = Graph.PrepInput(splits["test"]["X"], splits["test"]["M"])
        y_train = splits["train"]['y']
        y_valid = splits["val"]["y"]
        y_test = splits["test"]["y"]

        #load data from the GraphData function
        TrainLoader = torch.utils.data(GraphData(x_train, y_train), batch_size = TrainConfig.batch, shuffle = True, num_workers=0, pin_memory = True )
        ValidationLoader = torch.utils.data(GraphData(x_valid, y_valid), batch_size = TrainConfig.batch, shuffle = False, num_workers = 0, pin_mmory = True ) 
        TestLoader = torch.utils.data(GraphData(x_test, y_test), batch_size = TrainConfig.batch, shuffle = False, num_workers = 0, pin_memory = True)

        #Build graph attention mask to share full all patients
        adj = Graph.AdjType(var, ModelConfig)
        attention_mask = Graph.ADJ2AtMask(adj).to(device)

        number_of_edges = adj.sum().item() - ModelConfig.seq_len
        print(f"    Graph: {var}  |  {ModelConfig.seq_len} nodes  | {number_of_edges} edges  |  density={adj.float().mean():.3f}")

        #model
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        model = Models.TemporalGraph(ModelConfig).to(device)
        number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    NUMBER OF PARAMS: {number_of_params}")


        #Function to find loss given capped pos_weight
        raw_pw  = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        capped  = min(float(raw_pw), TrainConfig.max_pos_weight)
        pos_wt  = torch.tensor([capped], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)
        print(f"    pos_weight: raw={raw_pw:.2f}, capped={capped:.2f}")

        #optimise:    
        optimizer = torch.optim(model.parameters(), lr=TrainConfig.Lr, weight_decay=TrainConfig.WeightDecay)
        scheduler = Warmup(optimizer, TrainConfig.warmup_epochs, TrainConfig.epochs)

        #Train loop for all Graphical models: "Clinical", "Temporal", "Full combine model"
        #simplpe loop to repetily chack AUROC values and replace with new top value

        TopAUROC = 0
        Best = None
        Patience = 0
        log = {
            "epoch": [], "train_loss": [],
        "val_auroc": [], "val_auprc": [], "Lr": [],
        }
        for epoch in range(1, TrainConfig.epochs + 1):
        # train step
            model.train()
            losses = []
            for xb, yb in TrainLoader:
                xb, yb = xb.to(device), yb.float().to(device)
                logits = model(xb, attention_mask)
                loss   = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()

            # Valitate res
            val_m  = Optimiser(model, ValidationLoader, attention_mask, device)
            cur_lr = optimizer.param_groups[0]["lr"]

            log["epoch"].append(epoch)
            log["train_loss"].append(float(numpy.mean(losses)))
            log["val_auroc"].append(val_m["auroc"])
            log["val_auprc"].append(val_m["auprc"])
            log["lr"].append(cur_lr)

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}  loss={numpy.mean(losses):.4f}  "
                    f"val_auroc={val_m['auroc']:.4f}  "
                    f"val_auprc={val_m['auprc']:.4f}  lr={cur_lr:.6f}")

            # Stop early if needed or if patience runs out
            if val_m["auroc"] > best_val_auroc:
                best_val_auroc = val_m["auroc"]
                best_state = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= TrainConfig.patience:
                    print(f"    Early stop at epoch {epoch}:    (best val_auroc={best_val_auroc:.4f})")
                    break

        # Restore the best recorded checkpoint 
        model.load_state_dict(best_state)
        model.to(device)

        # Final evaluation 
        val_metrics  = Optimiser(model, ValidationLoader,  attention_mask, device)
        test_metrics = Optimiser(model, TestLoader, attention_mask, device)

        return {"val": val_metrics, "test": test_metrics}, log
