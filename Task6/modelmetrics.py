#Get y_true, y_prob, test_ids, and proper AUROC varibles
import time
import math
import numpy
import torch
import sklearn.metrics

from Task4.experiment import run_single, RunSingleWithPred
from Task5 import ClinGroups, Graph, Models, Train

import Task6.config

class GraphDS(torch.utils.data.Dataset):
    def __init__(self, X_graph, y):
        self.X = torch.from_numpy(X_graph).float()
        self.Y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def auroc(model, loader, mask, device):
    model.eval()
    ps = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            ps.append(torch.sigmoid(model(xb, mask)).cpu().numpy())
            ys.append(yb.numpy())
    return sklearn.metrics.roc_auc_score(numpy.concatenate(ys), numpy.concatenate(ps))

def Predict(model, loader, mask, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            ps.append(torch.sigmoid(model(xb, mask)).cpu().numpy())
            ys.append(yb.numpy())
    return numpy.concatenate(ys), numpy.concatenate(ps)
    

#warmup cosine -> Task5.Train.Warmup

def TAPTGT(varient: str, splits: dict, seed: int, device: str = "cpu") -> tuple:
    ModelConfig = ClinGroups.TGTModelConfig()
    TrainConfig = ClinGroups.TrainConfig()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #Node Features
    X_Train = Graph.PrepInput(splits["train"]["X"], splits["train"]["M"])
    X_Validation = Graph.PrepInput(splits["val"]["X"], splits["val"]["M"])
    X_Test = Graph.PrepInput(splits["test"]["X"], splits["test"]["M"])
    Y_Train = splits["train"]["y"]
    Y_Test = splits["test"]["y"]
    Y_Validation = splits["val"]["y"]


    TrainLoader = torch.utils.data.DataLoader(GraphDS(X_Train, Y_Train), batch_size=TrainConfig.batch, shuffle=True, num_workers=0, pin_memory=True)
    ValidateLoader = torch.utils.data.DataLoader(GraphDS(X_Validation, Y_Validation), batch_size=TrainConfig.batch, shuffle=False, num_workers=0, pin_memory=True )
    TestLoader = torch.utils.data.DataLoader(GraphDS(X_Test, Y_Test), batch_size=TrainConfig.batch, shuffle=False, num_workers=0, pin_memory=True)

    adj = Graph.AdjType(varient, ModelConfig.seq_len)
    attention_mask = Graph.ADJ2AtMask(adj).to(device)

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    model = Models.TemporalGraph(ModelConfig).to(device)

    raw_pw  = (Y_Train == 0).sum() / max((Y_Train == 1).sum(), 1)
    capped  = min(float(raw_pw), TrainConfig.max_pos_weight)
    pos_wt  = torch.tensor([capped], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    print(f"{Task6.config.AG}pos_weight: raw={raw_pw:.2f}, capped={capped:.2f}{Task6.config.AG}")

    Optimser = torch.optim.AdamW(model.parameters(), lr=TrainConfig.Lr, weight_decay=TrainConfig.WeightDecay)
    Sched = Train.Warmup(Optimser, TrainConfig.warmup_epochs, TrainConfig.epochs)

    #Train -> val -> Test
    BestAUROC:float = 0
    BestSTATE = None
    patience = 0

    #Train
    for epoch in range(1, TrainConfig.epochs + 1):
        model.train()
        for xb, yb in TrainLoader:
            xb, yb = xb.to(device), yb.float().to(device)
            logits = model(xb, attention_mask)
            loss = criterion(logits, yb)
            Optimser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            Optimser.step()
        Sched.step()

        #Validate
        ValAUROC = auroc(model, ValidateLoader, attention_mask, device)
        if epoch %10 == 0 or epoch ==1:
            print(f"EPOCH: {epoch:3d} AUROC VAL: {ValAUROC:.4f}")
        
        if ValAUROC > BestAUROC:
            BestAUROC = ValAUROC
            BestSTATE = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= TrainConfig.patience:
                print(f"EARLY STOP @{epoch}\n BEST AUROC VALUE={BestAUROC:.4f})")
                break
    model.load_state_dict(BestSTATE)
    model.to(device)
    YTrue, YProb = Predict(model, TestLoader, attention_mask, device)
    return YTrue, YProb

def CollectMetrics(splits: dict, seed) -> dict:
    predictions = {}
    for name in Task6.config.BASELINE_MODELS:
        print(f"{Task6.config.BAG}\nTraining: {name}\n{Task6.config.BAG}")
        t0 = time.time()
        _, y_true, y_prob = RunSingleWithPred(name, splits, seed, device="cpu")

        print(f"{Task6.config.BAG}  TRAINING FOR {name} DONE\n  Time {time.time() - t0 :.1f}\nAUROC: {sklearn.metrics.roc_auc_score(y_true, y_prob):.4f}\n{Task6.config.BAG}")

        predictions[name] = (y_true, y_prob)

    for TGT in Task6.config.TRANSFORMMERS:
        label = f"tgt_{TGT}"
        print(f"{Task6.config.BAG}\n    TRAINING: {TGT}\n{Task6.config.BAG}")
        t0 = time.time()

        y_true, y_prob = TAPTGT(TGT, splits, seed)
        print(f"{Task6.config.BAG}\n    DONE IN {time.time() - t0:.1f}\n    AUROC: {sklearn.metrics.roc_auc_score(y_true, y_prob):.4f}")
        predictions[label] = (y_true, y_prob)

    return predictions

def CollectMetricsMultiSeed(splits: dict, seeds: list) -> tuple:
    all_predictions = {}
    per_seed = {}

    for seed in seeds:
        print(f"\n{'='*60}\nSEED: {seed}\n{'='*60}")
        preds = CollectMetrics(splits, seed)
        per_seed[seed] = preds
        for model_name, (yt, yp) in preds.items():
            if model_name not in all_predictions:
                all_predictions[model_name] = []
            all_predictions[model_name].append((yt, yp))

    averaged = {}
    for model_name, runs in all_predictions.items():
        y_true = runs[0][0]
        y_probs = numpy.stack([r[1] for r in runs], axis=0)
        averaged[model_name] = (y_true, y_probs.mean(axis=0))
    return averaged, per_seed