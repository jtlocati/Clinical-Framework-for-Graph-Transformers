"""
Handles:
  - Traditional model training (LogReg, LightGBM)
  - Neural model training with early stopping
  - PretrainedTST two phase training (pretrain -> fine tune)
  - Metric computation (AUROC, AUPRC)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import math

import config
from models import get_model, PretrainedTST, PretrainedTSTEncoder, PretrainedTSTForPretraining
from dataset import  make_seq_loaders, make_flat_data
from torch.optim.lr_scheduler import LambdaLR

#metrics
def compute_metrics(y_true: np.array, y_prob: np.ndarray) -> dict:
    #Return AUROC and AUPRC
    return{"auroc": roc_auc_score(y_true, y_prob), "auprc": average_precision_score(y_true, y_prob),}

#Traditional Models:
def train_logistic_regression(flat_data: dict, seed: int) -> dict:
    #Logistic Regression on flattened summary features.
    X_tr, y_tr = flat_data["train"].as_arrays()
    X_val, y_val = flat_data["val"].as_arrays()
    X_te, y_te = flat_data["test"].as_arrays()

    model = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # predictions
    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_te)[:, 1]

    return {
        "val":  compute_metrics(y_val, p_val),
        "test": compute_metrics(y_te, p_test),
    }


def train_lightgbm(flat_data: dict, seed: int) -> dict:
    #LightGBM on flattened summary features.
    X_tr, y_tr = flat_data["train"].as_arrays()
    X_val, y_val = flat_data["val"].as_arrays()
    X_te, y_te = flat_data["test"].as_arrays()

    params = {**config.LGBM_PARAMS, "seed": seed}

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOP),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(params, dtrain, num_boost_round=config.ROUNDS, valid_sets=[dval], valid_names=["val"], callbacks=callbacks,)

    p_val  = model.predict(X_val)
    p_test = model.predict(X_te)

    return {
        "val":  compute_metrics(y_val, p_val),
        "test": compute_metrics(y_te, p_test),
    }


#  Neural model training loop
def _get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
#Linear warmup for `warmup_epochs`, then cosine decay to 0.
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs # linear 0→1
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) # cosine 1→0
    return LambdaLR(optimizer, lr_lambda)


def train_neural(model: nn.Module, loaders: dict, tc: config.TrainConfig = None, device: str = "cuda") -> dict:
    """
    Standard training loop with:
      - Binary loss
      - AdamW + cosine annealing
      - Early stopping on val AUROC
    """
    tc = tc or config.TrainConfig()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # class weight: cap pos_weight to prevent gradient explosion
    y_train = loaders["train"].dataset.y.numpy()
    raw_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    capped_pw = min(raw_pw, tc.max_pos_weight)
    pos_weight = torch.tensor([capped_pw], dtype=torch.float32, device=device)
    print(f"    pos_weight: raw={raw_pw:.2f}, capped={capped_pw:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer + warmup‑cosine scheduler
    optimizer = AdamW(model.parameters(), lr=tc.Lr, weight_decay=tc.WeightDecay)
    scheduler = _get_warmup_cosine_scheduler(optimizer, tc.warmup_epochs, tc.epochs)

    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, tc.epochs + 1):
        # train
        model.train()
        losses = []
        for x, m, y in loaders["train"]:
            x, m, y = x.to(device), m.to(device), y.float().to(device)
            logits = model(x, m)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # validate
        val_metrics = evaluate_neural(model, loaders["val"], device)
        val_auroc = val_metrics["auroc"]

        cur_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  loss={np.mean(losses):.4f}  "
                  f"val_auroc={val_auroc:.4f}  val_auprc={val_metrics['auprc']:.4f}  "
                  f"lr={cur_lr:.6f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc.patience:
                print(f"    Early stop at epoch {epoch}  (best val_auroc={best_val_auroc:.4f})")
                break

    # restore best
    model.load_state_dict(best_state)
    model.to(device)

    val_m  = evaluate_neural(model, loaders["val"], device)
    test_m = evaluate_neural(model, loaders["test"], device)
    return {"val": val_m, "test": test_m}


def evaluate_neural(model: nn.Module, loader, device) -> dict:
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, m, y in loader:
            x, m = x.to(device), m.to(device)
            logits = model(x, m)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_y.append(y.numpy())
            all_p.append(probs)
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)
    return compute_metrics(y_true, y_prob)


#  PretrainedTST two‑phase training


def train_pretrained_tst(pretrain_model: PretrainedTSTForPretraining,
                         loaders: dict, device: str = "cuda") -> dict:
    """
    Phase 1: SelFsupervised masked reconstruction pretraining
    Phase 2: Finetune encoder + classification head
    """
    pc = config.PretrainedTSTConfig()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    pretrain_model = pretrain_model.to(device)

    #  Phase 1: pretrain 
    print("    Phase 1: Selfsupervised pretraining …")
    optimizer = AdamW(pretrain_model.parameters(), lr=pc.pretrain_lr)
    for epoch in range(1, pc.pretrain_epochs + 1):
        pretrain_model.train()
        losses = []
        for x, m, _ in loaders["train"]:
            x, m = x.to(device), m.to(device)
            loss = pretrain_model(x, m)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"      Pretrain epoch {epoch:3d}  recon_loss={np.mean(losses):.4f}")

    #  Phase 2: fine‑tune 
    print("    Phase 2: Fine‑tuning for classification …")
    encoder = pretrain_model.encoder #extract pretrained encoder
    ft_model = PretrainedTST(encoder, pc).to(device)

    return train_neural(ft_model, loaders, device=str(device))


#Master experiment-runner


ALL_MODELS = [
    "LogisticRegression",
    "LightGBM",
    "LSTM",
    "StandardTransformer",
    "ChronoFormer",
    "PatchTST",
    "PretrainedTST",
]


def run_single(model_name: str, splits: dict, seed: int,
               device: str = "cuda") -> dict:
    #Run one model on one seed.  Returns dict with val/test metrics.

    print(f"\n  ▶ {model_name}  (seed={seed})")
    t0 = time.time()

    if model_name == "LogisticRegression":
        flat = make_flat_data(splits)
        result = train_logistic_regression(flat, seed)

    elif model_name == "LightGBM":
        flat = make_flat_data(splits)
        result = train_lightgbm(flat, seed)

    elif model_name == "PretrainedTST":
        loaders = make_seq_loaders(splits, num_workers=0)
        pretrain_model, _ = get_model("PretrainedTST", device=torch.device(device))
        result = train_pretrained_tst(pretrain_model, loaders, device)

    else:
        # LSTM, StandardTransformer, ChronoFormer, PatchTST
        loaders = make_seq_loaders(splits, num_workers=0)
        model, _ = get_model(model_name, device=torch.device(device))
        result = train_neural(model, loaders, device=device)

    elapsed = time.time() - t0
    print(f"    ✅ Done in {elapsed:.1f}s  |  "
          f"test AUROC={result['test']['auroc']:.4f}  "
          f"AUPRC={result['test']['auprc']:.4f}")
    return result
