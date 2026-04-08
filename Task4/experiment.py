"""
Handles:
  - Traditional model training (LogReg, LightGBM)
  - Neural model training with early stopping
  - PretrainedTST two phase training (pretrain -> fine tune)
  - Metric computation (AUROC, AUPRC)
"""

import math
import time
from pathlib import Path

import numpy
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import Task4.config as config
from Task4.dataset import make_flat_data, make_seq_loaders
from Task4.models import PretrainedTST, PretrainedTSTForPretraining, get_model


def compute_metrics(y_true: numpy.array, y_prob: numpy.ndarray) -> dict:
    #Return AUROC and AUPRC
    return {"auroc": roc_auc_score(y_true, y_prob), "auprc": average_precision_score(y_true, y_prob)}


def train_logistic_regression(flat_data: dict, seed: int) -> dict:
    #Logistic Regression on flattened summary features.
    X_tr, y_tr = flat_data["train"].as_arrays()
    X_val, y_val = flat_data["val"].as_arrays()
    X_te, y_te = flat_data["test"].as_arrays()

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_te)[:, 1]

    return {
        "val": compute_metrics(y_val, p_val),
        "test": compute_metrics(y_te, p_test),
        "test_outputs": {
            "ids": flat_data["test"].ids,
            "y_true": y_te,
            "y_prob": p_test,
        },
    }


def train_lightgbm(flat_data: dict, seed: int) -> dict:
    #LightGBM on flattened summary features.
    X_tr, y_tr = flat_data["train"].as_arrays()
    X_val, y_val = flat_data["val"].as_arrays()
    X_te, y_te = flat_data["test"].as_arrays()

    params = {**config.LGBM_PARAMS, "seed": seed}

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOP),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config.ROUNDS,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    p_val = model.predict(X_val)
    p_test = model.predict(X_te)

    return {
        "val": compute_metrics(y_val, p_val),
        "test": compute_metrics(y_te, p_test),
        "test_outputs": {
            "ids": flat_data["test"].ids,
            "y_true": y_te,
            "y_prob": p_test,
        },
    }


def _get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    #Linear warmup for `warmup_epochs`, then cosine decay to 0.
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _load_race_lookup():
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "Tests" / "outputs" / "Task2Demographics.csv"
    demo = pd.read_csv(demo_path)
    demo.columns = demo.columns.str.strip()
    return dict(zip(demo["stay_id"], demo["race"]))


def _compute_train_race_weights(train_ids, max_weight: float = 10.0):
    race_lookup = _load_race_lookup()

    counts = {}
    for sid in train_ids:
        race = race_lookup.get(int(sid), None)
        if race is None or race == "Multipule":
            continue
        counts[race] = counts.get(race, 0) + 1

    if not counts:
        return {}

    total = sum(counts.values())
    raw_weights = {race: total / count for race, count in counts.items()}
    return {race: min(weight, max_weight) for race, weight in raw_weights.items()}


def _compute_train_id_weight_lookup(train_ids, max_weight: float = 10.0):
    race_lookup = _load_race_lookup()
    race_weight_lookup = _compute_train_race_weights(train_ids, max_weight=max_weight)

    id_weight_lookup = {}
    for sid in train_ids:
        sid = int(sid)
        race = race_lookup.get(sid, None)
        id_weight_lookup[sid] = race_weight_lookup.get(race, 1.0)

    return id_weight_lookup


def _compute_train_id_race_lookup(train_ids):
    race_lookup = _load_race_lookup()
    return {int(sid): race_lookup.get(int(sid), None) for sid in train_ids}


def _compute_batch_group_loss_variance(ids, per_sample_loss, id_group_lookup):
    group_losses = {}
    for sid, loss_val in zip(ids, per_sample_loss):
        group = id_group_lookup.get(int(sid), None)
        if group is None or group == "Multipule":
            continue
        group_losses.setdefault(group, []).append(loss_val)

    group_means = []
    for losses in group_losses.values():
        if len(losses) == 0:
            continue
        group_means.append(torch.stack(losses).mean())

    if len(group_means) < 2:
        return per_sample_loss.new_tensor(0.0)

    return torch.stack(group_means).var(unbiased=False)


def _make_batch_sample_weights(ids, id_weight_lookup):
    if torch.is_tensor(ids):
        id_values = ids.detach().cpu().tolist()
    else:
        id_values = list(ids)

    weights = []
    for sid in id_values:
        weights.append(id_weight_lookup.get(int(sid), 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_neural(model: torch.nn.Module,
                 loaders: dict,
                 tc: config.TrainConfig = None,
                 device: str = "cuda",
                 loss_weighting_strategy: str = None,
                 max_loss_weight: float = 10.0,
                 subgroup_var_reg_strength: float = 0.0) -> dict:
    """
    Standard training loop with:
      - Binary loss
      - AdamW + cosine annealing
      - Early stopping on val AUROC
    """
    tc = tc or config.TrainConfig()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_race_weighted_loss = loss_weighting_strategy in {"race_weighted", "train_race_weighted"}

    y_train = loaders["train"].dataset.y.numpy()
    raw_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    capped_pw = min(raw_pw, tc.max_pos_weight)
    pos_weight = torch.tensor([capped_pw], dtype=torch.float32, device=device)
    print(f"    pos_weight: raw={raw_pw:.2f}, capped={capped_pw:.2f}")
    train_id_weight_lookup = None
    train_id_race_lookup = None
    if use_race_weighted_loss:
        train_id_weight_lookup = _compute_train_id_weight_lookup(
            loaders["train"].dataset.ids,
            max_weight=max_loss_weight,
        )
        train_id_race_lookup = _compute_train_id_race_lookup(loaders["train"].dataset.ids)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=tc.Lr, weight_decay=tc.WeightDecay)
    scheduler = _get_warmup_cosine_scheduler(optimizer, tc.warmup_epochs, tc.epochs)

    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, tc.epochs + 1):
        model.train()
        losses = []
        for ids, x, m, y in loaders["train"]:
            x, m, y = x.to(device), m.to(device), y.float().to(device)
            logits = model(x, m)

            if use_race_weighted_loss:
                per_sample_loss = criterion(logits, y)
                batch_weights = _make_batch_sample_weights(
                    ids,
                    train_id_weight_lookup,
                ).to(device)
                base_loss = (per_sample_loss * batch_weights).mean()

                if subgroup_var_reg_strength > 0.0:
                    var_loss = _compute_batch_group_loss_variance(
                        ids,
                        per_sample_loss,
                        train_id_race_lookup,
                    )
                    loss = base_loss + subgroup_var_reg_strength * var_loss
                else:
                    loss = base_loss
            else:
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_metrics = evaluate_neural(model, loaders["val"], device)
        val_auroc = val_metrics["auroc"]

        cur_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:3d}  loss={numpy.mean(losses):.4f}  "
                f"val_auroc={val_auroc:.4f}  val_auprc={val_metrics['auprc']:.4f}  "
                f"lr={cur_lr:.6f}"
            )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc.patience:
                print(f"    Early stop at epoch {epoch}  (best val_auroc={best_val_auroc:.4f})")
                break

    model.load_state_dict(best_state)
    model.to(device)

    val_m = evaluate_neural(model, loaders["val"], device)
    test_out = evaluate_neural(model, loaders["test"], device, return_outputs=True)
    return {
        "val": val_m,
        "test": {
            "auroc": test_out["auroc"],
            "auprc": test_out["auprc"],
        },
        "test_outputs": {
            "ids": test_out["ids"],
            "y_true": test_out["y_true"],
            "y_prob": test_out["y_prob"],
        },
    }


def evaluate_neural(model: torch.nn.Module, loader, device, return_outputs: bool = False):
    model.eval()
    all_ids, all_y, all_p = [], [], []
    with torch.no_grad():
        for ids, x, m, y in loader:
            x, m = x.to(device), m.to(device)
            logits = model(x, m)
            probs = torch.sigmoid(logits).cpu().numpy()
            if torch.is_tensor(ids):
                all_ids.append(ids.cpu().numpy())
            else:
                all_ids.append(numpy.asarray(ids))
            all_y.append(y.numpy())
            all_p.append(probs)

    y_ids = numpy.concatenate(all_ids)
    y_true = numpy.concatenate(all_y)
    y_prob = numpy.concatenate(all_p)
    metrics = compute_metrics(y_true, y_prob)

    if return_outputs:
        return {
            "ids": y_ids,
            "y_true": y_true,
            "y_prob": y_prob,
            **metrics,
        }

    return metrics


def train_pretrained_tst(pretrain_model: PretrainedTSTForPretraining,
                         loaders: dict,
                         device: str = "cuda",
                         loss_weighting_strategy: str = None,
                         max_loss_weight: float = 10.0,
                         subgroup_var_reg_strength: float = 0.0) -> dict:
    """
    Phase 1: Selfsupervised masked reconstruction pretraining
    Phase 2: Finetune encoder + classification head
    """
    pc = config.PretrainedTSTConfig()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    pretrain_model = pretrain_model.to(device)

    print("    Phase 1: Selfsupervised pretraining …")
    optimizer = AdamW(pretrain_model.parameters(), lr=pc.pretrain_lr)
    for epoch in range(1, pc.pretrain_epochs + 1):
        pretrain_model.train()
        losses = []
        for _, x, m, _ in loaders["train"]:
            x, m = x.to(device), m.to(device)
            loss = pretrain_model(x, m)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Pretrain epoch {epoch:3d}  recon_loss={numpy.mean(losses):.4f}")

    print("    Phase 2: Fine-tuning for classification …")
    encoder = pretrain_model.encoder
    ft_model = PretrainedTST(encoder, pc).to(device)

    return train_neural(
        ft_model,
        loaders,
        device=str(device),
        loss_weighting_strategy=loss_weighting_strategy,
        max_loss_weight=max_loss_weight,
        subgroup_var_reg_strength=subgroup_var_reg_strength,
    )


ALL_MODELS = [
    "LogisticRegression",
    "LightGBM",
    "LSTM",
    "StandardTransformer",
    "ChronoFormer",
    "PatchTST",
    "PretrainedTST",
    "TGTTemporal",
    "TGTClinical",
    "TGTFull",
    "TGT",
]


def run_single(model_name: str,
               splits: dict,
               seed: int,
               device: str = "cuda",
               sampling_strategy: str = None,
               max_sampling_weight: float = 10.0,
               loss_weighting_strategy: str = None,
               max_loss_weight: float = 10.0,
               subgroup_var_reg_strength: float = 0.0) -> dict:

    print(f"\n  ▶ {model_name}  (seed={seed})")
    t0 = time.time()

    if model_name == "LogisticRegression":
        flat = make_flat_data(splits)
        result = train_logistic_regression(flat, seed)

    elif model_name == "LightGBM":
        flat = make_flat_data(splits)
        result = train_lightgbm(flat, seed)

    elif model_name == "PretrainedTST":
        loaders = make_seq_loaders(
            splits,
            num_workers=0,
            sampling_strategy=sampling_strategy,
            max_sampling_weight=max_sampling_weight,
        )
        pretrain_model, _ = get_model("PretrainedTST", device=torch.device(device))
        result = train_pretrained_tst(
            pretrain_model,
            loaders,
            device,
            loss_weighting_strategy=loss_weighting_strategy,
            max_loss_weight=max_loss_weight,
            subgroup_var_reg_strength=subgroup_var_reg_strength,
        )

    else:
        loaders = make_seq_loaders(
            splits,
            num_workers=0,
            sampling_strategy=sampling_strategy,
            max_sampling_weight=max_sampling_weight,
        )
        model, _ = get_model(model_name, device=torch.device(device))
        result = train_neural(
            model,
            loaders,
            device=device,
            loss_weighting_strategy=loss_weighting_strategy,
            max_loss_weight=max_loss_weight,
            subgroup_var_reg_strength=subgroup_var_reg_strength,
        )

    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.1f}s  |  "
          f"test AUROC={result['test']['auroc']:.4f}  "
          f"AUPRC={result['test']['auprc']:.4f}")
    return result
