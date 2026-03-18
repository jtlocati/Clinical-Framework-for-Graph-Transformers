#central spot for config for ICU Mortality Benchmark

from dataclasses import dataclass, field
from typing import List
import os

#initalize big query:
BQ_Project = "project-bb25f1c3-ff0a-47c9-9d0"
MIMIC = "physionet-data.mimiciv_3_1_derived"

#Define Tasks:
#Watch for patrine TP of 48 hours
ObservionWindow = 48
MinAge = 18

VitalFeatures = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose",]

LabFeatures  = ["albumin", "aniongap", "bicarbonate", "bun", "calcium", "chloride", "creatinine", "glucose_lab", "hematocrit", "hemoglobin","lactate", "platelet", "potassium", "ptt", "inr", "sodium", "wbc",]

TotalFeatures = VitalFeatures + LabFeatures

#26 -> 25
NumberOfFeatures = len(TotalFeatures)

#Outline Train to .70 second splits:
ValFrac = .25 #-> sets the validation time to .25 seconds
Test_Frac = .15 #-> sets the Test time to .15 seconds

Seeds = [42, 123, 2024]

#Training params:
@dataclass
class TrainConfig:
    batch: int = 128       # smaller batches = more stable gradients
    Lr: float = 5e-4            # lower LR to prevent overshoot
    WeightDecay: float = 1e-4
    epochs: int = 80
    patience: int = 10          # more patience — neural models need time
    warmup_epochs: int = 5      # linear warmup before cosine decay
    max_pos_weight: float = 3.0 # cap class weight to prevent gradient explosion
    num_workers: int = 0
    #num_workers = 4  #possibly edit to zero if running on a Microsoft machine to avoid 'SHAP' value errors?

#Model params:
@dataclass
class LSTMConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True

@dataclass
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2

@dataclass
class PatchTSTConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.2
    mask_ratio: float = 0.40    # fraction of time steps masked during pretraining
    pretrain_epochs: int = 30
    pretrain_lr: float = 5e-4

@dataclass
class ChronoFormerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    num_scales: int = 3         # temporal resolutions

@dataclass
class PretrainedTSTConfig:
#- Phase 1: self‑supervised masked‑reconstruction pretraining
#- Phase 2: fine-tune the encoder for classification
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.2
    mask_ratio: float = 0.40    # fraction of time steps masked during pretraining
    pretrain_epochs: int = 30
    pretrain_lr: float = 1e-3

#Light GBM params
LGBM_PARAMS = {"objective": "binary","metric": ["auc", "average_precision"],"learning_rate": 0.05,"num_leaves": 63,"max_depth": 7,"min_child_samples": 50,"subsample": 0.8,"colsample_bytree": 0.8,"verbose": -1,"n_jobs": -1,}
ROUNDS = 1000
EARLY_STOP = 50

#Route Path
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIECTORY = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(RESULTS_DIECTORY, exist_ok=True)