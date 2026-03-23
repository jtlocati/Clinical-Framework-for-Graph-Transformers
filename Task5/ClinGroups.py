"""
Initalization of catagories for task 5 edges
Steps:
    - Establish clinical feature groups
    - Skip distances
    - Model hyper-params
    - Graph varient names
"""

from dataclasses import dataclass
from typing import Dict, List
import os

#check spelling
CLINICAL_GROUPS: Dict[str, List[str]] = {
    "Cardiovascular": ["heart_rate", "sbp", "dbp", "mbp"],
    "Respiratory": ["resp_rate", "temperature", "spo2"],
    "Renal": ["bun", "creatinine", "calcium", "potassum", "sodium", "chloride", "bicarbonate"],
    "Metabolic": ["glucose", "glucose_lab", "lactate", "aniongap", "albumin"],
    "Hematologic": ["hematocrit", "hemoglobin", "platelet", "wbc", "ptt", "inr"]
} 



"""
Milti-sclae skip connections to eval throughout:
k values = an hour and we evalu in ascending order as specifyed by time skip
"""

TIME_SKIP = [2, 4, 8, 12, 24, 48]


SEEDS = [42, 123, 2024] #Same as task 4 for fair comparison

GRAPH_TYPES = ["temporal_only", "clinical_only", "full_graph"]


#Model Hyperparams(same as Task 4)
@dataclass
class TGTModelConfig:
    input_dim: int = 50 #for 25 features + 25 masks
    d_model: int = 64
    nhead = 4
    layers = 3
    feedforward = 128
    dropout = .2
    seq_len = 48

# TrainConfig in this context == Task4\config TrainConfig but may be implemented in this file dir if needed 
RESULTS_DIRECTORY = os.path.join(os.path.dirname(__file__), "outputs")