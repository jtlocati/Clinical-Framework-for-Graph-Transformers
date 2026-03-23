"""Initalization of catagories for task 5 edges
    - Establish clinical feature groups
    - Skip distances
    - Model hyper-params
    - Graph varient names
"""

from dataclasses import dataclass
from typing import Dict, List

#Establish clinical groups

#check spelling
CLINICAL_GROUPS: Dict[str, List[str]] = {
    "Cardiovascular": ["heart_rate", "sbp", "dbp", "mbp"],
    "Respiratory": ["resp_rate", "temperature", "spo2"],
    "Renal": ["bun", "creatinine", "calcium", "potassum", "sodium", "chloride", "bicarbonate"],
    "Metabolic": ["glucose", "glucose_lab", "lactate", "aniongap", "albumin"],
    "Hematologic": ["hematocrit", "hemoglobin", "platelet", "wbc", "ptt", "inr"]
} 
