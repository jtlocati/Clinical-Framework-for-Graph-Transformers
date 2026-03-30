import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path

# Make Task4 importable when this script is run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
TASK4_DIR = REPO_ROOT / "Task4"
sys.path.insert(0, str(TASK4_DIR))

import config  # noqa: E402
from models import TGTModel  # noqa: E402


def compute_density_for_patient(missing_mask: torch.Tensor, edge_mode: str) -> tuple[int, float]:
    model = TGTModel(config.TGTConfig(edge_mode=edge_mode))
    adj = model._build_adj(missing_mask)  # (T, T) bool

    seq_len = adj.shape[0]
    num_edges = int(adj.sum().item() - seq_len)  # remove self-loops
    density = float(adj.float().mean().item())

    return num_edges, density


def main():
    data = np.load(TASK4_DIR / "data" / "mimic4_icu_mortality_48h.npz")
    M = data["M"]  # [num_patients, time_steps, num_features]

    variants = {
        "TGTTemporal": "temporal_only",
        "TGTClinical": "clinical_only",
        "TGTFull": "both",
    }

    rows = []
    for name, mode in variants.items():
        all_edges = []
        all_density = []

        for i in range(M.shape[0]):
            missing_mask = torch.tensor(M[i], dtype=torch.float32)
            num_edges, density = compute_density_for_patient(missing_mask, mode)
            all_edges.append(num_edges)
            all_density.append(density)

        rows.append(
            {
                "Model": name,
                "edge_mode": mode,
                "mean_num_edges": np.mean(all_edges),
                "std_num_edges": np.std(all_edges),
                "min_num_edges": np.min(all_edges),
                "max_num_edges": np.max(all_edges),
                "mean_density": np.mean(all_density),
                "std_density": np.std(all_density),
                "min_density": np.min(all_density),
                "max_density": np.max(all_density),
                "n_patients": M.shape[0],
            }
        )

    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / "Task5" / "results" / "graph_density_summary.csv"
    df.to_csv(out_path, index=False)

    print(df)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()