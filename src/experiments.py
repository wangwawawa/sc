from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .baseline import run_baseline
from .bamecv1 import run_bamec
from .config import ExperimentConfig
from .io_utils import read_h5ad
from .metrics import calc_acc, calc_ari, calc_nmi
from .preprocess import preprocess

DATASETS = {
    "turtle": "data/Tosches_turtle.h5ad",
    "lung": "data/Quake_Lung.h5ad",
    "diaphragm": "data/Quake_Diaphragm.h5ad",
}


def evaluate(true_labels, pred_labels) -> Dict[str, float]:
    return {
        "nmi": calc_nmi(true_labels, pred_labels),
        "ari": calc_ari(true_labels, pred_labels),
        "acc": calc_acc(true_labels, pred_labels),
    }


def run_dataset(name: str, path: str, cfg: ExperimentConfig) -> List[dict]:
    print(f"\n========== DATASET: {name} ==========")
    print(f"[{name}] reading: {path}")
    adata, labels, _ = read_h5ad(path)

    print(f"[{name}] preprocessing...")
    adata_proc = preprocess(adata, cfg.preprocess)

    results = []

    # ---- baseline ----
    print(f"[{name}] running BASELINE (k={cfg.baseline.n_neighbors}, res={cfg.baseline.resolution}, pcs={cfg.baseline.n_pcs}, seed={cfg.baseline.seed}) ...")
    baseline_pred = run_baseline(adata_proc.copy(), cfg.baseline)
    baseline_metrics = evaluate(labels, baseline_pred)
    print(f"[{name}] BASELINE done -> NMI={baseline_metrics['nmi']:.6f}, ARI={baseline_metrics['ari']:.6f}, ACC={baseline_metrics['acc']:.6f}")

    results.append(
        {
            "dataset": name,
            "method": "baseline",
            "seed": cfg.baseline.seed,
            "n_neighbors": cfg.baseline.n_neighbors,
            "resolution": cfg.baseline.resolution,
            "views": "pca",
            "ensemble_runs": 1,
            **baseline_metrics,
        }
    )

    # ---- bamec ----
    total_runs = len(cfg.bamec.view_neighbors) * len(cfg.bamec.resolutions) * len(cfg.bamec.seeds) * 3
    print(f"[{name}] running BAMEC (ensemble_runs={total_runs}, consensus_k={cfg.bamec.consensus_neighbors}, consensus_res={cfg.bamec.consensus_resolution}) ...")
    bamec_pred = run_bamec(adata_proc.copy(), cfg.bamec)
    bamec_metrics = evaluate(labels, bamec_pred)
    print(f"[{name}] BAMEC done -> NMI={bamec_metrics['nmi']:.6f}, ARI={bamec_metrics['ari']:.6f}, ACC={bamec_metrics['acc']:.6f}")

    results.append(
        {
            "dataset": name,
            "method": "bamec",
            "seed": cfg.bamec.consensus_seed,
            "n_neighbors": cfg.bamec.consensus_neighbors,
            "resolution": cfg.bamec.consensus_resolution,
            "views": ",".join(["pca", "wpca", "diffmap"]),
            "ensemble_runs": total_runs,
            **bamec_metrics,
        }
    )

    print(f"========== DATASET: {name} finished ==========\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--output", default="results/metrics.csv")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    all_results = []

    for name in args.datasets:
        print(f"\n>>> Running dataset: {name}")

        path = DATASETS.get(name)
        if path is None:
            raise ValueError(f"Unknown dataset: {name}")
        full_path = Path(path)
        all_results.extend(run_dataset(name, str(full_path), cfg))

    df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df)


if __name__ == "__main__":
    main()