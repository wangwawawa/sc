from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .baseline import run_baseline
from .config import ExperimentConfig
from .io_utils import read_h5ad
from .metrics import calc_ari, calc_nmi, calc_acc
from .preprocess import preprocess


DEFAULT_DATASETS = {
    "turtle": "data/Tosches_turtle.h5ad",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="turtle", choices=list(DEFAULT_DATASETS.keys()))
    parser.add_argument("--output", default="results/baseline_grid.csv")

    # 网格：你也可以按需改大/改小
    parser.add_argument("--neighbors", nargs="*", type=int, default=[10, 15, 30, 50])
    parser.add_argument("--resolutions", nargs="*", type=float, default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.5])

    # 可选：也扫 PC 数（先不扫也行，保持简单）
    parser.add_argument("--pcs", type=int, default=None)

    args = parser.parse_args()

    cfg = ExperimentConfig()
    data_path = DEFAULT_DATASETS[args.dataset]

    adata, labels, _ = read_h5ad(data_path)
    adata_proc = preprocess(adata, cfg.preprocess)

    results = []
    best_row = None
    best_ari = -1.0

    # 如果你想扫 PC 数，可以把 cfg.baseline.n_pcs 也放进循环
    base_pcs = cfg.baseline.n_pcs if args.pcs is None else args.pcs

    for k in args.neighbors:
        for r in args.resolutions:
            cfg.baseline.n_neighbors = int(k)
            cfg.baseline.resolution = float(r)
            cfg.baseline.n_pcs = int(base_pcs)

            pred = run_baseline(adata_proc.copy(), cfg.baseline)

            nmi = calc_nmi(labels, pred)
            ari = calc_ari(labels, pred)
            acc = calc_acc(labels, pred)

            row = {
                "dataset": args.dataset,
                "n_neighbors": k,
                "resolution": r,
                "n_pcs": base_pcs,
                "seed": cfg.baseline.seed,
                "nmi": nmi,
                "ari": ari,
                "acc": acc,
            }
            results.append(row)

            if ari > best_ari:
                best_ari = ari
                best_row = row

            print(f"[grid] k={k:<3d} res={r:<4.1f} pcs={base_pcs:<3d} -> ARI={ari:.6f}")

    df = pd.DataFrame(results).sort_values(by="ari", ascending=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n===== BEST BASELINE (by ARI) =====")
    print(best_row)
    print(f"Saved grid results to: {out_path}")


if __name__ == "__main__":
    main()
