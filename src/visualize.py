from __future__ import annotations

from pathlib import Path

import scanpy as sc

from .baseline import run_baseline
from .bamecv1 import run_bamec
from .config import ExperimentConfig
from .io_utils import read_h5ad
from .preprocess import preprocess


def plot_umap(adata_proc: sc.AnnData, label_key: str, output_dir: Path, dataset: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(output_dir)
    sc.tl.pca(adata_proc, n_comps=50)
    sc.pp.neighbors(adata_proc, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_proc)

    sc.pl.umap(adata_proc, color=label_key, save=f"_{dataset}_truth.png", show=False)
    if "baseline" in adata_proc.obs:
        sc.pl.umap(adata_proc, color="baseline", save=f"_{dataset}_baseline.png", show=False)
    if "bamec" in adata_proc.obs:
        sc.pl.umap(adata_proc, color="bamec", save=f"_{dataset}_bamec.png", show=False)


def main() -> None:
    cfg = ExperimentConfig()
    dataset = "turtle"
    path = "data/Tosches_turtle.h5ad"

    adata, _, label_key = read_h5ad(path)
    adata_proc = preprocess(adata, cfg.preprocess)

    adata_proc.obs["baseline"] = run_baseline(adata_proc.copy(), cfg.baseline)
    adata_proc.obs["bamec"] = run_bamec(adata_proc.copy(), cfg.bamec)

    plot_umap(adata_proc, label_key, Path("results/figures"), dataset)


if __name__ == "__main__":
    main()