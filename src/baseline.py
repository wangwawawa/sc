from __future__ import annotations

import scanpy as sc

from .config import BaselineConfig


def run_baseline(adata_proc: sc.AnnData, cfg: BaselineConfig):
    sc.tl.pca(adata_proc, n_comps=cfg.n_pcs)
    sc.pp.neighbors(adata_proc, n_neighbors=cfg.n_neighbors, n_pcs=cfg.n_pcs)
    sc.tl.leiden(
    adata_proc,
    resolution=cfg.resolution,
    random_state=cfg.seed,
    flavor="leidenalg",
    )
    return adata_proc.obs["leiden"].to_numpy()
