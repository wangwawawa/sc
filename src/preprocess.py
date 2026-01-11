from __future__ import annotations

import scanpy as sc

from .config import PreprocessConfig


def preprocess(adata: sc.AnnData, cfg: PreprocessConfig) -> sc.AnnData:
    adata_proc = adata.copy()

    sc.pp.normalize_total(adata_proc, target_sum=cfg.target_sum)
    sc.pp.log1p(adata_proc)
    sc.pp.highly_variable_genes(adata_proc, n_top_genes=cfg.n_top_genes)
    adata_proc = adata_proc[:, adata_proc.var["highly_variable"]].copy()
    sc.pp.scale(adata_proc, max_value=cfg.scale_max)

    return adata_proc