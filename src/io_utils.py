from __future__ import annotations

from typing import Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp

LABEL_CANDIDATES = [
    "celltype",
    "cell_type",
    "celltype_major",
    "cell_type_major",
    "cell_type.fine",
    "cell_type_fine",
    "celltype_fine",
    "CellType",
    "cluster",
]


def read_h5ad(path: str) -> Tuple[sc.AnnData, np.ndarray, str]:
    adata = sc.read_h5ad(path)
    if sp.issparse(adata.X):
        if np.isnan(adata.X.data).any():
            adata.X.data = np.nan_to_num(adata.X.data)
    else:
        if np.isnan(adata.X).any():
            adata.X = np.nan_to_num(adata.X)

    label_col = None
    for candidate in LABEL_CANDIDATES:
        if candidate in adata.obs.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(
            f"No label column found. Available columns: {list(adata.obs.columns)}"
        )

    labels = adata.obs[label_col].to_numpy()
    return adata, labels, label_col