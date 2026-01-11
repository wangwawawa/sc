from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import silhouette_score

from .config import BAMECConfig


@dataclass
class BaseClustering:
    labels: np.ndarray
    weight: float
    name: str


def _compute_gene_weights(adata: sc.AnnData) -> np.ndarray:
    if "dispersions_norm" in adata.var:
        weights = adata.var["dispersions_norm"].to_numpy()
    else:
        mean = np.asarray(adata.X.mean(axis=0)).ravel()
        var = np.asarray(adata.X.var(axis=0)).ravel()
        weights = var / (mean + 1e-8)
    weights = np.nan_to_num(weights)
    weights = weights / (weights.mean() + 1e-8)
    return weights


def _weighted_pca(adata: sc.AnnData, n_pcs: int) -> np.ndarray:
    weights = _compute_gene_weights(adata)
    x = adata.X
    if sp.issparse(x):
        x_weighted = x.multiply(weights)
    else:
        x_weighted = x * weights

    adata_wpca = sc.AnnData(x_weighted)
    sc.tl.pca(adata_wpca, n_comps=n_pcs)
    return adata_wpca.obsm["X_pca"]


def build_views(adata: sc.AnnData, cfg: BAMECConfig) -> Dict[str, np.ndarray]:
    """构建多个数据视图"""
    views: Dict[str, np.ndarray] = {}

    # PCA视图
    sc.tl.pca(adata, n_comps=cfg.n_pcs)
    views["pca"] = adata.obsm["X_pca"]

    # 加权PCA视图
    views["wpca"] = _weighted_pca(adata, cfg.n_pcs)

    # Diffusion map视图
    sc.pp.neighbors(adata, n_neighbors=max(cfg.view_neighbors), n_pcs=cfg.n_pcs)
    sc.tl.diffmap(adata)
    views["diffmap"] = adata.obsm["X_diffmap"][:, : cfg.diffusion_components]

    return views


def _prepare_graphs_for_views(
    views: Dict[str, np.ndarray],
    cfg: BAMECConfig,
) -> Dict[Tuple[str, int], sc.AnnData]:
    """为每个(view, n_neighbors)组合预先构建邻居图"""
    graphs: Dict[Tuple[str, int], sc.AnnData] = {}
    for view_name, view in views.items():
        for k in cfg.view_neighbors:
            ad = sc.AnnData(view)
            sc.pp.neighbors(ad, n_neighbors=k)
            graphs[(view_name, k)] = ad
    return graphs


def generate_base_clusterings(
    graphs: Dict[Tuple[str, int], sc.AnnData],
    cfg: BAMECConfig,
) -> List[BaseClustering]:
    """生成基聚类"""
    base_clusterings: List[BaseClustering] = []
    
    for (view_name, k), ad in graphs.items():
        for resolution in cfg.resolutions:
            for seed in cfg.seeds:
                sc.tl.leiden(
                    ad,
                    resolution=resolution,
                    random_state=seed,
                    flavor="leidenalg",
                )
                
                labels = ad.obs["leiden"].to_numpy().copy()
                name = f"{view_name}_k{k}_r{resolution}_s{seed}"
                base_clusterings.append(
                    BaseClustering(labels=labels, weight=1.0, name=name)
                )
    
    return base_clusterings


def _compute_quality_weights(
    base_clusterings: List[BaseClustering],
    views: Dict[str, np.ndarray],
    sample_size: int,
    cfg: BAMECConfig,
) -> List[BaseClustering]:
    """使用多个视图的平均silhouette score"""
    if len(base_clusterings) == 1:
        base_clusterings[0].weight = 1.0
        return base_clusterings

    scores = []
    for clustering in base_clusterings:
        labels = clustering.labels
        view_scores = []
        
        # 在多个视图上评估
        for view_name, view in views.items():
            try:
                score = silhouette_score(
                    view,
                    labels,
                    sample_size=min(sample_size, view.shape[0]),
                    random_state=0,
                )
                view_scores.append(score)
            except ValueError:
                view_scores.append(-1.0)
        
        # 使用平均分数
        scores.append(np.mean(view_scores))

    scores = np.array(scores, dtype=float)
    
    # 温度 softmax
    tau = max(float(cfg.weight_temperature), 1e-8)
    w = np.exp((scores - scores.max()) / tau)
    w = w / (w.sum() + 1e-8)

    # Top-K 筛选
    top_k = int(getattr(cfg, "top_k", 0) or 0)
    if top_k > 0 and top_k < len(base_clusterings):
        idx = np.argsort(-w)[:top_k]
        base_clusterings = [base_clusterings[i] for i in idx]
        w = w[idx]
        w = w / (w.sum() + 1e-8)

    for clustering, weight in zip(base_clusterings, w):
        clustering.weight = float(weight)

    print(f"[BAMEC] Selected {len(base_clusterings)} base clusterings after filtering")
    
    return base_clusterings


def _build_consensus_graph(
    base_clusterings: List[BaseClustering],
    edge_index: Tuple[np.ndarray, np.ndarray],
    n_cells: int,
) -> sp.csr_matrix:
    """构建共识图"""
    row, col = edge_index
    weights = np.zeros_like(row, dtype=float)

    # 加权累加每个聚类的贡献
    for clustering in base_clusterings:
        same = clustering.labels[row] == clustering.labels[col]
        weights += clustering.weight * same.astype(float)

    # 构建对称的稀疏矩阵
    consensus = sp.coo_matrix((weights, (row, col)), shape=(n_cells, n_cells))
    consensus = consensus.tocsr()
    consensus = consensus.maximum(consensus.T)
    
    return consensus


def _get_edge_index(
    adata: sc.AnnData, n_neighbors: int, n_pcs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """获取邻居图的边索引"""
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    conn = adata.obsp["connectivities"].tocoo()
    return conn.row, conn.col


def run_bamec(adata_proc: sc.AnnData, cfg: BAMECConfig) -> np.ndarray:
    """
    运行BAMEC算法
    
    Args:
        adata_proc: 预处理后的AnnData对象
        cfg: BAMEC配置
        
    Returns:
        聚类标签数组
    """
    print("[BAMEC] Building views...")
    views = build_views(adata_proc, cfg)
    
    print("[BAMEC] Preparing graphs...")
    graphs = _prepare_graphs_for_views(views, cfg)
    
    print("[BAMEC] Generating base clusterings...")
    base_clusterings = generate_base_clusterings(graphs, cfg)
    print(f"[BAMEC] Generated {len(base_clusterings)} base clusterings")

    # 计算质量权重
    if cfg.ensemble_weights == "quality":
        print("[BAMEC] Computing quality-based weights...")
        base_clusterings = _compute_quality_weights(
            base_clusterings,
            views=views,  # 传入所有视图
            sample_size=cfg.silhouette_sample_size,
            cfg=cfg,
        )
    else:
        # 均匀权重
        uniform_weight = 1.0 / max(len(base_clusterings), 1)
        for clustering in base_clusterings:
            clustering.weight = uniform_weight

    # 构建共识图
    print("[BAMEC] Building consensus graph...")
    edge_index = _get_edge_index(
        adata_proc.copy(), cfg.consensus_neighbors, cfg.n_pcs
    )
    consensus = _build_consensus_graph(
        base_clusterings, edge_index, adata_proc.n_obs
    )

    # 在共识图上运行聚类
    print("[BAMEC] Running consensus clustering...")
    best_labels = None
    best_score = -1e18

    # 确定要尝试的分辨率列表
    if cfg.auto_select_consensus_resolution:
        res_list = list(cfg.consensus_resolutions)
    else:
        res_list = [cfg.consensus_resolution]

    # 尝试不同分辨率，选择最佳结果
    for res in res_list:
        adata_cons = adata_proc.copy()
        adata_cons.obsp["connectivities"] = consensus

        sc.tl.leiden(
            adata_cons,
            resolution=float(res),
            random_state=cfg.consensus_seed,
            flavor="leidenalg",
        )

        labels = adata_cons.obs["leiden"].to_numpy()

        # 使用silhouette score评估
        try:
            s = silhouette_score(
                views["pca"],
                labels,
                sample_size=min(cfg.silhouette_sample_size, views["pca"].shape[0]),
                random_state=0,
            )
        except ValueError:
            s = -1e18

        n_clusters = len(np.unique(labels))
        print(f"[BAMEC] Resolution {res:.2f}: silhouette={s:.4f}, n_clusters={n_clusters}")

        if s > best_score:
            best_score = s
            best_labels = labels
            cfg.consensus_resolution = float(res)

    print(f"[BAMEC] Best resolution: {cfg.consensus_resolution:.2f}, score: {best_score:.4f}")
    print(f"[BAMEC] Final number of clusters: {len(np.unique(best_labels))}")
    
    return best_labels