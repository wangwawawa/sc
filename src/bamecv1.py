from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
    views: Dict[str, np.ndarray] = {}

    sc.tl.pca(adata, n_comps=cfg.n_pcs)
    views["pca"] = adata.obsm["X_pca"]

    views["wpca"] = _weighted_pca(adata, cfg.n_pcs)

    sc.pp.neighbors(adata, n_neighbors=max(cfg.view_neighbors), n_pcs=cfg.n_pcs)
    sc.tl.diffmap(adata)
    views["diffmap"] = adata.obsm["X_diffmap"][:, : cfg.diffusion_components]

    return views

def _prepare_graphs_for_views(
    views: Dict[str, np.ndarray],
    cfg: BAMECConfig,
) -> Dict[Tuple[str, int], sc.AnnData]:
    """
    为每个 (view_name, n_neighbors) 预先构建一次 neighbors 图，后续重复跑 leiden 不再重建图。
    """
    graphs: Dict[Tuple[str, int], sc.AnnData] = {}
    for view_name, view in views.items():
        for k in cfg.view_neighbors:
            ad = sc.AnnData(view)
            sc.pp.neighbors(ad, n_neighbors=k)  # 只做一次
            graphs[(view_name, k)] = ad
    return graphs

# def _cluster_from_view(
#     view: np.ndarray,
#     n_neighbors: int,
#     resolution: float,
#     seed: int,
# ) -> np.ndarray:
#     adata_view = sc.AnnData(view)
#     sc.pp.neighbors(adata_view, n_neighbors=n_neighbors)
#     sc.tl.leiden(
#     adata_view,
#     resolution=resolution,
#     random_state=seed,
#     flavor="leidenalg",
#     )
#     return adata_view.obs["leiden"].to_numpy()

def _compute_quality_weights_with_penalty(
    base_clusterings: List[BaseClustering],
    views: Dict[str, np.ndarray],
    sample_size: int,
    cfg: BAMECConfig,
) -> List[BaseClustering]:
    """添加对聚类数量的惩罚，避免过度细分"""
    if len(base_clusterings) == 1:
        base_clusterings[0].weight = 1.0
        return base_clusterings

    scores = []
    for clustering in base_clusterings:
        labels = clustering.labels
        n_clusters = len(np.unique(labels))
        
        # 计算silhouette score
        try:
            sil_score = silhouette_score(
                views["pca"],
                labels,
                sample_size=min(sample_size, views["pca"].shape[0]),
                random_state=0,
            )
        except ValueError:
            sil_score = -1.0
        
        # 添加聚类数量惩罚（避免过多或过少的聚类）
        # 假设期望的聚类数在10-20之间
        expected_k = 15
        k_penalty = -0.1 * abs(n_clusters - expected_k) / expected_k
        
        final_score = sil_score + k_penalty
        scores.append(final_score)

    scores = np.array(scores, dtype=float)
    
    tau = max(float(cfg.weight_temperature), 1e-8)
    w = np.exp((scores - scores.max()) / tau)
    w = w / (w.sum() + 1e-8)

    top_k = int(getattr(cfg, "top_k", 0) or 0)
    if top_k > 0 and top_k < len(base_clusterings):
        idx = np.argsort(-w)[:top_k]
        base_clusterings = [base_clusterings[i] for i in idx]
        w = w[idx]
        w = w / (w.sum() + 1e-8)

    for clustering, weight in zip(base_clusterings, w):
        clustering.weight = float(weight)

    return base_clusterings

def _compute_quality_weights(
    base_clusterings: List[BaseClustering],
    space: np.ndarray,
    sample_size: int,
    cfg: BAMECConfig,
) -> List[BaseClustering]:
    if len(base_clusterings) == 1:
        base_clusterings[0].weight = 1.0
        return base_clusterings

    scores = []
    for clustering in base_clusterings:
        labels = clustering.labels
        try:
            score = silhouette_score(
                space,
                labels,
                sample_size=min(sample_size, space.shape[0]),
                random_state=0,
            )
        except ValueError:
            score = -1.0
        scores.append(score)

    scores = np.array(scores, dtype=float)

    # 温度 softmax（让权重更“尖锐”）
    tau = getattr(cfg, "weight_temperature", 0.1)
    tau = max(float(tau), 1e-8)
    w = np.exp((scores - scores.max()) / tau)
    w = w / (w.sum() + 1e-8)

    # 精英筛选：只保留 top fraction
    elite_frac = getattr(cfg, "elite_fraction", 1.0)
    elite_frac = float(elite_frac)
    if elite_frac < 1.0:
        k = max(1, int(len(base_clusterings) * elite_frac))
        idx = np.argsort(-w)[:k]
        base_clusterings = [base_clusterings[i] for i in idx]
        w = w[idx]
        w = w / (w.sum() + 1e-8)

    for clustering, weight in zip(base_clusterings, w):
        clustering.weight = float(weight)

    # ---- Top-K 精英筛选：只保留权重最大的K个，减少噪声也加速 ----
    print("[BAMEC] cfg.top_k =", getattr(cfg, "top_k", None),
      "elite_fraction =", getattr(cfg, "elite_fraction", None),
      "tau =", getattr(cfg, "weight_temperature", None))
    
    top_k = int(getattr(cfg, "top_k", 0) or 0)
    if top_k > 0 and top_k < len(base_clusterings):
        base_clusterings = sorted(base_clusterings, key=lambda x: x.weight, reverse=True)[:top_k]
        s = sum(c.weight for c in base_clusterings) + 1e-8
        for c in base_clusterings:
            c.weight = float(c.weight / s)
    print(f"[BAMEC] after top-k: kept {len(base_clusterings)} clusterings")

    return base_clusterings

def _compute_quality_weights_improved(
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

    return base_clusterings


def _hierarchical_consensus(
    base_clusterings: List[BaseClustering],
    edge_index: Tuple[np.ndarray, np.ndarray],
    n_cells: int,
) -> sp.csr_matrix:
    """先在每个view内部做共识，再在view之间做共识"""
    
    # 按view分组
    view_groups = {}
    for clustering in base_clusterings:
        view_name = clustering.name.split('_')[0]
        if view_name not in view_groups:
            view_groups[view_name] = []
        view_groups[view_name].append(clustering)
    
    row, col = edge_index
    final_weights = np.zeros_like(row, dtype=float)
    
    # 每个view内部先做共识
    for view_name, clusterings in view_groups.items():
        view_weights = np.zeros_like(row, dtype=float)
        total_weight = sum(c.weight for c in clusterings)
        
        for clustering in clusterings:
            same = clustering.labels[row] == clustering.labels[col]
            view_weights += (clustering.weight / total_weight) * same.astype(float)
        
        # view之间平等权重
        final_weights += view_weights / len(view_groups)
    
    consensus = sp.coo_matrix((final_weights, (row, col)), shape=(n_cells, n_cells))
    consensus = consensus.tocsr()
    consensus = consensus.maximum(consensus.T)
    return consensus


def _build_consensus_graph(
    labels_list: Iterable[BaseClustering],
    edge_index: Tuple[np.ndarray, np.ndarray],
    n_cells: int,
) -> sp.csr_matrix:
    row, col = edge_index
    weights = np.zeros_like(row, dtype=float)

    for clustering in labels_list:
        same = clustering.labels[row] == clustering.labels[col]
        weights += clustering.weight * same.astype(float)

    consensus = sp.coo_matrix((weights, (row, col)), shape=(n_cells, n_cells))
    consensus = consensus.tocsr()
    consensus = consensus.maximum(consensus.T)
    return consensus


def _get_edge_index(adata: sc.AnnData, n_neighbors: int, n_pcs: int) -> Tuple[np.ndarray, np.ndarray]:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    conn = adata.obsp["connectivities"].tocoo()
    return conn.row, conn.col


# def generate_base_clusterings(
#     views: Dict[str, np.ndarray],
#     cfg: BAMECConfig,
# ) -> List[BaseClustering]:
#     base_clusterings = []
#     for view_name, view in views.items():
#         for n_neighbors in cfg.view_neighbors:
#             for resolution in cfg.resolutions:
#                 for seed in cfg.seeds:
#                     labels = _cluster_from_view(view, n_neighbors, resolution, seed)
#                     name = f"{view_name}_k{n_neighbors}_r{resolution}_s{seed}"
#                     base_clusterings.append(BaseClustering(labels=labels, weight=1.0, name=name))
#     return base_clusterings

def generate_base_clusterings_fast(
    graphs: Dict[Tuple[str, int], sc.AnnData],
    cfg: BAMECConfig,
) -> List[BaseClustering]:
    base_clusterings: List[BaseClustering] = []
    for (view_name, k), ad in graphs.items():
        for resolution in cfg.resolutions:
            for seed in cfg.seeds:
                # 在同一张图上反复跑 leiden，不再重建 neighbors
                # sc.tl.leiden(ad, resolution=resolution, random_state=seed)
                sc.tl.leiden(
                    ad,
                    resolution=resolution,
                    random_state=seed,
                    flavor="leidenalg",
                )

                labels = ad.obs["leiden"].to_numpy().copy()  # copy 防止后续覆盖
                name = f"{view_name}_k{k}_r{resolution}_s{seed}"
                base_clusterings.append(BaseClustering(labels=labels, weight=1.0, name=name))
    return base_clusterings


def run_bamec(adata_proc: sc.AnnData, cfg: BAMECConfig) -> np.ndarray:
    views = build_views(adata_proc, cfg)
    # base_clusterings = generate_base_clusterings(views, cfg)
    graphs = _prepare_graphs_for_views(views, cfg)
    base_clusterings = generate_base_clusterings_fast(graphs, cfg)

    print(f"[BAMEC] Generated {len(base_clusterings)} base clusterings")

    # if cfg.ensemble_weights == "quality":
    #     base_clusterings = _compute_quality_weights(
    #         base_clusterings,
    #         space=views["pca"],
    #         sample_size=cfg.silhouette_sample_size,
    #         cfg=cfg,
    #     )

    # 根据配置选择质量权重计算方法
    use_penalty = getattr(cfg, "use_penalty", False)
    
    if cfg.ensemble_weights == "quality":
        if use_penalty:
            print("[BAMEC] Using quality weights WITH penalty (方案3)")
            base_clusterings = _compute_quality_weights_with_penalty(
                base_clusterings,
                views=views,
                sample_size=cfg.silhouette_sample_size,
                cfg=cfg,
            )
        else:
            print("[BAMEC] Using quality weights WITHOUT penalty (原始)")
            base_clusterings = _compute_quality_weights(
                base_clusterings,
                space=views["pca"],
                sample_size=cfg.silhouette_sample_size,
                cfg=cfg,
            )
    else:
        uniform_weight = 1.0 / max(len(base_clusterings), 1)
        for clustering in base_clusterings:
            clustering.weight = uniform_weight

    edge_index = _get_edge_index(adata_proc.copy(), cfg.consensus_neighbors, cfg.n_pcs)
    
    # 根据配置选择共识图构建方法
    use_hierarchical = getattr(cfg, "use_hierarchical", False)
    
    if use_hierarchical:
        print("[BAMEC] Using hierarchical consensus (方案4)")
        consensus = _hierarchical_consensus(base_clusterings, edge_index, adata_proc.n_obs)
    else:
        print("[BAMEC] Using standard consensus (原始)")
        consensus = _build_consensus_graph(base_clusterings, edge_index, adata_proc.n_obs)

    best_labels = None
    best_score = -1e18

    if getattr(cfg, "auto_select_consensus_resolution", False):
        res_list = list(getattr(cfg, "consensus_resolutions", [cfg.consensus_resolution]))
    else:
        res_list = [cfg.consensus_resolution]

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

        try:
            s = silhouette_score(
                views["pca"],
                labels,
                sample_size=min(cfg.silhouette_sample_size, views["pca"].shape[0]),
                random_state=0,
            )
        except ValueError:
            s = -1e18

        print(f"[BAMEC] Resolution {res}: silhouette={s:.4f}, n_clusters={len(np.unique(labels))}")

        if s > best_score:
            best_score = s
            best_labels = labels
            cfg.consensus_resolution = float(res)

    print(f"[BAMEC] Best resolution: {cfg.consensus_resolution}, best_score: {best_score:.4f}")
    return best_labels