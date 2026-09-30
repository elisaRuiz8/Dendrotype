import os
import types
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import uuid
import traceback
import colorsys

import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import scvi
except ImportError:
    scvi = None

try:
    import torch
except ImportError:
    torch = None

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

try:
    import scanpy as sc
except ImportError:
    sc = None

try:
    import bbknn
except ImportError:
    bbknn = None

try:
    import celltypist
except ImportError:
    celltypist = None

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend so pyplot never opens its own live Tk window;
    # we embed every figure manually via FigureCanvasTkAgg instead.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
except ImportError:
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    FigureCanvasAgg = None
    plt = None

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    Image = None
    ImageDraw = None
    ImageTk = None

BG_MAIN = "#1f232a"
BG_PANEL = "#262c36"
BG_PANEL_ALT = "#2d3440"
BG_SELECTED = "#455a64"  # blue-gray hover color for regular buttons
HIGHLIGHT_FG = "#111827"  # dark text so it stays readable on the light branch-color backgrounds
BG_INPUT = "#20252d"
FG_MAIN = "#e5e7eb"
FG_MUTED = "#cbd5e1"
ACCENT = "#60a5fa"
BORDER = "#4b5563"
LOG_BG = "#151922"
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939"
]

SELECTED_BORDER_COLOR = "#D1D5DB"  # thicker light-gray outline for the currently-selected row
SELECTED_BORDER_WIDTH = 2
#DEFAULT_CELLTYPIST_MODEL = "Human_IPF_Lung.pkl"  # auto-loaded at startup; override via 'Load CellTypist Model'
DEFAULT_CELLTYPIST_MODEL = "Human_Lung_Atlas.pkl"  # auto-loaded at startup; override via 'Load CellTypist Model'
GREEN_HIGHLIGHT_ROW_COLOR = "#A5D6A7"  # row background for a node with its own assigned cell type
UNASSIGNED_LEAF_ROW_COLOR = "#D0D3D8"  # light grey - a leaf that still needs a cell type assigned
NOT_ASSIGNED_CELL_TYPE_LABEL = "Not assigned"  # explicit combo box option to clear/undo a cell type assignment
NOT_CLEAR_CELL_TYPE_LABEL = "Not_clear"  # auto-assigned when no single predicted cell type reaches the "Not clear below" threshold


@dataclass
class ClusterNode:
    node_id: str
    label: str
    adata: object
    parent_id: Optional[str] = None
    cluster_value: Optional[str] = None
    depth: int = 0
    children: List[str] = field(default_factory=list)
    saved: bool = False
    last_cluster_key: Optional[str] = None
    child_resolution: Optional[float] = None
    resolution: Optional[float] = None
    umap_adata: Optional[object] = None  # processed/embedded copy used only for the UMAP preview; never overwrites adata
    predicted_labels: Optional[object] = None  # pandas Series (index=cell barcode) of per-cell CellTypist predicted_labels
    assigned_cell_type: Optional[str] = None  # manually assigned via the Tree Nodes list combo box; only leaves may have one
    integration_adata: Optional[object] = None  # batch-corrected copy (BBKNN or SCVI) with a neighbor graph, awaiting Leiden clustering
    integration_method: Optional[str] = None  # "bbknn" or "scvi" - which correction produced integration_adata
    neighbors_within_batch: Optional[int] = None  # the 'Neighbors/Batch' minimum used in the correction that produced this node (mirrors `resolution`)
    child_neighbors_within_batch: Optional[int] = None  # the 'Neighbors/Batch' minimum this node itself last used to correct+split its own children (mirrors `child_resolution`)
    last_stage_status: Optional[str] = None  # short human-readable status shown in the Tree Nodes list, e.g. "BBKNN done 14:02:31"

    def n_cells(self) -> int:
        return 0 if self.adata is None else self.adata.n_obs


class ClusterTreeModel:
    def __init__(self):
        self.nodes: Dict[str, ClusterNode] = {}
        self.root_id: Optional[str] = None
        self.saved_node_ids: List[str] = []

    def clear(self):
        self.nodes.clear()
        self.root_id = None
        self.saved_node_ids.clear()

    def create_root(self, adata, label: str = "root") -> str:
        self.clear()
        node_id = self._new_id()
        self.nodes[node_id] = ClusterNode(node_id=node_id, label=label, adata=adata, depth=0)
        self.root_id = node_id
        return node_id

    def add_child(self, parent_id: str, adata, label: str, cluster_value: str, resolution: Optional[float] = None, neighbors_within_batch: Optional[int] = None) -> str:
        parent = self.nodes[parent_id]
        node_id = self._new_id()
        self.nodes[node_id] = ClusterNode(
            node_id=node_id,
            label=label,
            adata=adata,
            parent_id=parent_id,
            cluster_value=str(cluster_value),
            depth=parent.depth + 1,
            resolution=resolution,
            neighbors_within_batch=neighbors_within_batch,
        )
        parent.children.append(node_id)
        return node_id

    def count_descendants(self, node_id: str) -> int:
        node = self.nodes[node_id]
        count = 0
        stack = list(node.children)
        while stack:
            current_id = stack.pop()
            current = self.nodes.get(current_id)
            if current is None:
                continue
            count += 1
            stack.extend(current.children)
        return count

    def subtree_ids(self, node_id: str) -> set:
        """Return the given node's id plus the ids of all of its descendants."""
        result = {node_id}
        stack = list(self.nodes[node_id].children) if node_id in self.nodes else []
        while stack:
            current_id = stack.pop()
            if current_id in result:
                continue
            result.add(current_id)
            current = self.nodes.get(current_id)
            if current is not None:
                stack.extend(current.children)
        return result

    def remove_children(self, node_id: str):
        node = self.nodes[node_id]
        to_remove = list(node.children)
        while to_remove:
            current_id = to_remove.pop()
            current = self.nodes.get(current_id)
            if current is None:
                continue
            to_remove.extend(current.children)
            if current_id in self.saved_node_ids:
                self.saved_node_ids.remove(current_id)
            del self.nodes[current_id]
        node.children = []

    def mark_saved(self, node_id: str, saved: bool = True):
        node = self.nodes[node_id]
        node.saved = saved
        if saved and node_id not in self.saved_node_ids:
            self.saved_node_ids.append(node_id)
        elif not saved and node_id in self.saved_node_ids:
            self.saved_node_ids.remove(node_id)

    def iter_saved_nodes(self) -> List[ClusterNode]:
        return [self.nodes[nid] for nid in self.saved_node_ids if nid in self.nodes]

    def parent_display_text(self, node_id: str) -> str:
        node = self.nodes[node_id]
        if node.parent_id is None:
            return "Parent group: none (root)"
        parent = self.nodes[node.parent_id]
        parent_cluster = parent.cluster_value if parent.cluster_value is not None else "root"
        return f"Parent group: {parent.label} | parent cluster: {parent_cluster}"

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())[:8]


class H5ADClusterService:
    def __init__(self, batch_key: str = "Batch", cluster_key_prefix: str = "cluster"):
        self.batch_key = batch_key
        self.cluster_key_prefix = cluster_key_prefix
        self.source_h5ad_path: Optional[str] = None  # path of the last h5ad opened via load_h5ad(); used by
        # run_scvi_correction() to save trained models next to the source file

    def load_h5ad(self, path: str, logger: Optional[Callable[[str], None]] = None):
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        if logger:
            logger(f"Loading h5ad file: {path}")
        adata = sc.read_h5ad(path)
        self.source_h5ad_path = path
        if logger:
            logger(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")
            logger(f"adata.raw present: {adata.raw is not None}")
        return adata

    def sanitize_batch_column(self, adata, logger: Optional[Callable[[str], None]] = None):
        if self.batch_key not in adata.obs.columns:
            raise KeyError(f"batch key '{self.batch_key}' was not found in adata.obs")
        raw = adata.obs[self.batch_key]
        missing_before = int(raw.isna().sum()) if hasattr(raw, "isna") else 0
        cleaned = raw.astype("object").where(~raw.isna(), "unknown_batch")
        cleaned = cleaned.map(lambda x: "unknown_batch" if x is None else str(x).strip())
        cleaned = cleaned.replace("", "unknown_batch")
        adata.obs[self.batch_key] = pd.Categorical(cleaned)
        if logger:
            unique_vals = [str(x) for x in adata.obs[self.batch_key].cat.categories.tolist()]
            preview = ", ".join(unique_vals[:10]) if unique_vals else "none"
            logger(f"Sanitized batch column '{self.batch_key}'")
            logger(f"Missing batch values replaced: {missing_before}")
            logger(f"Detected {len(unique_vals)} batch groups; first values: {preview}")
        if len(adata.obs[self.batch_key].cat.categories) < 2:
            raise ValueError(f"batch key '{self.batch_key}' must contain at least 2 groups for BBKNN")

    def filter_small_batches(self, adata, neighbors_within_batch: int, logger: Optional[Callable[[str], None]] = None):
        """BBKNN and scVI both need every batch group to contain enough cells to be usable (BBKNN
        fails outright with 'Not all batches have at least `neighbors_within_batch` cells in them.'
        if not; scVI's batch covariate becomes meaningless for groups this small). Small
        subclusters (especially after a few rounds of reclustering) can easily end up with one or
        more batch/sample groups below that threshold. Rather than letting the correction step
        reject the whole run, drop just those undersized batch groups here, with a clear log entry
        explaining what was excluded and why."""
        counts = adata.obs[self.batch_key].value_counts()
        small_batches = counts[counts < neighbors_within_batch].index.tolist()
        if not small_batches:
            return adata
        before = adata.n_obs
        dropped_cells = int(counts.loc[small_batches].sum())
        keep_mask = ~adata.obs[self.batch_key].isin(small_batches)
        adata = adata[keep_mask].copy()
        adata.obs[self.batch_key] = adata.obs[self.batch_key].cat.remove_unused_categories()
        if logger:
            details = ", ".join(f"{b} ({int(counts[b])} cells)" for b in small_batches)
            logger(
                f"Excluded {len(small_batches)} batch group(s) with fewer than "
                f"{neighbors_within_batch} cells (the 'Neighbors/Batch' minimum): {details}"
            )
            logger(f"Removed {dropped_cells} cells from undersized batches ({before} -> {adata.n_obs} cells remaining)")
        remaining_batches = adata.obs[self.batch_key].cat.categories
        if len(remaining_batches) < 2:
            raise ValueError(
                f"After excluding batch group(s) with fewer than {neighbors_within_batch} cells, only "
                f"{len(remaining_batches)} group(s) remain for batch key '{self.batch_key}' - batch "
                f"correction needs at least 2. Try a lower resolution (fewer, larger subclusters), a "
                f"different/coarser batch key, or lowering the 'Neighbors/Batch' value so fewer groups "
                f"get excluded."
            )
        if adata.n_obs < 2:
            raise ValueError("Not enough cells remain after excluding undersized batches.")
        return adata

    def _describe_counts_matrix(self, x) -> str:
        """Small diagnostic used right before correction starts, so the log makes it visible
        whether the matrix handed to normalize_total/scVI actually looks like raw integer counts
        (min>=0, whole numbers) rather than something already normalized/log-transformed."""
        try:
            data = x.data if (sp is not None and sp.issparse(x)) else np.asarray(x)
            if data.size == 0:
                return "empty matrix"
            sample = data if data.size <= 200000 else data.ravel()[:200000]
            sample_min = float(sample.min())
            sample_max = float(sample.max())
            looks_like_counts = bool(sample_min >= 0 and np.allclose(sample, np.round(sample), atol=1e-6))
            return f"min={sample_min:.4f}, max={sample_max:.4f}, looks_like_raw_integer_counts={looks_like_counts}"
        except Exception as exc:
            return f"could not inspect matrix ({exc})"

    def prepare_counts_for_clustering(self, node: ClusterNode, logger: Optional[Callable[[str], None]] = None):
        source_adata = node.adata
        if "counts" in source_adata.layers:
            adata = source_adata.copy()
            adata.X = adata.layers["counts"].copy()
            if logger:
                logger(
                    f"Using adata.layers['counts'] as the source for reclustering counts "
                    f"(true raw UMI counts) - {self._describe_counts_matrix(adata.X)}"
                )
            return adata

        adata = source_adata.copy()
        if adata.raw is not None and getattr(adata.raw, "X", None) is not None:
            raw_adata = adata.raw.to_adata()
            raw_adata.obs = adata.obs.copy()
            adata = raw_adata
            if logger:
                logger(
                    "No 'counts' layer found; falling back to adata.raw as the source for reclustering "
                    "counts - note adata.raw holds whatever was assigned to it upstream and is not "
                    f"guaranteed to be true raw UMI counts - {self._describe_counts_matrix(adata.X)}"
                )
        else:
            if logger:
                logger("Neither a 'counts' layer nor adata.raw is available; using the current node matrix as-is")
        return adata

    def cluster_node(self, node: ClusterNode, n_pcs: int = 30, resolution: float = 0.8, neighbors_within_batch: int = 3, use_rep: Optional[str] = None, logger: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        if bbknn is None:
            raise ImportError("bbknn is required. Install with: pip install bbknn")
        node_tag = f"[Cluster {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        adata = self.prepare_counts_for_clustering(node, logger=tagged_logger)
        tagged_logger(f"Starting clustering at depth {node.depth} with {adata.n_obs} cells and {adata.n_vars} genes")
        if adata.n_obs < 2:
            raise ValueError("Need at least 2 cells to cluster")

        self.sanitize_batch_column(adata, logger=tagged_logger)
        # Not a QC step - a technical requirement of BBKNN, which errors out if any batch group
        # has fewer than `neighbors_within_batch` cells. Input cells are assumed already QC'd.
        adata = self.filter_small_batches(adata, neighbors_within_batch=neighbors_within_batch, logger=tagged_logger)
        if adata.n_obs < 2:
            raise ValueError("Not enough cells remain after excluding undersized batches")

        tagged_logger("Step: normalize_total (target_sum=10000.0)")
        sc.pp.normalize_total(adata, target_sum=1e4)

        tagged_logger("Step: log1p transform (no additional parameters)")
        sc.pp.log1p(adata)

        hvg_n_top_genes = min(2000, adata.n_vars)
        tagged_logger(f"Step: highly_variable_genes (flavor='seurat', n_top_genes={hvg_n_top_genes})")
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=hvg_n_top_genes)
        if "highly_variable" in adata.var.columns and int(adata.var["highly_variable"].sum()) > 0:
            hvgs = int(adata.var["highly_variable"].sum())
            tagged_logger(f"Subsetting from {adata.n_vars} genes to {hvgs} highly variable genes")
            adata = adata[:, adata.var["highly_variable"]].copy()

        tagged_logger("Step: scale (zero_center=True, max_value=10)")
        sc.pp.scale(adata, max_value=10)

        effective_n_pcs = min(n_pcs, max(2, min(adata.n_obs - 1, adata.n_vars - 1)))
        tagged_logger(f"Step: PCA (svd_solver='arpack', n_comps={effective_n_pcs}, requested n_pcs={n_pcs})")
        sc.tl.pca(adata, svd_solver="arpack", n_comps=effective_n_pcs)
        rep_to_use = "X_pca" if use_rep is None else str(use_rep)
        bbknn_n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
        tagged_logger(
            f"Step: BBKNN batch correction (batch_key='{self.batch_key}', "
            f"neighbors_within_batch={neighbors_within_batch}, n_pcs={bbknn_n_pcs}, use_rep='{rep_to_use}')"
        )
        bbknn.bbknn(adata, batch_key=self.batch_key, neighbors_within_batch=neighbors_within_batch, n_pcs=bbknn_n_pcs, use_rep=rep_to_use)
        tagged_logger("BBKNN completed successfully")
        tagged_logger("Step: UMAP (default parameters)")
        sc.tl.umap(adata)
        cluster_key = f"{self.cluster_key_prefix}_d{node.depth + 1}"
        tagged_logger(
            f"Step: Leiden clustering into obs['{cluster_key}'] (resolution={resolution}, "
            f"flavor='igraph', directed=False, n_iterations=2)"
        )
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key, flavor="igraph", directed=False, n_iterations=2)
        child_map = {}
        raw_clusters = pd.unique(adata.obs[cluster_key])
        unique_clusters = sorted(["" if x is None else str(x) for x in raw_clusters])
        tagged_logger(f"Found {len(unique_clusters)} clusters: {', '.join(unique_clusters) if unique_clusters else 'none'}")
        for cluster_value in unique_clusters:
            mask = adata.obs[cluster_key].astype(str) == cluster_value
            cell_barcodes = adata.obs_names[mask]
            # Subset from the node's own original raw-count matrix (matched by cell barcode),
            # not from the normalized/HVG-subset/scaled pipeline matrix above. This way every
            # child starts from true raw counts, so a future re-cluster of that child redoes
            # normalization/HVG selection/scaling/PCA from scratch instead of compounding
            # transformations on top of already-processed values.
            child_map[cluster_value] = node.adata[cell_barcodes].copy()
            tagged_logger(f"Created child node for cluster {cluster_value} with {child_map[cluster_value].n_obs} cells (raw counts)")
        return {"cluster_key": cluster_key, "clustered_adata": adata, "children": child_map}

    def run_bbknn_correction(self, node: ClusterNode, n_pcs: int = 30, neighbors_within_batch: int = 3, use_rep: Optional[str] = None, logger: Optional[Callable[[str], None]] = None):
        """Preprocess (normalize_total -> log1p -> HVG -> scale -> PCA) then apply BBKNN batch
        correction. Returns an AnnData with a BBKNN neighbor graph, ready for
        run_leiden_clustering. This is the same preprocessing + BBKNN logic cluster_node has
        always used, split out so it can be triggered on its own from the Tree Nodes list."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        if bbknn is None:
            raise ImportError("bbknn is required. Install with: pip install bbknn")
        node_tag = f"[BBKNN {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        adata = self.prepare_counts_for_clustering(node, logger=tagged_logger)
        tagged_logger(f"Starting BBKNN correction at depth {node.depth} with {adata.n_obs} cells and {adata.n_vars} genes")
        if adata.n_obs < 2:
            raise ValueError("Need at least 2 cells to cluster")

        self.sanitize_batch_column(adata, logger=tagged_logger)
        # Not a QC step - a technical requirement of BBKNN, which errors out if any batch group
        # has fewer than `neighbors_within_batch` cells. Input cells are assumed already QC'd.
        adata = self.filter_small_batches(adata, neighbors_within_batch=neighbors_within_batch, logger=tagged_logger)
        if adata.n_obs < 2:
            raise ValueError("Not enough cells remain after excluding undersized batches")

        tagged_logger("Step: normalize_total (target_sum=10000.0)")
        sc.pp.normalize_total(adata, target_sum=1e4)

        tagged_logger("Step: log1p transform (no additional parameters)")
        sc.pp.log1p(adata)

        hvg_n_top_genes = min(2000, adata.n_vars)
        tagged_logger(f"Step: highly_variable_genes (flavor='seurat', n_top_genes={hvg_n_top_genes})")
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=hvg_n_top_genes)
        if "highly_variable" in adata.var.columns and int(adata.var["highly_variable"].sum()) > 0:
            hvgs = int(adata.var["highly_variable"].sum())
            tagged_logger(f"Subsetting from {adata.n_vars} genes to {hvgs} highly variable genes")
            adata = adata[:, adata.var["highly_variable"]].copy()

        tagged_logger("Step: scale (zero_center=True, max_value=10)")
        sc.pp.scale(adata, max_value=10)

        effective_n_pcs = min(n_pcs, max(2, min(adata.n_obs - 1, adata.n_vars - 1)))
        tagged_logger(f"Step: PCA (svd_solver='arpack', n_comps={effective_n_pcs}, requested n_pcs={n_pcs})")
        sc.tl.pca(adata, svd_solver="arpack", n_comps=effective_n_pcs)
        rep_to_use = "X_pca" if use_rep is None else str(use_rep)
        bbknn_n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
        tagged_logger(
            f"Step: BBKNN batch correction (batch_key='{self.batch_key}', "
            f"neighbors_within_batch={neighbors_within_batch}, n_pcs={bbknn_n_pcs}, use_rep='{rep_to_use}')"
        )
        bbknn.bbknn(adata, batch_key=self.batch_key, neighbors_within_batch=neighbors_within_batch, n_pcs=bbknn_n_pcs, use_rep=rep_to_use)
        tagged_logger("BBKNN completed successfully; ready for Leiden clustering")
        return adata

    def run_scvi_correction(self, node: ClusterNode, max_epochs: Optional[int] = None, neighbors_within_batch: int = 25, logger: Optional[Callable[[str], None]] = None):
        """Preprocess raw counts, train an SCVI model to correct for batch effects, and attach the
        resulting latent embedding plus a neighbor graph built on that embedding to a copy of the
        node's data. Returns an AnnData with a neighbor graph on the scVI latent representation,
        ready for run_leiden_clustering - mirrors run_bbknn_correction's contract."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        if scvi is None:
            raise ImportError("scvi-tools is required. Install with: pip install scvi-tools")
        if torch is None:
            raise ImportError("torch is required for SCVI training. Install with: pip install torch")
        node_tag = f"[SCVI {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        adata = self.prepare_counts_for_clustering(node, logger=tagged_logger)
        tagged_logger(f"Starting SCVI correction at depth {node.depth} with {adata.n_obs} cells and {adata.n_vars} genes")
        if adata.n_obs < 2:
            raise ValueError("Need at least 2 cells to cluster")

        self.sanitize_batch_column(adata, logger=tagged_logger)

        # SCVI models raw counts directly, so stash the (as-yet untouched) counts matrix in its
        # own layer before anything below normalizes or subsets adata.X.
        tagged_logger("Stashing raw counts into adata.layers['counts'] for SCVI")
        adata.layers["counts"] = adata.X.copy()

        # Normalize + log1p per batch group (mirrors the reference script's per-donor loop),
        # grouping by self.batch_key rather than a hardcoded 'donor' column so this stays in sync
        # with whatever batch key the rest of the pipeline (filtering, scVI's covariate) uses.
        # normalize_total/log1p are per-cell operations, so this produces the same values as doing
        # it in one call on the whole object - the loop is kept only to mirror the reference exactly.
        group_values = adata.obs[self.batch_key].unique().tolist()
        adata_list = []
        for group in group_values:
            tagged_logger(f"Step: normalize_total (target_sum=10000.0) + log1p for batch group '{group}'")
            subset_adata = adata[adata.obs[self.batch_key] == group].copy()
            subset_adata.X = subset_adata.layers["counts"].copy()
            sc.pp.normalize_total(subset_adata, target_sum=1e4)  # scale each cell to a common library size
            sc.pp.log1p(subset_adata)  # log(expression + 1)
            adata_list.append(subset_adata)
        adata = sc.concat(adata_list, merge="first", join="outer")
        del adata_list

        # Drop undersized batch groups after normalization (same order as the reference script),
        # using the same helper run_bbknn_correction relies on so both paths stay consistent.
        adata = self.filter_small_batches(adata, neighbors_within_batch=neighbors_within_batch, logger=tagged_logger)
        if adata.n_obs < 2:
            raise ValueError("Not enough cells remain after excluding undersized batches")

        # flag (but do not subset) highly variable genes on the object handed in; whatever genes
        # `adata` already has are retained untouched - this only decides which subset SCVI itself
        # trains on. Matches process_scvi() exactly: default flavor on the already-normalized
        # `.X` (not seurat_v3/raw counts).
        hvg_n_top_genes = min(1200, adata.n_vars)
        tagged_logger(f"Step: highly_variable_genes (n_top_genes={hvg_n_top_genes}, batch_key='{self.batch_key}')")

        sc.pp.highly_variable_genes(
                adata,
                n_top_genes=hvg_n_top_genes,
                subset=False,
                batch_key=self.batch_key,
            )

        # train SCVI on a throwaway copy restricted to these HVGs only, to keep the network/training
        # cost down; the shared `adata` used for everything else (clustering, CellTypist, etc.) keeps
        # every gene it came in with. Cell order/count here matches `adata` exactly, since scvi_adata
        # is a straight per-cell copy of it (only genes get subset below).
        scvi_adata = adata[:, adata.var["highly_variable"]].copy()
        tagged_logger(f"Training SCVI on {scvi_adata.n_obs} cells x {scvi_adata.n_vars} HVGs")

        # batch_key is what tells scVI which column to actually correct for: it conditions both
        # the encoder and decoder on this covariate, so the model learns to push batch identity
        # into the part it reconstructs FROM and keep it out of the latent embedding. This is
        # NOT the same as categorical_covariate_keys, which is for secondary nuisance covariates
        # layered on top of an already-specified batch_key via a separate, weaker embedding
        # pathway - using it here instead of batch_key would leave scVI never properly told what
        # to integrate over.
        tagged_logger(f"Step: SCVI.setup_anndata(batch_key='{self.batch_key}')")
        scvi.model.SCVI.setup_anndata(
            scvi_adata,
            layer="counts",
            batch_key=self.batch_key,
        )

        model = scvi.model.SCVI(scvi_adata)

        # ---- accelerator selection: GPU first, then CPU, otherwise train unaccelerated ----
        if torch.cuda.is_available():
            accelerator = "gpu"
            tagged_logger(f"GPU detected ({torch.cuda.device_count()} device(s)); training with accelerator='gpu'")
        elif (os.cpu_count() or 0) > 0:
            accelerator = "cpu"
            tagged_logger(f"No GPU detected; {os.cpu_count()} CPU core(s) available, training with accelerator='cpu'")
        else:
            accelerator = None
            tagged_logger("No GPU or CPU resources detected; training without an explicit accelerator")

        train_kwargs = {} if max_epochs is None else {"max_epochs": max_epochs}
        if accelerator:
            tagged_logger(f"Step: model.train(accelerator='{accelerator}')")
            model.train(accelerator=accelerator, **train_kwargs)
        else:
            tagged_logger("Step: model.train() (no accelerator argument)")
            model.train(**train_kwargs)

        # ---- save the trained model next to the source h5ad, falling back to the working directory ----
        h5ad_dir = None
        source_path = getattr(self, "source_h5ad_path", None)
        if source_path:
            candidate_dir = os.path.dirname(os.path.abspath(source_path))
            if candidate_dir and os.path.isdir(candidate_dir):
                h5ad_dir = candidate_dir
        save_dir = h5ad_dir if h5ad_dir else os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(save_dir, f"{node.node_id}_scvi_model_{timestamp}")
        tagged_logger(f"Step: saving trained SCVI model to '{model_dir}'")
        model.save(model_dir, overwrite=True)

        latent_key = "X_scVI"
        # obsm is indexed per-cell only, so the latent embedding attaches directly to the full,
        # un-subsetted object; row (cell) order is unaffected by the gene/column subsetting used to
        # train the model.
        tagged_logger("Step: extracting latent representation")
        latent = model.get_latent_representation()
        adata.obsm[latent_key] = latent

        tagged_logger(f"Step: neighbors graph on '{latent_key}' (use_rep='{latent_key}')")
        sc.pp.neighbors(adata, use_rep=latent_key)

        tagged_logger("SCVI completed successfully; ready for Leiden clustering")
        return adata

    def _relabel_clusters_by_descending_size(self, adata, cluster_key: str, logger: Optional[Callable[[str], None]] = None) -> List[str]:
        """Renumber the Leiden cluster codes in adata.obs[cluster_key] so that '0' is the
        largest cluster, '1' the second largest, and so on, purely by cell count - the actual
        grouping of cells is untouched, only the integer label each group is known by changes.
        Returns the new labels in size order (['0', '1', ...]), which is also their natural
        sort order, so every downstream consumer of this column (UMAP coloring, dot plots, the
        stacked bar chart, child node creation) shows/creates cluster 0 as the biggest without
        needing any special-case sorting of its own."""
        size_by_label = adata.obs[cluster_key].astype(str).value_counts()  # already sorted descending by count
        old_labels_by_size = size_by_label.index.tolist()
        remap = {old_label: str(new_index) for new_index, old_label in enumerate(old_labels_by_size)}
        if logger:
            summary = ", ".join(
                f"'{old}' ({size_by_label[old]} cells) -> '{new}'" for old, new in remap.items()
            )
            logger(f"Renumbering clusters by descending size so '0' is the biggest: {summary}")
        new_labels_in_size_order = [remap[old] for old in old_labels_by_size]
        adata.obs[cluster_key] = pd.Categorical(
            adata.obs[cluster_key].astype(str).map(remap), categories=new_labels_in_size_order, ordered=True
        )
        return new_labels_in_size_order

    def run_leiden_clustering(self, node: ClusterNode, integrated_adata, resolution: float = 0.8, umap_min_dist: float = 0.5, umap_spread: float = 1.0, logger: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
        """Run UMAP + Leiden clustering on an already batch-corrected AnnData (the result of
        run_bbknn_correction or run_scvi_correction) and partition the node's cells into child
        nodes by cluster. This is the tail end of the original cluster_node pipeline, split out
        so it can run on its own once BBKNN or SCVI correction has already produced a neighbor
        graph. umap_min_dist/umap_spread default to scanpy's own UMAP defaults (0.5/1.0); pass
        tighter values (e.g. 0.1/1.0, as used for the scVI path) for more visually separated
        clusters."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        node_tag = f"[Leiden {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        adata = integrated_adata
        tagged_logger(f"Step: UMAP (min_dist={umap_min_dist}, spread={umap_spread})")
        sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)
        cluster_key = f"{self.cluster_key_prefix}_d{node.depth + 1}"
        tagged_logger(
            f"Step: Leiden clustering into obs['{cluster_key}'] (resolution={resolution}, "
            f"flavor='igraph', directed=False, n_iterations=2)"
        )
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key, flavor="igraph", directed=False, n_iterations=2)
        # Leiden's own cluster numbers are arbitrary (not size-ordered) - relabel them so '0' is
        # always the biggest cluster, matching what the user expects to see in the Tree Nodes
        # list, the UMAP legend, and the stacked bar chart.
        unique_clusters = self._relabel_clusters_by_descending_size(adata, cluster_key, logger=tagged_logger)
        tagged_logger(f"Found {len(unique_clusters)} clusters (0=largest): {', '.join(unique_clusters) if unique_clusters else 'none'}")
        child_map = {}
        for cluster_value in unique_clusters:
            mask = adata.obs[cluster_key].astype(str) == cluster_value
            cell_barcodes = adata.obs_names[mask]
            # Subset from the node's own original raw-count matrix (matched by cell barcode), not
            # from the normalized/HVG-subset/scaled/integrated matrix above - so every child
            # starts from true raw counts and a future correction+clustering of that child redoes
            # normalization/HVG/scale/integration from scratch instead of compounding transforms.
            child_map[cluster_value] = node.adata[cell_barcodes].copy()
            tagged_logger(f"Created child node for cluster {cluster_value} with {child_map[cluster_value].n_obs} cells (raw counts)")
        return {"cluster_key": cluster_key, "clustered_adata": adata, "children": child_map}

    def run_resolution_sweep(self, node: ClusterNode, graph_adata, resolutions: List[float], logger: Optional[Callable[[str], None]] = None) -> List[Dict[str, object]]:
        """Try several Leiden resolutions on an AnnData that already has a neighbor graph
        (either node.integration_adata, fresh off a correction, or node.umap_adata from an
        earlier correction+clustering run - both carry the neighbor graph a resolution sweep
        needs; only the resolution changes, not the correction). For each resolution, records
        how many clusters it produced and how similar that partition is to the previous
        resolution's, via the Adjusted Rand Index (1.0 = identical partitions, ~0.0 = no more
        agreement than random). A stable stretch of resolutions - cluster count unchanged, ARI
        close to 1 - is a plateau: a sign that split reflects real structure in the graph rather
        than an arbitrary artifact of exactly where the resolution knob happened to sit. This
        never touches node.last_cluster_key/node.umap_adata/node.children - it's a read-only
        diagnostic; use the toolbar's Resolution field + Process button to actually commit to a
        chosen resolution afterwards."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        try:
            from sklearn.metrics import adjusted_rand_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for the resolution sweep's Adjusted Rand Index. "
                "Install with: pip install scikit-learn"
            )
        node_tag = f"[Sweep {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        results: List[Dict[str, object]] = []
        previous_labels = None
        temp_keys = []
        try:
            for resolution in resolutions:
                sweep_key = f"__resolution_sweep_{resolution}"
                temp_keys.append(sweep_key)
                tagged_logger(f"Running Leiden at resolution={resolution}...")
                sc.tl.leiden(graph_adata, resolution=resolution, key_added=sweep_key, flavor="igraph", directed=False, n_iterations=2)
                labels = graph_adata.obs[sweep_key].astype(str)
                n_clusters = int(labels.nunique())
                if previous_labels is not None:
                    ari_vs_previous = float(adjusted_rand_score(previous_labels.values, labels.values))
                    tagged_logger(f"resolution={resolution}: {n_clusters} clusters, ARI vs previous resolution={ari_vs_previous:.3f}")
                else:
                    ari_vs_previous = None
                    tagged_logger(f"resolution={resolution}: {n_clusters} clusters")
                results.append({"resolution": resolution, "n_clusters": n_clusters, "ari_vs_previous": ari_vs_previous})
                previous_labels = labels
        finally:
            # Sweep columns are scratch space only - clean them up so they don't linger in
            # .obs and collide with the real cluster_key naming scheme (cluster_d1, cluster_d2...).
            for key in temp_keys:
                if key in graph_adata.obs.columns:
                    del graph_adata.obs[key]
        return results

    def compute_adhoc_leiden_labels(self, node: ClusterNode, resolution: float, logger: Optional[Callable[[str], None]] = None):
        """Run Leiden once at an explicit resolution, reusing the neighbor graph already sitting
        on node.umap_adata (the same graph its real, committed clustering used - re-cutting it
        at a different resolution needs no recomputed neighbors or UMAP). Used by the "New UMAP
        at Resolution..." / "New Stacked Bar at Resolution..." buttons for a one-off exploratory
        view; this is read-only/non-destructive like run_resolution_sweep - a temporary obs
        column is added and cleaned up, and node.umap_adata/last_cluster_key are never touched.
        Clusters are relabeled by descending size (0=largest), same convention as a real
        Process run, so numbers here mean the same thing they would if this resolution were
        actually committed. Returns a pandas Series of string labels indexed by cell barcode."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        graph_adata = node.umap_adata
        if graph_adata is None or "neighbors" not in graph_adata.uns:
            raise ValueError(
                "This node needs to be clustered at least once (Process) before previewing "
                "another resolution."
            )
        node_tag = f"[Adhoc resolution {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        temp_key = "__adhoc_resolution_view"
        try:
            tagged_logger(f"Running Leiden at resolution={resolution} on the existing neighbor graph...")
            sc.tl.leiden(graph_adata, resolution=resolution, key_added=temp_key, flavor="igraph", directed=False, n_iterations=2)
            labels = graph_adata.obs[temp_key].astype(str).copy()
        finally:
            if temp_key in graph_adata.obs.columns:
                del graph_adata.obs[temp_key]
        size_by_label = labels.value_counts()  # already sorted descending by count
        remap = {old_label: str(new_index) for new_index, old_label in enumerate(size_by_label.index)}
        labels = labels.map(remap)
        tagged_logger(f"resolution={resolution}: {labels.nunique()} clusters (0=largest)")
        return labels

    def _compute_batch_mixing_metrics(self, distances, batch_labels: np.ndarray, logger: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
        """Shared by both neighbors sweeps. For every cell, looks at its actual graph neighbors
        (from `distances`, a cells x cells sparse matrix where nonzero entries are that cell's
        neighbors) and computes two batch-mixing measures:

          - cross_batch_fraction: the simple fraction of a cell's neighbors that come from a
            DIFFERENT batch than the cell itself. Easy to read, but blind to HOW those
            cross-batch neighbors are spread out - a cell whose neighbors are 100% from just one
            other batch (ignoring every other batch that exists) scores identically to a cell
            whose neighbors are evenly spread across every other batch, even though the second
            case is the actually well-mixed one.
          - effective_batches: the inverse Simpson index over each cell's neighbor batch
            proportions (1 / sum(p_b^2), the same "effective number of categories" measure used
            by LISI/iLISI for exactly this reason) - it distinguishes those two cases. A cell
            whose neighbors are one single batch (own or otherwise) scores 1.0 regardless of
            cross_batch_fraction; a cell whose neighbors are spread evenly across N batches
            scores close to N. Averaged across all cells, then compared against
            n_batches_total (how many batches exist in this data) to say e.g. "neighborhoods
            behave like 6.2 out of 10 batches, on average" - much closer to what "well mixed"
            actually means than the fraction alone.

        Returns a dict with mean_cross_batch_fraction, mean_effective_batches, n_batches_total.
        """
        n_unique_batches = len(set(batch_labels))
        if n_unique_batches < 2 and logger:
            logger(
                f"WARNING: the batch column has only {n_unique_batches} unique value(s) in this "
                f"data - both mixing metrics will be stuck at their minimum regardless of the "
                f"neighbor parameter, which almost certainly means the wrong batch key is set "
                f"(check the 'Batch Key' box), not that batches failed to mix."
            )
        n_cells = len(batch_labels)
        cross_batch_fractions = np.full(n_cells, np.nan)
        effective_batches = np.full(n_cells, np.nan)
        for i in range(n_cells):
            neighbor_idx = distances.getrow(i).indices
            if len(neighbor_idx) == 0:
                continue
            neighbor_batches = batch_labels[neighbor_idx]
            cross_batch_fractions[i] = np.mean(neighbor_batches != batch_labels[i])
            _, counts = np.unique(neighbor_batches, return_counts=True)
            proportions = counts / counts.sum()
            effective_batches[i] = 1.0 / np.sum(proportions ** 2)
        return {
            "mean_cross_batch_fraction": float(np.nanmean(cross_batch_fractions)),
            "mean_effective_batches": float(np.nanmean(effective_batches)),
            "n_batches_total": n_unique_batches,
        }

    def run_bbknn_neighbors_sweep(self, node: ClusterNode, neighbor_values: List[int], resolution: float, n_pcs: int = 30, use_rep: Optional[str] = None, logger: Optional[Callable[[str], None]] = None) -> List[Dict[str, object]]:
        """Try several BBKNN `neighbors_within_batch` values on this node. Unlike
        run_resolution_sweep (which reuses one already-built graph), this REBUILDS the whole
        graph from scratch for every value - by calling run_bbknn_correction() itself once per
        value - because neighbors_within_batch changes the graph, not just how it's cut.
        (filter_small_batches, which depends on neighbors_within_batch, runs before
        normalize/HVG/PCA inside run_bbknn_correction, so even the preprocessing genuinely
        differs per value; nothing upstream of BBKNN can safely be shared across neighbor
        values here the way UMAP+Leiden can be shared across resolutions.) This reuses the exact
        same run_bbknn_correction() used by the real Process button, so results here are
        guaranteed consistent with what a real Process click would produce for the same value.

        For each neighbor value, reports:
          - mean_cross_batch_fraction / mean_effective_batches / n_batches_total: see
            _compute_batch_mixing_metrics() for what these mean and how they differ.
          - n_connected_components: 1 means the neighbor graph is a single connected whole; >1
            means the graph is fragmented - one or more batches got barely/never linked to the
            rest, which will quietly break clustering and UMAP regardless of resolution.
          - n_clusters: Leiden cluster count at the given FIXED resolution, so cluster counts
            are comparable across neighbor values instead of also varying with resolution.
          - ari_vs_previous: Adjusted Rand Index between this neighbor value's clustering (at
            the fixed resolution) and the previous neighbor value's - high means the final
            clustering answer is not sensitive to exactly which neighbor value you picked.
        This never modifies node.integration_adata/node.umap_adata/node.children - run_bbknn_
        correction() returns a fresh AnnData each call and nothing here writes it back onto the
        node; it's a read-only diagnostic same as run_resolution_sweep."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        if bbknn is None:
            raise ImportError("bbknn is required. Install with: pip install bbknn")
        try:
            from sklearn.metrics import adjusted_rand_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for the neighbors sweep's Adjusted Rand Index. "
                "Install with: pip install scikit-learn"
            )
        try:
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            raise ImportError(
                "scipy is required for the neighbors sweep's connected-components check. "
                "Install with: pip install scipy"
            )
        node_tag = f"[BBKNN neighbors sweep {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        results: List[Dict[str, object]] = []
        previous_labels = None
        for neighbors_within_batch in neighbor_values:
            tagged_logger(f"=== Rebuilding BBKNN graph with neighbors_within_batch={neighbors_within_batch} ===")
            try:
                corrected = self.run_bbknn_correction(
                    node, n_pcs=n_pcs, neighbors_within_batch=neighbors_within_batch, use_rep=use_rep, logger=logger,
                )
            except ValueError as exc:
                tagged_logger(f"Skipped neighbors_within_batch={neighbors_within_batch}: {exc}")
                results.append({
                    "neighbors_within_batch": neighbors_within_batch, "mean_cross_batch_fraction": None,
                    "mean_effective_batches": None, "n_batches_total": None,
                    "n_connected_components": None, "n_clusters": None, "ari_vs_previous": None,
                    "skipped_reason": str(exc),
                })
                continue

            batch_labels = corrected.obs[self.batch_key].astype(str).values
            mixing = self._compute_batch_mixing_metrics(corrected.obsp["distances"], batch_labels, logger=tagged_logger)
            n_components, _ = connected_components(corrected.obsp["connectivities"], directed=False)

            sweep_key = f"__neighbors_sweep_{neighbors_within_batch}"
            tagged_logger(f"Running Leiden at fixed resolution={resolution} to compare clustering across neighbor values...")
            sc.tl.leiden(corrected, resolution=resolution, key_added=sweep_key, flavor="igraph", directed=False, n_iterations=2)
            labels = corrected.obs[sweep_key].astype(str)
            n_clusters = int(labels.nunique())

            ari_vs_previous = None
            if previous_labels is not None:
                # Different neighbor values can end up excluding different undersized batches
                # (filter_small_batches depends on neighbors_within_batch), so the exact set of
                # cells present can differ between two values - compare only cells present in
                # both rather than assuming the same cells survived both runs.
                common = previous_labels.index.intersection(labels.index)
                if len(common) > 1:
                    ari_vs_previous = float(adjusted_rand_score(previous_labels.loc[common].values, labels.loc[common].values))

            tagged_logger(
                f"neighbors_within_batch={neighbors_within_batch}: mean cross-batch neighbor fraction="
                f"{mixing['mean_cross_batch_fraction']:.3f}, effective batches in neighborhood="
                f"{mixing['mean_effective_batches']:.2f} / {mixing['n_batches_total']} total, "
                f"connected components={n_components}, clusters at resolution={resolution}: {n_clusters}"
                + (f", ARI vs previous value={ari_vs_previous:.3f}" if ari_vs_previous is not None else "")
            )
            results.append({
                "neighbors_within_batch": neighbors_within_batch,
                "mean_cross_batch_fraction": mixing["mean_cross_batch_fraction"],
                "mean_effective_batches": mixing["mean_effective_batches"],
                "n_batches_total": mixing["n_batches_total"],
                "n_connected_components": int(n_components),
                "n_clusters": n_clusters,
                "ari_vs_previous": ari_vs_previous,
                "skipped_reason": None,
            })
            previous_labels = labels

        return results

    def run_scvi_neighbors_sweep(self, node: ClusterNode, graph_adata, neighbor_values: List[int], resolution: float, logger: Optional[Callable[[str], None]] = None) -> List[Dict[str, object]]:
        """scVI counterpart to run_bbknn_neighbors_sweep - try several `n_neighbors` values on an
        AnnData that already carries a trained scVI latent embedding (obsm['X_scVI'], produced
        by run_scvi_correction). Unlike the BBKNN sweep, this does NOT retrain anything per
        value: scVI training is the expensive step and n_neighbors has no effect on the trained
        latent space, so each value only rebuilds the neighbor graph itself
        (sc.pp.neighbors(..., use_rep='X_scVI', n_neighbors=...)) on a fresh copy of graph_adata,
        leaving the original untouched.

        Reports the same diagnostics as the BBKNN sweep, but the cross-batch mixing fraction
        means something importantly different here: BBKNN forces a fixed number of neighbors
        from every batch group by construction, so its mixing fraction mostly just reflects how
        many batches you have, regardless of the value chosen (see run_bbknn_neighbors_sweep).
        scVI's neighbor graph is an ordinary nearest-neighbor search in latent space with no such
        quota, so a high mixing fraction here is a genuine, unforced signal that scVI actually
        learned to place cells from different batches near each other - and a low one is a real
        red flag that the batch effect wasn't removed, not just an artifact of the graph-building
        method. n_connected_components and the resolution-fixed cluster count / ARI-vs-previous
        mean the same thing as in the BBKNN sweep. Read-only: never modifies node state or
        graph_adata itself."""
        if sc is None:
            raise ImportError("scanpy is required. Install with: pip install scanpy")
        if "X_scVI" not in graph_adata.obsm:
            raise ValueError(
                "This AnnData has no 'X_scVI' latent embedding yet - run Batch Correction with "
                "'SCVI' selected on this node first, then try the sweep again."
            )
        try:
            from sklearn.metrics import adjusted_rand_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for the neighbors sweep's Adjusted Rand Index. "
                "Install with: pip install scikit-learn"
            )
        try:
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            raise ImportError(
                "scipy is required for the neighbors sweep's connected-components check. "
                "Install with: pip install scipy"
            )
        node_tag = f"[SCVI neighbors sweep {node.label} | id {node.node_id}]"

        def tagged_logger(message: str) -> None:
            if logger:
                logger(f"{node_tag} {message}")

        results: List[Dict[str, object]] = []
        previous_labels = None
        for n_neighbors in neighbor_values:
            tagged_logger(f"=== Rebuilding neighbor graph on X_scVI with n_neighbors={n_neighbors} ===")
            working = graph_adata.copy()
            sc.pp.neighbors(working, use_rep="X_scVI", n_neighbors=n_neighbors)

            # Batch-mixing scores: see _compute_batch_mixing_metrics() - a genuine signal here
            # (unlike the same metrics on a BBKNN graph, see docstring above).
            batch_labels = working.obs[self.batch_key].astype(str).values
            mixing = self._compute_batch_mixing_metrics(working.obsp["distances"], batch_labels, logger=tagged_logger)

            n_components, _ = connected_components(working.obsp["connectivities"], directed=False)

            sweep_key = f"__scvi_neighbors_sweep_{n_neighbors}"
            tagged_logger(f"Running Leiden at fixed resolution={resolution} to compare clustering across n_neighbors values...")
            sc.tl.leiden(working, resolution=resolution, key_added=sweep_key, flavor="igraph", directed=False, n_iterations=2)
            labels = working.obs[sweep_key].astype(str)
            n_clusters = int(labels.nunique())

            ari_vs_previous = None
            if previous_labels is not None:
                common = previous_labels.index.intersection(labels.index)
                if len(common) > 1:
                    ari_vs_previous = float(adjusted_rand_score(previous_labels.loc[common].values, labels.loc[common].values))

            tagged_logger(
                f"n_neighbors={n_neighbors}: mean cross-batch neighbor fraction="
                f"{mixing['mean_cross_batch_fraction']:.3f}, effective batches in neighborhood="
                f"{mixing['mean_effective_batches']:.2f} / {mixing['n_batches_total']} total, "
                f"connected components={n_components}, clusters at resolution={resolution}: {n_clusters}"
                + (f", ARI vs previous value={ari_vs_previous:.3f}" if ari_vs_previous is not None else "")
            )
            results.append({
                "n_neighbors": n_neighbors,
                "mean_cross_batch_fraction": mixing["mean_cross_batch_fraction"],
                "mean_effective_batches": mixing["mean_effective_batches"],
                "n_batches_total": mixing["n_batches_total"],
                "n_connected_components": int(n_components),
                "n_clusters": n_clusters,
                "ari_vs_previous": ari_vs_previous,
                "skipped_reason": None,
            })
            previous_labels = labels

        return results

    def export_saved_nodes(self, nodes: List[ClusterNode], output_csv: str, logger: Optional[Callable[[str], None]] = None):
        if not nodes:
            raise ValueError("No saved nodes to export")
        if logger:
            logger(f"Exporting {len(nodes)} saved nodes to CSV")
        frames = []
        for node in nodes:
            obs = node.adata.obs.copy()
            obs.insert(0, "cell_id", obs.index.astype(str))
            obs["tree_node_id"] = node.node_id
            obs["tree_node_label"] = node.label
            obs["tree_depth"] = node.depth
            obs["selected_cluster"] = node.cluster_value if node.cluster_value is not None else "root"
            frames.append(obs)
            if logger:
                logger(f"Added node {node.label} with {len(obs)} rows to export")
        merged = pd.concat(frames, axis=0)
        merged.to_csv(output_csv, index=False)
        if logger:
            logger(f"CSV written to {output_csv} with {len(merged)} rows")
        return merged


class ClusterTreeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("H5AD Cluster Tree Explorer")
        self.geometry("1650x980")
        self.configure(bg=BG_MAIN)
        self.model = ClusterTreeModel()
        self.service = H5ADClusterService()
        self.selected_node_id: Optional[str] = None
        self.busy = False
        self.umap_canvas = None
        self.umap_figure = None
        # Each loaded markers CSV becomes its own entry here (in load order = stacking order):
        # {"id": int, "path": str, "name": str, "marker_dict": Dict[str, List[str]]}. Re-loading
        # the same path updates that entry in place rather than adding a duplicate stacked plot.
        self.marker_file_sets: List[Dict[str, object]] = []
        self._next_marker_set_id = 1
        self.stackedbar_canvas = None
        self.stackedbar_figure = None
        self.celltypist_model = None
        self.celltypist_model_path: Optional[str] = None
        self.available_cell_types: set = set()  # union of predicted cell types seen across all nodes' stacked bars
        self._row_style_cache: Dict[tuple, tuple] = {}
        # Shared between the header row and every data row (both built in refresh_node_list) so
        # the header's columns always line up with the rows underneath, including while
        # scrolling horizontally - the header is part of the same scrollable content, not a
        # separate fixed row above it.
        self._node_col_widths = {
            "name": 8,
            "cells": 5,
            "resolution": 5,
            "neighbors": 5,
            "cell_type": 24,
            "action": 10,
        }
        self._configure_theme()
        self._build_ui()
        self._build_log_window()
        self._load_default_celltypist_model()

    def _configure_theme(self):
        style = ttk.Style(self)
        self.style = style
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background=BG_MAIN)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("PlotArea.TFrame", background="white")
        base_font_size = tkfont.nametofont("TkDefaultFont").cget("size")
        self._base_font_size = base_font_size
        style.configure("Card.TFrame", background=BG_PANEL_ALT, relief="solid", borderwidth=1)
        style.configure("CardRow.TFrame", background=BG_PANEL_ALT, relief="flat", borderwidth=0)
        style.configure("App.TLabel", background=BG_MAIN, foreground=FG_MAIN)
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_MAIN)
        style.configure("Card.TLabel", background=BG_PANEL_ALT, foreground=FG_MAIN)
        style.configure("Card.TCheckbutton", background=BG_PANEL_ALT, foreground=FG_MAIN)
        style.map("Card.TCheckbutton", background=[("active", BG_PANEL_ALT)])
        style.configure("NodeTitle.TLabel", background=BG_PANEL_ALT, foreground=FG_MAIN, font=("TkDefaultFont", base_font_size, "bold"))
        # Smaller font for the Tree Nodes list header only (kept separate from Panel.TLabel,
        # which is shared by many other panels/popups that shouldn't shrink along with this).
        style.configure("TreeNodesHeader.TLabel", background=BG_PANEL, foreground=FG_MAIN, font=("TkDefaultFont", 9))
        style.configure("TreeNodesRow.TCombobox", font=("TkDefaultFont", 9))
        style.configure("TreeNodesRow.TButton", font=("TkDefaultFont", 9))
        style.configure("TButton", background=BG_PANEL_ALT, foreground=FG_MAIN, bordercolor=BORDER, focusthickness=1, focuscolor=ACCENT)
        style.map("TButton", background=[("active", BG_SELECTED)], foreground=[("active", FG_MAIN)])
        style.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG_MAIN, insertcolor=FG_MAIN)
        style.configure("Treeview", background=BG_INPUT, fieldbackground=BG_INPUT, foreground=FG_MAIN, bordercolor=BORDER)
        style.map("Treeview", background=[("selected", BG_SELECTED)], foreground=[("selected", FG_MAIN)])
        style.configure("Treeview.Heading", background=BG_PANEL_ALT, foreground=FG_MAIN)
        style.configure("Vertical.TScrollbar", background=BG_PANEL_ALT, troughcolor=BG_MAIN, arrowcolor=FG_MAIN)
        style.configure("Horizontal.TScrollbar", background=BG_PANEL_ALT, troughcolor=BG_MAIN, arrowcolor=FG_MAIN)

    def _add_menu_command(self, menu: tk.Menu, label: str, command):
        """Add a command to a menu and remember its (menu, index) so set_busy can still
        disable/enable it, now that most actions live in menus instead of standalone buttons."""
        menu.add_command(label=label, command=command)
        self._busy_menu_items.append((menu, menu.index("end")))

    def _build_menu_bar(self):
        self._busy_menu_items: List[tuple] = []
        menubar = tk.Menu(self, tearoff=False)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        self._add_menu_command(file_menu, "Load H5AD...", self.load_h5ad_file)
        file_menu.add_separator()
        self._add_menu_command(file_menu, "Save Session...", self.save_session)
        self._add_menu_command(file_menu, "Load Session...", self.load_session)
        file_menu.add_separator()
        self._add_menu_command(file_menu, "Export Selected CSV...", self.export_selected_nodes)

        annotation_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Annotation", menu=annotation_menu)
        self._add_menu_command(annotation_menu, "Load CellTypist Model...", self.load_celltypist_model)
        self._add_menu_command(annotation_menu, "Load Cell Types", self.load_cell_types_for_annotation)
        self._add_menu_command(annotation_menu, "Load Cell Types CSV...", self.load_cell_types_csv)
        self._add_menu_command(annotation_menu, "Add Custom Cell Type...", self.add_custom_cell_type)
        self._add_menu_command(annotation_menu, "Load Markers CSV...", self.load_markers_csv)
        self._add_menu_command(annotation_menu, "Clear Marker Plots", self.clear_marker_plots)
        annotation_menu.add_separator()
        self._add_menu_command(annotation_menu, "Save Cell Type Annotations...", self.save_cell_type_annotations)
        self._add_menu_command(annotation_menu, "Save Cell Type Annotations as CSV...", self.save_cell_type_annotations_csv)
        self._add_menu_command(annotation_menu, "Visualize UMAP by Annotation", self.visualize_umap_by_annotation)

        tools_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        self._add_menu_command(tools_menu, "Show Log Window", self.show_log_window)
        tools_menu.add_separator()
        self._add_menu_command(tools_menu, "Resolution Sweep...", self.open_resolution_sweep_dialog)
        self._add_menu_command(tools_menu, "BBKNN Neighbors Sweep...", self.open_bbknn_neighbors_sweep_dialog)
        self._add_menu_command(tools_menu, "SCVI Neighbors Sweep...", self.open_scvi_neighbors_sweep_dialog)

    def _get_magnifying_glass_icon(self):
        """A small magnifying-glass icon (drawn with PIL, not an emoji glyph, so it renders
        consistently regardless of the platform's font/emoji support) for the plot panels' Open
        buttons. Drawn once and cached - Tk requires keeping a reference to a PhotoImage or it
        gets garbage-collected and the button goes blank."""
        cached = getattr(self, "_magnifying_glass_icon", None)
        if cached is not None:
            return cached
        if Image is None or ImageDraw is None or ImageTk is None:
            return None
        size = 16
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        lens_bbox = (1, 1, 10, 10)
        draw.ellipse(lens_bbox, outline=FG_MAIN, width=2)
        draw.line((9, 9, 14, 14), fill=FG_MAIN, width=2)
        icon = ImageTk.PhotoImage(img)
        self._magnifying_glass_icon = icon
        return icon

    def _qualitative_colors(self, n: int) -> List[tuple]:
        """Build a list of n visually distinct RGBA colors for categorical data (predicted cell
        types, UMAP annotation categories). A single matplotlib qualitative palette like tab20
        only has 20 distinct colors, so once a plot has more than 20 categories (e.g. up to ~50
        CellTypist labels) it starts silently repeating colors and different cell types become
        indistinguishable. To avoid that, this concatenates matplotlib's three curated 20-color
        qualitative palettes - tab20, tab20b, tab20c - giving 60 distinct colors before any
        repeats are needed at all. If more than 60 categories ever show up, the remainder are
        generated from an evenly spaced HSV color wheel (alternating lightness between passes
        so neighboring hues stay distinguishable) rather than silently reusing colors."""
        palette: List[tuple] = []
        for cmap_name in ("tab20", "tab20b", "tab20c"):
            cmap = plt.get_cmap(cmap_name)
            palette.extend(cmap(i) for i in range(cmap.N))
        if n <= len(palette):
            return palette[:n]
        extra_needed = n - len(palette)
        for i in range(extra_needed):
            hue = (i / extra_needed) % 1.0
            lightness = 0.45 if i % 2 == 0 else 0.65
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.65)
            palette.append((r, g, b, 1.0))
        return palette[:n]

    def _attach_plot_toolbar(self, canvas, parent):
        """Attach matplotlib's standard interactive toolbar below an embedded plot: a rubber-band
        zoom-to-rectangle tool, a pan (click-and-drag) tool, back/forward view history, and a
        'home' button to reset to the original view. This is the standard, built-in way to add
        zoom/pan to a matplotlib figure embedded in Tkinter - no custom event handling needed."""
        if NavigationToolbar2Tk is None:
            return None
        toolbar_frame = ttk.Frame(parent, style="Panel.TFrame")
        toolbar_frame.pack(side="bottom", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        return toolbar

    def _make_open_icon_button(self, parent, command):
        """Create the small 'open in a bigger window' button for a plot panel header, using the
        magnifying-glass icon when Pillow is available, falling back to plain text otherwise."""
        icon = self._get_magnifying_glass_icon()
        if icon is not None:
            return ttk.Button(parent, image=icon, command=command)
        return ttk.Button(parent, text="Open", command=command)

    def _build_ui(self):
        self._build_menu_bar()

        outer = ttk.Frame(self, padding=10, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        # Only the primary, frequently-adjusted clustering controls stay in the toolbar itself -
        # everything else now lives in the menu bar above (File / Annotation / Tools), grouped by
        # function instead of one long row of buttons that no longer fit the window. The actual
        # The single Process action lives per-row in the Tree Nodes list; these controls just set
        # the parameters/method/tool (and, via the checkboxes, which stages) that action uses.
        controls = ttk.Frame(outer, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="Batch Key:", style="App.TLabel").pack(side="left", padx=(4, 4))
        self.batch_key_var = tk.StringVar(value="Batch")
        ttk.Entry(controls, textvariable=self.batch_key_var, width=18).pack(side="left")
        ttk.Label(controls, text="Resolution:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.resolution_var = tk.StringVar(value="0.5")
        ttk.Entry(controls, textvariable=self.resolution_var, width=8).pack(side="left")
        ttk.Label(controls, text="Neighbors/Batch:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.neighbors_within_batch_var = tk.StringVar(value="25")
        ttk.Entry(controls, textvariable=self.neighbors_within_batch_var, width=6).pack(side="left")
        ttk.Label(controls, text="Max Epochs:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.max_epochs_var = tk.StringVar(value="")
        ttk.Entry(controls, textvariable=self.max_epochs_var, width=6).pack(side="left")

        # Each of the three pipeline stages below (Batch Correction, Clustering, Cell Type Prediction Tool)
        # gets its own dropdown to pick the method/tool. Selecting "None" is what decides that
        # stage is skipped when the user clicks a row's "Process" button - there's no separate
        # checkbox anymore. See process_selected_node() for the execution order (Batch
        # Correction -> Clustering -> Cell Type Prediction Tool).
        ttk.Label(controls, text="Batch Correction:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.correction_method_var = tk.StringVar(value="BBKNN")
        ttk.Combobox(
            controls, textvariable=self.correction_method_var, values=["None", "BBKNN", "SCVI"],
            state="readonly", width=8,
        ).pack(side="left")

        ttk.Label(controls, text="Clustering:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.clustering_method_var = tk.StringVar(value="Leiden")
        ttk.Combobox(
            controls, textvariable=self.clustering_method_var, values=["None", "Leiden"],
            state="readonly", width=8,
        ).pack(side="left")

        ttk.Label(controls, text="Cell Type Prediction Tool:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.annotation_tool_var = tk.StringVar(value="None")
        ttk.Combobox(
            controls, textvariable=self.annotation_tool_var, values=["None", "CellTypist"],
            state="readonly", width=10,
        ).pack(side="left")

        ttk.Label(controls, text="Auto-assign at:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.auto_assign_threshold_var = tk.StringVar(value="70")
        ttk.Entry(controls, textvariable=self.auto_assign_threshold_var, width=4).pack(side="left")
        ttk.Label(controls, text="%", style="App.TLabel").pack(side="left", padx=(2, 0))

        ttk.Label(controls, text="Not clear below:", style="App.TLabel").pack(side="left", padx=(20, 4))
        self.not_clear_threshold_var = tk.StringVar(value="40")
        ttk.Entry(controls, textvariable=self.not_clear_threshold_var, width=4).pack(side="left")
        ttk.Label(controls, text="%", style="App.TLabel").pack(side="left", padx=(2, 0))

        self.status_var = tk.StringVar(value="Load an h5ad file to begin.")
        ttk.Label(outer, textvariable=self.status_var, style="App.TLabel").pack(fill="x", pady=(0, 8))

        main = ttk.Panedwindow(outer, orient="horizontal")
        left = ttk.Frame(main, padding=6, style="Panel.TFrame")
        plots_panel = ttk.Frame(main, padding=6, style="Panel.TFrame")
        # weight=0 keeps the Tree Nodes pane pinned to a fixed width instead of growing to soak
        # up extra space when the window is resized/maximized; `plots_panel` (weight=1) absorbs
        # all the leftover space instead.
        main.add(left, weight=0)
        main.add(plots_panel, weight=1)

        ttk.Label(left, text="Tree Nodes", style="Panel.TLabel").pack(anchor="w")

        node_list_wrap = ttk.Frame(left, style="Panel.TFrame")
        node_list_wrap.pack(fill="both", expand=True, pady=(4, 8))
        self.node_list_canvas = tk.Canvas(node_list_wrap, bg=BG_MAIN, highlightthickness=0, bd=0)
        node_list_vscroll = ttk.Scrollbar(node_list_wrap, orient="vertical", command=self.node_list_canvas.yview)
        node_list_hscroll = ttk.Scrollbar(node_list_wrap, orient="horizontal", command=self.node_list_canvas.xview)
        self.node_list_canvas.configure(yscrollcommand=node_list_vscroll.set, xscrollcommand=node_list_hscroll.set)
        self.node_list_canvas.grid(row=0, column=0, sticky="nsew")
        node_list_vscroll.grid(row=0, column=1, sticky="ns")
        node_list_hscroll.grid(row=1, column=0, sticky="ew")
        node_list_wrap.rowconfigure(0, weight=1)
        node_list_wrap.columnconfigure(0, weight=1)

        self.node_list_container = ttk.Frame(self.node_list_canvas, style="Panel.TFrame")
        self.node_list_window = self.node_list_canvas.create_window((0, 0), window=self.node_list_container, anchor="nw")

        self.node_list_container.bind("<Configure>", lambda _e: self._sync_node_list_size())
        self.node_list_canvas.bind("<Configure>", lambda _e: self._sync_node_list_size())

        def _on_node_list_mousewheel(event):
            self.node_list_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.node_list_canvas.bind("<Enter>", lambda _e: self.node_list_canvas.bind_all("<MouseWheel>", _on_node_list_mousewheel))
        self.node_list_canvas.bind("<Leave>", lambda _e: self.node_list_canvas.unbind_all("<MouseWheel>"))

        self._left_pane = left
        self._main_paned = main

        # Plots panel replaces the old graphical Cluster Tree View: UMAP and the stacked bar
        # (predicted cell types) share the top row, half the panel's width each; the dot plot
        # spans the full width underneath. Each embedded plot auto-scales (via
        # _display_figure_scaled_to_fit) to fit whatever room its half/full-width pane ends up
        # with, rather than being drawn at a fixed size.
        plots_panel.rowconfigure(0, weight=1)
        plots_panel.rowconfigure(1, weight=1)
        plots_panel.columnconfigure(0, weight=1)

        top_row = ttk.Frame(plots_panel, style="Panel.TFrame")
        top_row.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        top_row.rowconfigure(0, weight=1)
        top_row.columnconfigure(0, weight=1)
        top_row.columnconfigure(1, weight=1)

        umap_col = ttk.Frame(top_row, style="Panel.TFrame")
        umap_col.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        umap_header = ttk.Frame(umap_col, style="Panel.TFrame")
        umap_header.pack(fill="x")
        self.umap_panel_title_var = tk.StringVar(value="UMAP")
        ttk.Label(umap_header, textvariable=self.umap_panel_title_var, style="Panel.TLabel").pack(side="left")
        self._make_open_icon_button(umap_header, self.open_umap_window).pack(side="right")
        ttk.Button(
            umap_header, text="New at Resolution...", width=16, command=self.open_new_umap_at_resolution,
        ).pack(side="right", padx=(0, 4))
        self.umap_frame = self._make_scrollable_plot_area(umap_col)

        stackedbar_col = ttk.Frame(top_row, style="Panel.TFrame")
        stackedbar_col.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        stackedbar_header = ttk.Frame(stackedbar_col, style="Panel.TFrame")
        stackedbar_header.pack(fill="x")
        ttk.Label(stackedbar_header, text="Predicted Cell Types", style="Panel.TLabel").pack(side="left")
        self._make_open_icon_button(stackedbar_header, self.open_stackedbar_window).pack(side="right")
        ttk.Button(
            stackedbar_header, text="New at Resolution...", width=16, command=self.open_new_stackedbar_at_resolution,
        ).pack(side="right", padx=(0, 4))
        self.stackedbar_frame = self._make_scrollable_plot_area(stackedbar_col)

        bottom_row = ttk.Frame(plots_panel, style="Panel.TFrame")
        bottom_row.grid(row=1, column=0, sticky="nsew")
        dotplot_section_header = ttk.Frame(bottom_row, style="Panel.TFrame")
        dotplot_section_header.pack(fill="x")
        ttk.Label(dotplot_section_header, text="Markers Dot Plots", style="Panel.TLabel").pack(side="left")
        # No single magnifying icon here - each loaded markers file gets its own stacked block
        # below (see _refresh_dotplot_stack), and each of those blocks has its own icon.
        self.dotplot_frame = self._make_scrollable_plot_area(bottom_row)

        main.pack(fill="both", expand=True)
        # Explicitly pin the Tree Nodes pane to a fixed width wide enough for its columns
        # (name, cells, resolution, cell-type combo, action button, saved checkbox).
        self.after(60, self._pin_tree_pane_width)

    def _pin_tree_pane_width(self):
        try:
            self.update_idletasks()
            self._main_paned.sashpos(0, 560)
        except (tk.TclError, AttributeError):
            pass

    def _make_scrollable_plot_area(self, parent):
        """Wrap a plot's drawing target in a horizontally + vertically scrollable canvas, so
        that when a plot's natural, data-driven size (e.g. a dot plot with many marker genes,
        or a stacked bar with many predicted cell types) exceeds the room the user has given
        that pane, it scrolls instead of being force-stretched/squished into an illegible
        shape. Returns the inner frame to pass as `target_frame` to the draw_* methods."""
        outer = ttk.Frame(parent, style="PlotArea.TFrame")
        outer.pack(fill="both", expand=True)
        scroll_canvas = tk.Canvas(outer, bg="white", highlightthickness=0, bd=0)
        v_scroll = ttk.Scrollbar(outer, orient="vertical", command=scroll_canvas.yview)
        h_scroll = ttk.Scrollbar(outer, orient="horizontal", command=scroll_canvas.xview)
        scroll_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        scroll_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        inner = ttk.Frame(scroll_canvas, style="PlotArea.TFrame")
        inner_window = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _apply_inner_size():
            # Size the embedded window to at least the visible viewport (so small content
            # still fills the pane), but let it grow bigger than that if the content genuinely
            # needs more room - which is what makes the scrollbars engage. Re-running this any
            # time the content's own required size changes (not just on window resize) is what
            # keeps a dynamically-resized image (e.g. our scaled-to-fit plots) from ever getting
            # silently clipped by a stale, smaller itemconfigure size.
            viewport_w = scroll_canvas.winfo_width()
            viewport_h = scroll_canvas.winfo_height()
            target_width = max(viewport_w, inner.winfo_reqwidth())
            target_height = max(viewport_h, inner.winfo_reqheight())
            scroll_canvas.itemconfigure(inner_window, width=target_width, height=target_height)
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        inner.bind("<Configure>", lambda _e: _apply_inner_size())
        scroll_canvas.bind("<Configure>", lambda _e: _apply_inner_size())

        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_shift_mousewheel(event):
            # Lets the mouse wheel (held with Shift) pan sideways - the only other way to reach
            # content off to the right (e.g. a wide dot plot's legend/colorbar) would be
            # dragging the thin horizontal scrollbar at the bottom, which is easy to miss.
            scroll_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_e):
            scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            scroll_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

        def _unbind_wheel(_e):
            scroll_canvas.unbind_all("<MouseWheel>")
            scroll_canvas.unbind_all("<Shift-MouseWheel>")

        scroll_canvas.bind("<Enter>", _bind_wheel)
        scroll_canvas.bind("<Leave>", _unbind_wheel)

        # Exposed so draw_* methods can read the TRUE, stable viewport size (scroll_canvas's own
        # size, which only changes on genuine window/pane resize) and force a re-sync after
        # swapping in a differently-sized image. Using `inner`'s own size for that instead would
        # create a feedback loop, since inner's size grows to match its content.
        inner._viewport_widget = scroll_canvas
        inner._resync_scrollable_area = _apply_inner_size

        return inner

    def _infer_embedded_figsize(self, frame, dpi: float = 100.0, min_dim: float = 3.0, max_dim: float = 16.0, chrome_px: float = 8.0):
        """Pick a figure size (in inches) that, once rendered at `dpi` and displayed via
        _display_figure_scaled_to_fit, fills the pane `frame` currently has available (scale
        ends up at ~1.0) instead of being capped by a mismatched aspect ratio or an indirectly
        undersized default. Falls back to a generic 6x5 default if the pane hasn't been laid out
        yet (e.g. very first render, before geometry has settled)."""
        if frame is None:
            return (6.0, 5.0)
        viewport = getattr(frame, "_viewport_widget", None)
        widget = viewport if viewport is not None else frame
        try:
            widget.update_idletasks()
            vw = widget.winfo_width()
            vh = widget.winfo_height()
        except tk.TclError:
            return (6.0, 5.0)
        if vw <= 1 or vh <= 1:
            return (6.0, 5.0)
        fig_w = (vw - chrome_px) / dpi
        fig_h = (vh - chrome_px) / dpi
        fig_w = min(max(fig_w, min_dim), max_dim)
        fig_h = min(max(fig_h, min_dim), max_dim)
        return (fig_w, fig_h)

    def _get_hidpi_render_scale(self) -> float:
        """How many actual physical screen pixels correspond to one 'logical' Tk pixel on this
        display - 1.0 on an ordinary monitor, ~2.0 on a Retina/HiDPI screen. Tk widget sizes
        (and therefore the target width/height _display_figure_scaled_to_fit scales an image
        down to) are always expressed in logical pixels, but the OS can render onto a screen
        with far more physical pixels per logical pixel than that. Rendering a matplotlib figure
        at a fixed DPI sized for logical pixels, then letting the OS stretch that same fixed
        pixel count across the physical screen, is exactly what makes the plots look soft/blurry
        on a Retina Mac or any display with OS-level scaling turned on - there simply aren't
        enough source pixels to fill the physical screen crisply. Multiplying the render DPI by
        this factor (see _display_figure_scaled_to_fit) fixes that at the source: the figure is
        rasterized with enough real pixels for the physical screen from the start, then scaled
        back down to the same logical display size, instead of being stretched up after the fact.
        Cached after the first call since a window's display doesn't change mid-session."""
        if getattr(self, "_hidpi_render_scale_cache", None) is not None:
            return self._hidpi_render_scale_cache
        try:
            # winfo_fpixels('1i') returns how many pixels Tk thinks are in one inch on this
            # display; 96 is Tk's standard reference (non-HiDPI) baseline.
            scale = self.winfo_fpixels("1i") / 96.0
        except Exception:
            scale = 1.0
        # Clamp to a sane range: never shrink (a bug shouldn't make things blurrier than
        # before), and cap how far we push it so a misdetected value can't demand an
        # enormous, slow-to-render raster.
        scale = max(1.0, min(scale, 3.0))
        self._hidpi_render_scale_cache = scale
        return scale

    def _display_figure_scaled_to_fit(self, fig, frame, cache_attr: str, min_scale: float = 0.15, measure_frame=None, fit_height: bool = True):
        """Render `fig` once at the natural size it was built with (which the caller has already
        sized generously enough to avoid dots/markers overlapping), then show it - centered on
        the container's white background - as a single image scaled DOWN (never up, never
        distorted - width and height always scale by the same factor, unless fit_height=False,
        see below) to fit the pane's available width AND height.

        This is deliberately different from just embedding a FigureCanvasTkAgg and letting Tk
        stretch it: that approach lets matplotlib redraw the figure at a new pixel size while
        marker/dot sizes stay fixed, so a narrower pane crowds the same number of dots into less
        room. Shrinking a single already-rendered picture uniformly instead preserves every dot's
        position and size relative to its neighbors - it just looks smaller, not more crowded.

        `frame` is where the rendered image actually gets placed (a Label is packed into it).
        `measure_frame` is what the available size is measured against, what `<Configure>` is
        bound to for re-rescaling on resize, and whose `_resync_scrollable_area()` gets called
        after every rescale - normally the same widget as `frame`, but for a stack of several
        plots sharing one outer scrollable area (see _refresh_dotplot_stack), `frame` is each
        plot's own small placement holder while `measure_frame` is the shared outer scrollable
        container: every stacked plot needs to fit the SAME outer viewport width, not whatever
        width its own placement holder happens to report (which, nested inside a canvas-based
        scrollable area, does not reliably reflect the true visible width). Defaults to `frame`
        when not given, matching every single-plot caller (UMAP, stacked bar chart).

        `fit_height` (default True) also shrinks to fit the available HEIGHT, appropriate for a
        single plot alone in its own pane. Pass False for a plot inside a panel the user scrolls
        vertically through (like the dot plot stack): there, only width should ever constrain the
        image - height should simply follow from that same width-based scale, however tall that
        makes it, since the surrounding panel scrolls to accommodate it rather than needing the
        whole image to already fit inside one screen's worth of vertical space.

        Binds with add="+" rather than replacing any existing binding, since multiple stacked
        plots can share one measure_frame and each needs its own rescale callback to keep firing
        on resize, not just whichever one bound most recently.

        Falls back to a plain, unscaled FigureCanvasTkAgg embed if Pillow / Agg rasterizing isn't
        available.
        """
        if Image is None or ImageTk is None or FigureCanvasAgg is None:
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
            canvas.get_tk_widget().pack(expand=True, padx=4, pady=4)
            setattr(self, cache_attr + "_canvas", canvas)
            setattr(self, cache_attr + "_figure", fig)
            return

        agg_canvas = FigureCanvasAgg(fig)
        # Re-rasterize at a higher pixel density on HiDPI displays (see
        # _get_hidpi_render_scale) - this changes only how many pixels the figure is drawn
        # with, not its layout: fonts, markers, and spacing are all specified in inches/points
        # in the figure, so they stay the same physical size, just crisper.
        hidpi_scale = self._get_hidpi_render_scale()
        if hidpi_scale > 1.0:
            fig.set_dpi(fig.get_dpi() * hidpi_scale)
        agg_canvas.draw()
        width_px, height_px = agg_canvas.get_width_height()
        native_image = Image.frombuffer(
            "RGBA", (width_px, height_px), agg_canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        ).copy()
        plt.close(fig)

        label = tk.Label(frame, bg="white", borderwidth=0, highlightthickness=0)
        label.pack(expand=True)

        measure_widget = measure_frame if measure_frame is not None else frame
        viewport = getattr(measure_widget, "_viewport_widget", None)
        resync = getattr(measure_widget, "_resync_scrollable_area", None)
        chrome_px = 8  # small margin so the plot never touches the pane's edge

        def _rescale(_event=None):
            if not label.winfo_exists():
                return
            if viewport is not None:
                avail_w = viewport.winfo_width()
                avail_h = viewport.winfo_height()
            else:
                avail_w = measure_widget.winfo_width()
                avail_h = measure_widget.winfo_height()
            if avail_w <= 1:
                avail_w = native_image.width
            if avail_h <= 1:
                avail_h = native_image.height
            scale_w = (avail_w - chrome_px) / native_image.width
            if fit_height:
                scale_h = (avail_h - chrome_px) / native_image.height
                scale = min(1.0, scale_w, scale_h)
            else:
                scale = min(1.0, scale_w)
            scale = max(min_scale, scale)
            target_w = max(1, int(native_image.width * scale))
            target_h = max(1, int(native_image.height * scale))
            resized = native_image.resize((target_w, target_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            setattr(self, cache_attr + "_photo", photo)  # keep a reference so Tk doesn't gc it
            label.configure(image=photo)
            if callable(resync):
                measure_widget.after_idle(resync)

        bind_target = viewport if viewport is not None else measure_widget
        bind_target.bind("<Configure>", _rescale, add="+")
        label.after(30, _rescale)  # first paint, once the widget has real dimensions
        setattr(self, cache_attr + "_native_image", native_image)
        setattr(self, cache_attr + "_label", label)

    def _build_log_window(self):
        self.log_window = tk.Toplevel(self)
        self.log_window.title("Processing Log")
        self.log_window.geometry("760x320")
        self.log_window.configure(bg=BG_PANEL)
        # Closing the window just hides it, so self.log_text stays alive and append_log never breaks.
        self.log_window.protocol("WM_DELETE_WINDOW", self.log_window.withdraw)

        outer = ttk.Frame(self.log_window, padding=8, style="Panel.TFrame")
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="Processing Log", style="Panel.TLabel").pack(side="left")
        ttk.Button(header, text="Save Log", command=self.save_log).pack(side="right", padx=(4, 0))
        ttk.Button(header, text="Clear Log", command=self.clear_log).pack(side="right", padx=(4, 0))

        log_frame = ttk.Frame(outer, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", bg=LOG_BG, fg=FG_MAIN, insertbackground=FG_MAIN, relief="flat")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def show_log_window(self):
        self.log_window.deiconify()
        self.log_window.lift()
        self.log_window.focus_force()

    def save_log(self):
        default_name = f"processing_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        path = filedialog.asksaveasfilename(
            title="Save processing log",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            content = self.log_text.get("1.0", "end")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            self.set_status(f"Log saved to {path}")
        except Exception as exc:
            messagebox.showerror("Save Log", f"Could not save log file:\n{exc}")

    def _describe_node_for_popup(self, node: Optional[ClusterNode]) -> str:
        """One-line summary shown at the top of each plot popup: which cluster it's showing,
        the resolution it was (re)clustered at, and how many cells it contains."""
        if node is None:
            return "No node selected."
        resolution = node.child_resolution
        resolution_text = f"{resolution:g}" if resolution is not None else "not yet clustered"
        return f"Cluster: {node.label}      Resolution: {resolution_text}      Cells: {node.n_cells():,}"

    def _current_node_label(self) -> str:
        if self.selected_node_id and self.selected_node_id in self.model.nodes:
            return self.model.nodes[self.selected_node_id].label
        return "plot"

    def _read_float_var(self, attr_name: str, default: float) -> float:
        """Read a float from a named StringVar attribute (e.g. a popup's font-scale field),
        falling back to `default` if it's missing, blank, or not a valid positive number."""
        var = getattr(self, attr_name, None)
        if var is None:
            return default
        text = var.get().strip()
        if not text:
            return default
        try:
            value = float(text)
            return value if value > 0 else default
        except (TypeError, ValueError):
            return default

    def _read_popup_figsize(self, attr_name: str):
        """Read the width/height (in inches) the user has set in a popup's Size controls.
        Returns None (meaning 'let the caller use its own auto-computed default') if either
        field is blank or not numeric, rather than silently forcing some fixed fallback size -
        that matters for dot plots in particular, where 'blank/auto' should mean the same
        generous per-gene/group sizing the embedded stack already uses, not a flat guess."""
        width_var = getattr(self, attr_name + "_width_var", None)
        height_var = getattr(self, attr_name + "_height_var", None)

        def _parse(var):
            if var is None:
                return None
            text = var.get().strip()
            if not text:
                return None
            try:
                value = float(text)
                return value if value > 0 else None
            except (TypeError, ValueError):
                return None

        width = _parse(width_var)
        height = _parse(height_var)
        if width is None or height is None:
            return None
        return (width, height)

    def _save_plot_popup(self, cache_attr: str):
        """Save the currently-rendered figure for one of the three plots to PNG, PDF, or TIFF,
        at print-quality resolution regardless of what's shown on screen."""
        fig = getattr(self, cache_attr + "_figure", None)
        if fig is None:
            messagebox.showwarning("Save Plot", "No plot has been rendered yet.")
            return
        default_name = f"{self._current_node_label()}_{cache_attr}"
        path = filedialog.asksaveasfilename(
            title="Save Plot As",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("TIFF image", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            fmt_map = {".png": "png", ".pdf": "pdf", ".tif": "tiff", ".tiff": "tiff"}
            fmt = fmt_map.get(ext, "png")
            fig.savefig(path, format=fmt, dpi=300, facecolor="white", bbox_inches="tight")
            self.append_log(f"Saved plot to {path}")
            self.set_status(f"Saved plot to {os.path.basename(path)}.")
        except Exception as exc:
            self.append_log(f"Failed to save plot: {exc}")
            messagebox.showerror("Save Plot", f"Could not save plot:\n{exc}")

    def _open_plot_popup(self, attr_name: str, title: str, draw_method, cache_attr: str, default_size=("6", "6"), show_dotplot_style_controls: bool = False):
        """Open (or, if already open, refresh + raise) a bigger popup window rendering one plot
        (UMAP preview, a specific loaded markers file's dot plot, or predicted cell types) for
        whichever node is currently selected in the Tree Nodes list. `default_size` seeds the
        Size (in) fields - pass ("", "") to start blank/auto instead of a fixed 6x6, which is
        what dot plot popups do so they open at the same generous, gene/group-aware size the
        embedded stack uses. `show_dotplot_style_controls` adds Font Scale and Dot Padding
        fields (dot plots only) - increasing padding is the direct fix for overlapping dots."""
        popup = getattr(self, attr_name, None)
        if popup is not None and popup.winfo_exists():
            popup.deiconify()
            popup.lift()
            popup.focus_force()
            self._refresh_plot_popup(attr_name, draw_method)
            return

        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("900x760")
        popup.configure(bg=BG_PANEL)
        setattr(self, attr_name, popup)

        width_var = tk.StringVar(value=default_size[0])
        height_var = tk.StringVar(value=default_size[1])
        setattr(self, attr_name + "_width_var", width_var)
        setattr(self, attr_name + "_height_var", height_var)

        header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        header.pack(fill="x")
        ttk.Label(header, text=title, style="Panel.TLabel").pack(side="left")
        ttk.Button(header, text="Refresh", command=lambda: self._refresh_plot_popup(attr_name, draw_method)).pack(side="right")
        ttk.Button(header, text="Save Plot...", command=lambda: self._save_plot_popup(cache_attr)).pack(side="right", padx=(0, 6))

        if show_dotplot_style_controls:
            # Starting values match draw_dot_plot's own new defaults (see there for why) so the
            # numbers shown here accurately describe what's already on screen, rather than
            # implying "1.0/1.0, unadjusted" when the embedded view is actually already using
            # more generous padding/font-scale defaults to avoid overlap out of the box.
            font_scale_var = tk.StringVar(value="0.65")
            padding_var = tk.StringVar(value="1.3")
            setattr(self, attr_name + "_font_scale_var", font_scale_var)
            setattr(self, attr_name + "_padding_var", padding_var)
            style_frame = ttk.Frame(header, style="Panel.TFrame")
            style_frame.pack(side="right", padx=(0, 12))
            ttk.Label(style_frame, text="Dot padding:", style="Panel.TLabel").pack(side="left")
            ttk.Entry(style_frame, textvariable=padding_var, width=4).pack(side="left", padx=(4, 10))
            ttk.Label(style_frame, text="Font scale:", style="Panel.TLabel").pack(side="left")
            ttk.Entry(style_frame, textvariable=font_scale_var, width=4).pack(side="left", padx=(4, 6))
            ttk.Button(style_frame, text="Apply", command=lambda: self._refresh_plot_popup(attr_name, draw_method)).pack(side="left")

        size_frame = ttk.Frame(header, style="Panel.TFrame")
        size_frame.pack(side="right", padx=(0, 12))
        ttk.Label(size_frame, text="Size (in):", style="Panel.TLabel").pack(side="left")
        ttk.Entry(size_frame, textvariable=width_var, width=4).pack(side="left", padx=(4, 2))
        ttk.Label(size_frame, text="x", style="Panel.TLabel").pack(side="left")
        ttk.Entry(size_frame, textvariable=height_var, width=4).pack(side="left", padx=(2, 6))
        ttk.Button(size_frame, text="Apply", command=lambda: self._refresh_plot_popup(attr_name, draw_method)).pack(side="left")

        info_var = tk.StringVar(value="")
        setattr(self, attr_name + "_info_var", info_var)
        info_row = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 4, 10, 0))
        info_row.pack(fill="x")
        ttk.Label(info_row, textvariable=info_var, style="Panel.TLabel", foreground=FG_MUTED).pack(anchor="w")
        if show_dotplot_style_controls:
            ttk.Label(
                info_row,
                text="Tip: if dots overlap, raise Dot padding first (e.g. 1.3-1.6) before "
                "enlarging Size - it shrinks dots directly instead of just adding blank space.",
                style="Panel.TLabel", foreground=FG_MUTED, wraplength=760, justify="left",
            ).pack(anchor="w", pady=(2, 0))

        content_wrap = ttk.Frame(popup, style="Panel.TFrame", padding=10)
        content_wrap.pack(fill="both", expand=True)
        content_frame = self._make_scrollable_plot_area(content_wrap)
        setattr(self, attr_name + "_content", content_frame)

        self._refresh_plot_popup(attr_name, draw_method)

    def _refresh_plot_popup(self, attr_name: str, draw_method):
        content_frame = getattr(self, attr_name + "_content", None)
        if content_frame is None or not content_frame.winfo_exists():
            return
        node = None
        if self.selected_node_id and self.selected_node_id in self.model.nodes:
            node = self.model.nodes[self.selected_node_id]
        info_var = getattr(self, attr_name + "_info_var", None)
        if info_var is not None:
            info_var.set(self._describe_node_for_popup(node))
        figsize_override = self._read_popup_figsize(attr_name)
        draw_method(node, target_frame=content_frame, big=True, figsize_override=figsize_override)

    def _refresh_open_plot_popups(self):
        """Refresh whichever plot popups the user currently has open (without forcing new ones
        open) - used when the selected node changes but clustering hasn't just run, so any
        windows already on screen stay in sync instead of showing stale data."""
        for attr_name, draw_method in (
            ("umap_popup_window", self.draw_umap_preview),
            ("stackedbar_popup_window", self.draw_stacked_bar_plot),
        ):
            popup = getattr(self, attr_name, None)
            if popup is not None and popup.winfo_exists():
                self._refresh_plot_popup(attr_name, draw_method)

    def _refresh_embedded_plots(self, node: Optional["ClusterNode"]):
        """Redraw the plots embedded in the main plots panel (UMAP + stacked bar half width each
        on top, one or more stacked marker dot plots full width below - one per loaded markers
        CSV) for the given node. Each uses big=False/its own natural size so it fits whatever
        room its pane currently has."""
        title_var = getattr(self, "umap_panel_title_var", None)
        if title_var is not None:
            title_var.set(f"UMAP: {node.label}" if node is not None else "UMAP")
        self.draw_umap_preview(node, target_frame=getattr(self, "umap_frame", None), big=False)
        self._refresh_dotplot_stack(node)
        self.draw_stacked_bar_plot(node, target_frame=getattr(self, "stackedbar_frame", None), big=False)

    def _refresh_all_plot_views(self):
        """Refresh both the embedded plots panel and any open plot popups for whichever node is
        currently selected."""
        node = None
        if self.selected_node_id and self.selected_node_id in self.model.nodes:
            node = self.model.nodes[self.selected_node_id]
        self._refresh_embedded_plots(node)
        self._refresh_open_plot_popups()

    def open_umap_window(self):
        self._open_plot_popup("umap_popup_window", "UMAP (large)", self.draw_umap_preview, cache_attr="umap")

    def open_stackedbar_window(self):
        self._open_plot_popup("stackedbar_popup_window", "Predicted Cell Types (large)", self.draw_stacked_bar_plot, cache_attr="stackedbar")

    def _prompt_and_compute_adhoc_labels(self, purpose: str):
        """Shared by "New UMAP at Resolution..." and "New Stacked Bar at Resolution...":
        validate a node is selected and already clustered, ask for a resolution, and compute
        Leiden labels at it via compute_adhoc_leiden_labels (without touching the node's real,
        committed clustering). Returns (node, resolution, labels), or None if the person
        canceled or something failed (an error dialog has already been shown in that case)."""
        if self.busy:
            return None
        if not self.selected_node_id or self.selected_node_id not in self.model.nodes:
            messagebox.showwarning("No selection", "Select a node first.")
            return None
        node = self.model.nodes[self.selected_node_id]
        if node.umap_adata is None or "X_umap" not in node.umap_adata.obsm.keys():
            messagebox.showerror(
                purpose,
                f"'{node.label}' needs to be clustered at least once (Process) before "
                "previewing another resolution here.",
            )
            return None
        default_res = node.child_resolution if node.child_resolution is not None else 0.5
        resolution = simpledialog.askfloat(
            purpose,
            f"Resolution to preview for '{node.label}' ({node.umap_adata.n_obs} cells):",
            initialvalue=default_res, minvalue=0.01, maxvalue=10.0, parent=self,
        )
        if resolution is None:
            return None
        try:
            self.set_busy(True, f"Computing Leiden at resolution {resolution} for {node.label}...")
            self.append_log(f"=== {purpose} requested at resolution {resolution} for node {node.label} ({node.node_id}) ===")
            labels = self.service.compute_adhoc_leiden_labels(node, resolution, logger=self.append_log)
            self.set_status(f"Computed Leiden r={resolution} for {node.label} ({labels.nunique()} clusters).")
        except ImportError as exc:
            messagebox.showerror(purpose, str(exc))
            self.set_status(f"{purpose} failed: missing dependency.")
            return None
        except Exception as exc:
            self.append_log(f"{purpose} failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror(purpose, str(exc))
            self.set_status(f"{purpose} failed.")
            return None
        finally:
            self.set_busy(False, self.status_var.get())
        return node, resolution, labels

    def open_new_umap_at_resolution(self):
        """Header button next to the UMAP panel's magnifying glass: preview the selected node's
        existing UMAP embedding colored by a fresh, one-off Leiden cut at a resolution the
        person chooses, in its own window - without disturbing the node's actual, committed
        clustering."""
        result = self._prompt_and_compute_adhoc_labels("New UMAP at Resolution")
        if result is None:
            return
        node, resolution, labels = result
        coords = np.asarray(node.umap_adata.obsm["X_umap"])
        self._show_adhoc_umap_window(node, resolution, coords, labels)

    def _show_adhoc_umap_window(self, node: ClusterNode, resolution: float, coords, labels):
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("New UMAP at Resolution", "Matplotlib is not installed.")
            return
        popup = tk.Toplevel(self)
        popup.title(f"UMAP \u2014 {node.label} (resolution {resolution})")
        popup.geometry("820x680")
        popup.configure(bg=BG_PANEL)

        header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        header.pack(fill="x")
        ttk.Label(
            header, text=f"Node: {node.label}    Resolution: {resolution}    Cells: {len(labels)}",
            style="Panel.TLabel",
        ).pack(side="left")

        plot_frame = ttk.Frame(popup, style="Panel.TFrame", padding=10)
        plot_frame.pack(fill="both", expand=True)

        fig = Figure(figsize=(7.5, 6.2), dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        labels_list = labels.astype(str).tolist()
        point_colors = self._color_array_for_clusters(labels_list)
        ax.scatter(coords[:, 0], coords[:, 1], s=7, alpha=0.85, c=point_colors)
        labels_array = np.asarray(labels_list)
        for cluster_id in sorted(set(labels_list)):
            mask = labels_array == cluster_id
            if np.any(mask):
                centroid_x = float(np.mean(coords[mask, 0]))
                centroid_y = float(np.mean(coords[mask, 1]))
                ax.text(
                    centroid_x, centroid_y, cluster_id,
                    fontsize=9, fontweight="bold", color="black",
                    ha="center", va="center", zorder=5,
                )
        ax.set_title(f"UMAP clusters r={resolution} \u2014 {node.label}", fontsize=10, color="black")
        ax.set_xlabel("UMAP1", fontsize=8, color="black")
        ax.set_ylabel("UMAP2", fontsize=8, color="black")
        ax.tick_params(labelsize=7, colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.grid(alpha=0.2, color="#d1d5db")
        fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.09)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        canvas.get_tk_widget().pack(expand=True, fill="both")
        self._attach_plot_toolbar(canvas, plot_frame)

    def open_new_stackedbar_at_resolution(self):
        """Header button next to the Predicted Cell Types panel's magnifying glass: preview the
        selected node's existing per-cell CellTypist predictions cross-tabulated against a
        fresh, one-off Leiden cut at a resolution the person chooses, in its own window -
        without disturbing the node's actual, committed clustering."""
        result = self._prompt_and_compute_adhoc_labels("New Stacked Bar at Resolution")
        if result is None:
            return
        node, resolution, labels = result
        if node.predicted_labels is None:
            messagebox.showerror(
                "New Stacked Bar at Resolution",
                "Load a CellTypist model and click 'Predict Cell Types' for this node first - "
                "the stacked bar needs per-cell predicted labels to cross-tabulate against "
                "clusters.",
            )
            return
        self._show_adhoc_stackedbar_window(node, resolution, labels)

    def _show_adhoc_stackedbar_window(self, node: ClusterNode, resolution: float, labels):
        if Figure is None or FigureCanvasTkAgg is None or plt is None:
            messagebox.showerror("New Stacked Bar at Resolution", "Matplotlib is not installed.")
            return
        try:
            common_barcodes = labels.index.intersection(node.predicted_labels.index)
            if len(common_barcodes) == 0:
                messagebox.showerror(
                    "New Stacked Bar at Resolution",
                    "No matching cells between this resolution's clusters and the CellTypist predictions.",
                )
                return
            clusters = labels.loc[common_barcodes].astype(str)
            predicted = node.predicted_labels.loc[common_barcodes].astype(str)
            crosstab = pd.crosstab(clusters, predicted, normalize="index")
            try:
                crosstab = crosstab.reindex(sorted(crosstab.index, key=lambda x: int(x)))
            except (ValueError, TypeError):
                crosstab = crosstab.sort_index()
            # Same "biggest type at the bottom of every bar" ordering as the main stacked bar.
            type_totals = predicted.value_counts()
            ordered_types = [t for t in type_totals.index if t in crosstab.columns]
            crosstab = crosstab.reindex(columns=ordered_types)

            fig_width = max(8.0, 0.75 * len(crosstab.index) + 3.5)
            fig = Figure(figsize=(fig_width, 6.2), dpi=150, facecolor="white")
            ax = fig.add_subplot(111)
            colors = self._qualitative_colors(len(crosstab.columns))
            crosstab.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Fraction of cells", fontsize=8, color="black")
            ax.set_xlabel(f"Leiden cluster (r={resolution})", fontsize=8, color="black")
            ax.set_title(f"Predicted cell types: {node.label} (resolution {resolution})", fontsize=9, color="black")
            ax.tick_params(labelsize=7, colors="black", rotation=0)
            for spine in ax.spines.values():
                spine.set_color("black")
            ax.legend(fontsize=6, title="Predicted type", title_fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            fig.tight_layout()

            popup = tk.Toplevel(self)
            popup.title(f"Predicted Cell Types \u2014 {node.label} (resolution {resolution})")
            popup.geometry("900x680")
            popup.configure(bg=BG_PANEL)
            header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
            header.pack(fill="x")
            ttk.Label(
                header, text=f"Node: {node.label}    Resolution: {resolution}    Clusters: {crosstab.shape[0]}",
                style="Panel.TLabel",
            ).pack(side="left")
            plot_frame = ttk.Frame(popup, style="Panel.TFrame", padding=10)
            plot_frame.pack(fill="both", expand=True)
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
            canvas.get_tk_widget().pack(expand=True, fill="both")
            self._attach_plot_toolbar(canvas, plot_frame)
        except Exception as exc:
            self.append_log(f"[New Stacked Bar | {node.label}] Failed to render: {exc}")
            messagebox.showerror("New Stacked Bar at Resolution", f"Could not render cell type distribution: {exc}")

    def set_status(self, text):
        self.status_var.set("" if text is None else str(text))
        self.update_idletasks()

    def append_log(self, message):
        safe_message = "" if message is None else str(message)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", safe_message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.set_status("Processing log cleared.")

    def set_busy(self, busy: bool, status: str):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for menu, index in getattr(self, "_busy_menu_items", []):
            menu.entryconfig(index, state=state)
        self.configure(cursor="watch" if busy else "")
        self.set_status(status)
        self.update_idletasks()

    def load_h5ad_file(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[("H5AD files", "*.h5ad"), ("All files", "*")])
        if not path:
            return
        try:
            self.set_busy(True, "Loading h5ad file...")
            self.append_log("=== Load requested ===")
            self.service.batch_key = self.batch_key_var.get().strip()
            adata = self.service.load_h5ad(path, logger=self.append_log)
            root_id = self.model.create_root(adata=adata, label="h5ad_root")
            self.selected_node_id = root_id
            self.refresh_views()
            self.append_log(f"Root node created with id {root_id}")
            self.set_status(f"Loaded {path} with {adata.n_obs} cells and {adata.n_vars} genes.")
        except Exception as exc:
            self.append_log(f"Load failed: {exc}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Load error", str(exc))
            self.set_status("Load failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def process_selected_node(self):
        """Run whichever pipeline stages are set to something other than 'None' at the top
        (Batch Correction, Clustering, Cell Type Prediction Tool) on the selected node, always in that fixed
        order - correction, then clustering, then annotation - regardless of which combination is
        selected. Each dropdown both picks the method/tool AND decides whether that stage runs:
        selecting 'None' skips it.

        Cell type annotation refuses to run unless the node has been batch-corrected and
        clustered - either earlier (a previous Process click already set node.last_cluster_key /
        node.umap_adata) or right now as part of this same click (Clustering is also selected).
        Clustering itself no longer requires a fresh correction in the same click - if there's no
        correction result recorded yet, it falls back to the node's own data (see below)."""
        if self.busy:
            return
        if not self.selected_node_id:
            messagebox.showwarning("No selection", "Select a node first.")
            return
        node = self.model.nodes[self.selected_node_id]

        correction_method = self.correction_method_var.get().strip()
        clustering_method = self.clustering_method_var.get().strip()
        annotation_tool = self.annotation_tool_var.get().strip()
        do_correction = correction_method != "None"
        do_clustering = clustering_method != "None"
        do_annotation = annotation_tool != "None"

        if not (do_correction or do_clustering or do_annotation):
            messagebox.showwarning(
                "Nothing to do",
                "Set at least one of Batch Correction, Clustering, and/or Cell Type Prediction Tool to "
                "something other than 'None' before clicking Process.",
            )
            return

        # Cell type annotation needs a corrected + clustered node. If Clustering isn't also
        # selected in this same click, the node must already have been clustered previously.
        if do_annotation and not do_clustering:
            already_clustered = (
                node.last_cluster_key is not None
                and node.umap_adata is not None
                and node.last_cluster_key in getattr(node.umap_adata, "obs", {}).columns
            )
            if not already_clustered:
                messagebox.showerror(
                    "Cannot run cell type annotation",
                    "This node hasn't been batch-corrected and clustered yet. Set 'Batch "
                    "Correction' and 'Clustering' above to something other than 'None' as well "
                    "(or process those first), then run cell type annotation.",
                )
                return

        # Clustering normally needs a batch-corrected node from this session. But the loaded
        # h5ad may already be batch-corrected externally (e.g. it already carries its own
        # neighbor graph/latent embedding), so if there's no correction result recorded yet,
        # fall back to the node's own data instead of blocking - run_leiden_clustering will
        # raise its own clear error if no neighbor graph actually exists.

        if do_clustering and clustering_method != "Leiden":
            messagebox.showerror("Unsupported clustering method", f"'{clustering_method}' is not supported yet.")
            return
        if do_annotation and annotation_tool != "CellTypist":
            messagebox.showerror("Unsupported cell type tool", f"'{annotation_tool}' is not supported yet.")
            return

        resolution = None
        if do_clustering:
            resolution_text = self.resolution_var.get().strip()
            if not resolution_text:
                messagebox.showerror("Invalid resolution", "Resolution cannot be blank. Enter a numeric value (e.g. 0.5).")
                return
            try:
                resolution = float(resolution_text)
            except ValueError:
                messagebox.showerror("Invalid resolution", f"'{resolution_text}' is not a valid resolution number.")
                return

        neighbors_within_batch = 25
        if do_correction and correction_method in ("BBKNN", "SCVI"):
            neighbors_within_batch_text = self.neighbors_within_batch_var.get().strip()
            if not neighbors_within_batch_text:
                messagebox.showerror("Invalid neighbors/batch", "Neighbors/batch cannot be blank. Enter a whole number (e.g. 25).")
                return
            try:
                neighbors_within_batch = int(neighbors_within_batch_text)
                if neighbors_within_batch < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid neighbors/batch", f"'{neighbors_within_batch_text}' is not a valid whole number >= 1.")
                return

        max_epochs = None
        if do_correction and correction_method == "SCVI":
            max_epochs_text = self.max_epochs_var.get().strip()
            if max_epochs_text:
                try:
                    max_epochs = int(max_epochs_text)
                    if max_epochs < 1:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("Invalid max epochs", f"'{max_epochs_text}' is not a valid whole number >= 1. Leave blank to use scvi-tools' automatic default.")
                    return

        if do_clustering and node.children:
            old_resolution = node.child_resolution
            descendant_count = self.model.count_descendants(node.node_id)
            confirm_message = (
                f"This node already has children created with resolution {old_resolution}. "
                f"Re-clustering with resolution {resolution} will delete {descendant_count} existing node(s) "
                f"(its children and all their descendants) and replace them. Continue?"
            )
            if not messagebox.askyesno("Already clustered", confirm_message):
                return

        stages_run = []
        try:
            corrected = node.integration_adata  # may already exist from an earlier correction
            if corrected is None and do_clustering and not do_correction:
                corrected = node.adata
                self.append_log(
                    f"No correction result recorded for {node.label} in this session - using its "
                    f"own data as-is for clustering (assuming it was already batch-corrected)."
                )

            if do_correction:
                self.set_busy(True, f"Running {correction_method} correction on {node.label}...")
                self.append_log(f"=== {correction_method} correction requested for node {node.label} ({node.node_id}) ===")
                self.service.batch_key = self.batch_key_var.get().strip()
                if correction_method == "BBKNN":
                    self.append_log(f"Using batch key '{self.service.batch_key}', neighbors/batch {neighbors_within_batch}")
                    corrected = self.service.run_bbknn_correction(node, neighbors_within_batch=neighbors_within_batch, logger=self.append_log)
                elif correction_method == "SCVI":
                    # See H5ADClusterService.run_scvi_correction() for where to add your scVI code.
                    self.append_log(f"Using batch key '{self.service.batch_key}', neighbors/batch {neighbors_within_batch}, max epochs {max_epochs if max_epochs is not None else 'auto'}")
                    corrected = self.service.run_scvi_correction(node, max_epochs=max_epochs, neighbors_within_batch=neighbors_within_batch, logger=self.append_log)
                else:
                    raise ValueError(f"Unknown batch correction method '{correction_method}'")
                node.integration_adata = corrected
                node.integration_method = correction_method.lower()
                node.child_neighbors_within_batch = neighbors_within_batch
                self.append_log(f"{correction_method} correction complete for {node.label} ({corrected.n_obs} cells).")
                stages_run.append(f"{correction_method} correction")

            if do_clustering:
                self.set_busy(True, f"Running {clustering_method} clustering on {node.label}...")
                self.append_log(f"=== {clustering_method} clustering requested for node {node.label} ({node.node_id}) ===")
                umap_kwargs = {"umap_min_dist": 0.1, "umap_spread": 1.0} if node.integration_method == "scvi" else {}
                result = self.service.run_leiden_clustering(node, corrected, resolution=resolution, logger=self.append_log, **umap_kwargs)
                new_cluster_count = len(result["children"])
                if new_cluster_count <= 1:
                    reason = (
                        f"Clustering '{node.label}' at resolution {resolution} found only {new_cluster_count} "
                        f"group(s). This node's cells already look homogeneous at this resolution, so no new "
                        f"subclusters were created \u2014 a single subcluster would just duplicate the parent "
                        f"node and add no information.\n\nTry raising the resolution, or check whether this "
                        f"node has enough distinct cells to justify splitting further."
                    )
                    self.append_log(f"Skipped subcluster creation for '{node.label}': {reason}")
                    messagebox.showinfo("Cannot subcluster further", reason)
                    self.set_status(f"No new subclusters created for {node.label} (only {new_cluster_count} group found).")
                    return
                if node.children:
                    removed_count = len(node.children)
                    self.append_log(
                        f"Removing {removed_count} old child branch(es) created with resolution {node.child_resolution} "
                        f"before attaching new resolution {resolution} results."
                    )
                    self.model.remove_children(node.node_id)
                    if self.selected_node_id not in self.model.nodes:
                        self.selected_node_id = node.node_id
                node.umap_adata = result["clustered_adata"]
                node.last_cluster_key = result["cluster_key"]
                node.child_resolution = resolution
                self.append_log(f"Stored clustered UMAP and labels in selected node using key {node.last_cluster_key}")
                for cluster_value, child_adata in result["children"].items():
                    if node.parent_id is None:
                        label = str(cluster_value)
                    else:
                        label = f"{node.label}.{cluster_value}"
                    child_id = self.model.add_child(self.selected_node_id, child_adata, label=label, cluster_value=cluster_value, resolution=resolution, neighbors_within_batch=node.child_neighbors_within_batch)
                    self.append_log(f"Attached child node {child_id} as {label}")
                # This node just gained children, so it's no longer a leaf - clear any manually
                # assigned cell type (only leaves may have one; its combo box will show disabled
                # and its highlight reverts to the default non-leaf color), and clear the consumed
                # integration result so a future correction+clustering run starts fresh. Also
                # clear predicted_labels: it's a pandas Series indexed by the OLD set of cell
                # barcodes from whenever annotation last actually ran, but node.umap_adata just
                # got replaced with a brand-new AnnData (a different partition, and potentially a
                # different subset of cells if a BBKNN re-run's filter_small_batches excluded
                # different cells this time). Without this reset, the stacked bar chart and any
                # future auto-assign would intersect the NEW cluster labels against the STALE
                # prediction's cell barcodes - any new cluster whose cells don't overlap with the
                # old prediction's barcodes silently gets ZERO matching cells and its bar simply
                # never appears, with no error to flag it. Clearing this forces a fresh
                # 'Predict Cell Types' run against the current clustering instead.
                node.assigned_cell_type = None
                node.integration_adata = None
                node.integration_method = None
                node.predicted_labels = None
                stages_run.append(f"{clustering_method} clustering ({len(result['children'])} subclusters)")

            if do_annotation:
                self.set_busy(True, f"Running {annotation_tool} annotation on {node.label}...")
                self.append_log(f"=== {annotation_tool} cell type annotation requested for node {node.label} ({node.node_id}) ===")
                if annotation_tool == "CellTypist":
                    if celltypist is None:
                        messagebox.showerror(
                            "Cell Type Annotation",
                            "The celltypist package is not installed.\nInstall with: pip install celltypist",
                        )
                    elif self.celltypist_model is None:
                        messagebox.showerror(
                            "Cell Type Annotation",
                            "No CellTypist model is loaded. Use Annotation > Load CellTypist Model... first.",
                        )
                    else:
                        node.predicted_labels = None  # force a fresh prediction for this run
                        self._run_celltypist_prediction(node)
                        stages_run.append("CellTypist annotation")

            self.refresh_views()
            self._refresh_embedded_plots(node)
            if stages_run:
                self.set_status(f"Completed for {node.label}: {', '.join(stages_run)}.")
            else:
                self.set_status(f"No stages completed for {node.label}.")
        except NotImplementedError as exc:
            self.append_log(f"{correction_method} correction is not implemented yet: {exc}")
            messagebox.showinfo(
                "SCVI Correction",
                "SCVI correction is a placeholder - implement your own code in "
                "H5ADClusterService.run_scvi_correction().",
            )
            self.set_status(f"{correction_method} correction is not implemented yet.")
        except Exception as exc:
            self.append_log("Processing failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Processing error", str(exc))
            self.set_status("Processing failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def open_resolution_sweep_dialog(self):
        """Tools > Resolution Sweep...: try several Leiden resolutions on the selected node's
        already-corrected data (reusing whichever neighbor graph is available - a fresh
        correction that hasn't been clustered yet, or an earlier correction+clustering result),
        without committing to any of them. Shows cluster count and Adjusted Rand Index vs the
        previous resolution for each value tried, so the user can pick a resolution based on
        where that curve is stable (a "plateau") rather than by trial and error re-clustering."""
        if self.busy:
            return
        if not self.selected_node_id:
            messagebox.showwarning("No selection", "Select a node first.")
            return
        node = self.model.nodes[self.selected_node_id]
        # Prefer a fresh, not-yet-clustered correction result if one is sitting on the node;
        # otherwise fall back to the neighbor graph baked into its last clustering result. Both
        # carry a real neighbor graph the sweep can reuse without re-running correction.
        graph_adata = node.integration_adata if node.integration_adata is not None else node.umap_adata
        if graph_adata is None:
            messagebox.showerror(
                "Resolution Sweep",
                "This node needs to be batch-corrected first (or already clustered) before a "
                "resolution sweep can run. Check 'Batch Correction' above and click Process "
                "(or Batch Correction + Clustering together), then try the sweep again.",
            )
            return

        default_resolutions = "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0"
        text = simpledialog.askstring(
            "Resolution Sweep",
            f"Resolutions to try on '{node.label}' ({graph_adata.n_obs} cells), comma-separated:",
            initialvalue=default_resolutions,
            parent=self,
        )
        if text is None:
            return
        try:
            resolutions = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
            if not resolutions:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a comma-separated list of numbers, e.g. 0.2, 0.4, 0.8, 1.2")
            return

        try:
            self.set_busy(True, f"Running resolution sweep on {node.label}...")
            self.append_log(f"=== Resolution sweep requested for node {node.label} ({node.node_id}): {resolutions} ===")
            results = self.service.run_resolution_sweep(node, graph_adata, resolutions, logger=self.append_log)
            self._show_resolution_sweep_results(node, results)
            self.set_status(f"Resolution sweep complete for {node.label} ({len(resolutions)} resolutions tried).")
        except ImportError as exc:
            messagebox.showerror("Resolution Sweep", str(exc))
            self.set_status("Resolution sweep failed: missing dependency.")
        except Exception as exc:
            self.append_log("Resolution sweep failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Resolution sweep error", str(exc))
            self.set_status("Resolution sweep failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def _show_resolution_sweep_results(self, node: ClusterNode, results: List[Dict[str, object]]):
        """Popup showing the resolution sweep results as a table plus a dual-axis chart (cluster
        count and Adjusted Rand Index vs. the previous resolution, both against resolution) so
        the user can visually spot a stable plateau before choosing a resolution to commit to
        via the toolbar's Resolution field + Process button."""
        popup = tk.Toplevel(self)
        popup.title(f"Resolution Sweep - {node.label}")
        popup.geometry("760x620")
        popup.configure(bg=BG_PANEL)

        header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        header.pack(fill="x")
        ttk.Label(
            header, text=f"Resolution sweep for '{node.label}' ({node.adata.n_obs} cells)", style="Panel.TLabel",
        ).pack(side="left")

        guidance = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 6, 10, 0))
        guidance.pack(fill="x")
        ttk.Label(
            guidance,
            text=(
                "Look for a stretch of resolutions where the cluster count stops changing and "
                "the ARI vs. the previous resolution stays close to 1.0 - that plateau is a "
                "resolution whose split reflects real, stable structure rather than an arbitrary "
                "cut. This is read-only; nothing here has been applied to the node."
            ),
            style="Panel.TLabel", wraplength=720, justify="left",
        ).pack(anchor="w")

        table_frame = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        table_frame.pack(fill="x")
        columns = ("resolution", "n_clusters", "ari")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=min(len(results), 8))
        tree.heading("resolution", text="Resolution")
        tree.heading("n_clusters", text="Clusters")
        tree.heading("ari", text="ARI vs previous")
        tree.column("resolution", width=100, anchor="center")
        tree.column("n_clusters", width=100, anchor="center")
        tree.column("ari", width=140, anchor="center")
        for row in results:
            ari_text = f"{row['ari_vs_previous']:.3f}" if row["ari_vs_previous"] is not None else "-"
            tree.insert("", "end", values=(row["resolution"], row["n_clusters"], ari_text))
        tree.pack(fill="x")

        plot_frame = ttk.Frame(popup, style="Panel.TFrame", padding=10)
        plot_frame.pack(fill="both", expand=True)

        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(plot_frame, text="Matplotlib is not installed; showing the table only.", style="Panel.TLabel").pack(anchor="w")
            return

        resolutions = [row["resolution"] for row in results]
        cluster_counts = [row["n_clusters"] for row in results]
        aris = [row["ari_vs_previous"] for row in results]

        fig = Figure(figsize=(7.2, 4.2), dpi=150, facecolor="white")
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(resolutions, cluster_counts, marker="o", color="#1D9E75", label="Number of clusters")
        ax1.set_xlabel("Resolution", fontsize=9, color="black")
        ax1.set_ylabel("Number of clusters", fontsize=9, color="#1D9E75")
        ax1.tick_params(axis="y", labelcolor="#1D9E75", labelsize=8)
        ax1.tick_params(axis="x", labelsize=8, colors="black")

        ax2 = ax1.twinx()
        # First resolution has no "previous" to compare against - skip it so the ARI line only
        # spans resolutions that actually have a value.
        ari_resolutions = [r for r, a in zip(resolutions, aris) if a is not None]
        ari_values = [a for a in aris if a is not None]
        line2, = ax2.plot(ari_resolutions, ari_values, marker="s", linestyle="--", color="#378ADD", label="ARI vs previous")
        ax2.set_ylabel("ARI vs previous resolution", fontsize=9, color="#378ADD")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="#378ADD", labelsize=8)

        ax1.set_title(f"Resolution sweep: {node.label}", fontsize=10, color="black")
        for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
            spine.set_color("black")
        ax1.legend(handles=[line1, line2], fontsize=7, loc="upper left")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        canvas.get_tk_widget().pack(expand=True, fill="both")
        self._attach_plot_toolbar(canvas, plot_frame)

    def open_bbknn_neighbors_sweep_dialog(self):
        """Tools > BBKNN Neighbors Sweep...: try several BBKNN `neighbors_within_batch` values on the
        selected node, rebuilding the graph from scratch for each one (unlike the Resolution
        Sweep, which reuses one graph) since this parameter changes the graph itself, not just
        how it's cut. For each value, reports how well batches actually mixed (not just what was
        asked for), whether the resulting graph is fully connected, and how many Leiden clusters
        it produces at a fixed resolution - plus the Adjusted Rand Index between each value's
        clustering and the previous value's, so you can see whether the final clustering answer
        is sensitive to this choice. This is noticeably heavier than the Resolution Sweep: every
        value re-runs the full normalize/HVG/scale/PCA/BBKNN pipeline, not just Leiden on an
        already-built graph, so a confirmation with the value count is shown before running."""
        if self.busy:
            return
        if not self.selected_node_id:
            messagebox.showwarning("No selection", "Select a node first.")
            return
        node = self.model.nodes[self.selected_node_id]

        default_neighbor_values = "3, 5, 8, 10, 15, 20, 25"
        neighbors_text = simpledialog.askstring(
            "Neighbors Sweep",
            f"Neighbors/Batch values to try on '{node.label}' ({node.adata.n_obs} cells), comma-separated:",
            initialvalue=default_neighbor_values,
            parent=self,
        )
        if neighbors_text is None:
            return
        try:
            neighbor_values = sorted({int(part.strip()) for part in neighbors_text.split(",") if part.strip()})
            if not neighbor_values or any(v < 1 for v in neighbor_values):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a comma-separated list of whole numbers >= 1, e.g. 3, 5, 10, 15")
            return

        default_resolution = self.resolution_var.get().strip() or "0.8"
        resolution_text = simpledialog.askstring(
            "Neighbors Sweep",
            "Fixed resolution to use for comparing cluster counts across neighbor values:",
            initialvalue=default_resolution,
            parent=self,
        )
        if resolution_text is None:
            return
        try:
            resolution = float(resolution_text.strip())
        except ValueError:
            messagebox.showerror("Invalid resolution", f"'{resolution_text}' is not a valid resolution number.")
            return

        if not messagebox.askyesno(
            "Neighbors Sweep",
            f"This will rebuild the full batch-correction pipeline (normalize, HVG, PCA, BBKNN) "
            f"from scratch for each of {len(neighbor_values)} neighbor value(s): "
            f"{', '.join(str(v) for v in neighbor_values)}.\n\n"
            f"This is significantly slower than the Resolution Sweep - continue?",
        ):
            return

        try:
            self.set_busy(True, f"Running neighbors sweep on {node.label}...")
            # Sync the toolbar's "Batch Key" textbox into the service before running - every
            # other entry point that calls into batch correction (Process, the legacy row
            # buttons, session loading) does this same sync; without it here, the sweep would
            # silently keep whatever batch key the service last had (its own default is the
            # literal string "Batch") instead of whatever the user actually typed in the box.
            self.service.batch_key = self.batch_key_var.get().strip()
            self.append_log(
                f"=== Neighbors sweep requested for node {node.label} ({node.node_id}): "
                f"batch key='{self.service.batch_key}', values={neighbor_values}, fixed resolution={resolution} ==="
            )
            results = self.service.run_bbknn_neighbors_sweep(node, neighbor_values, resolution, logger=self.append_log)
            self._show_bbknn_neighbors_sweep_results(node, results, resolution)
            self.set_status(f"Neighbors sweep complete for {node.label} ({len(neighbor_values)} values tried).")
        except ImportError as exc:
            messagebox.showerror("Neighbors Sweep", str(exc))
            self.set_status("Neighbors sweep failed: missing dependency.")
        except Exception as exc:
            self.append_log("Neighbors sweep failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Neighbors sweep error", str(exc))
            self.set_status("Neighbors sweep failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def _show_bbknn_neighbors_sweep_results(self, node: ClusterNode, results: List[Dict[str, object]], resolution: float):
        """Popup showing the neighbors sweep results as a table plus a dual-axis chart: two
        mixing measures (mean cross-batch neighbor fraction, and normalized effective batches
        per neighborhood) on the left axis, cluster count at the fixed resolution on the right,
        both against neighbors_within_batch. Rows where the graph came out fragmented (connected
        components > 1) or where the value had to be skipped entirely are called out
        separately, since those are correctness problems no amount of resolution tuning can
        fix."""
        popup = tk.Toplevel(self)
        popup.title(f"Neighbors Sweep - {node.label}")
        popup.geometry("820x680")
        popup.configure(bg=BG_PANEL)

        header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        header.pack(fill="x")
        ttk.Label(
            header, text=f"Neighbors sweep for '{node.label}' ({node.adata.n_obs} cells), fixed resolution={resolution}",
            style="Panel.TLabel",
        ).pack(side="left")

        guidance = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 6, 10, 0))
        guidance.pack(fill="x")
        ttk.Label(
            guidance,
            text=(
                "Cross-batch fraction just asks 'same batch as me or not' per neighbor, so it "
                "can look high even if all a cell's cross-batch neighbors come from just one "
                "other batch. Effective batches (an inverse-Simpson/iLISI-style measure) checks "
                "HOW those neighbors are spread across batches - close to the total batch count "
                "means neighborhoods draw broadly from every batch, not just one or two. "
                "Connected components should be 1. High ARI vs the previous value means the "
                "final clustering isn't sensitive to this choice. Read-only; nothing here has "
                "been applied to the node."
            ),
            style="Panel.TLabel", wraplength=780, justify="left",
        ).pack(anchor="w")

        skipped = [row for row in results if row.get("skipped_reason")]
        if skipped:
            warn_frame = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 6, 10, 0))
            warn_frame.pack(fill="x")
            skipped_text = "; ".join(f"{row['neighbors_within_batch']}: {row['skipped_reason']}" for row in skipped)
            ttk.Label(
                warn_frame, text=f"Skipped value(s) - {skipped_text}", style="Panel.TLabel",
                wraplength=780, justify="left", foreground="#E24B4A",
            ).pack(anchor="w")

        table_frame = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        table_frame.pack(fill="x")
        columns = ("neighbors", "mixing", "effective", "components", "n_clusters", "ari")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=min(len(results), 8))
        tree.heading("neighbors", text="Neighbors/Batch")
        tree.heading("mixing", text="Cross-batch fraction")
        tree.heading("effective", text="Effective batches")
        tree.heading("components", text="Connected components")
        tree.heading("n_clusters", text="Clusters")
        tree.heading("ari", text="ARI vs previous")
        tree.column("neighbors", width=100, anchor="center")
        tree.column("mixing", width=130, anchor="center")
        tree.column("effective", width=120, anchor="center")
        tree.column("components", width=140, anchor="center")
        tree.column("n_clusters", width=80, anchor="center")
        tree.column("ari", width=130, anchor="center")
        for row in results:
            if row.get("skipped_reason"):
                tree.insert("", "end", values=(row["neighbors_within_batch"], "skipped", "skipped", "skipped", "skipped", "skipped"))
                continue
            mixing_text = f"{row['mean_cross_batch_fraction']:.3f}"
            effective_text = f"{row['mean_effective_batches']:.2f} / {row['n_batches_total']}"
            components_text = str(row["n_connected_components"])
            ari_text = f"{row['ari_vs_previous']:.3f}" if row["ari_vs_previous"] is not None else "-"
            tree.insert("", "end", values=(row["neighbors_within_batch"], mixing_text, effective_text, components_text, row["n_clusters"], ari_text))
        tree.pack(fill="x")

        plot_frame = ttk.Frame(popup, style="Panel.TFrame", padding=10)
        plot_frame.pack(fill="both", expand=True)

        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(plot_frame, text="Matplotlib is not installed; showing the table only.", style="Panel.TLabel").pack(anchor="w")
            return

        valid_rows = [row for row in results if not row.get("skipped_reason")]
        if not valid_rows:
            ttk.Label(plot_frame, text="Every value was skipped; no chart to show.", style="Panel.TLabel").pack(anchor="w")
            return

        neighbor_values = [row["neighbors_within_batch"] for row in valid_rows]
        mixing_fractions = [row["mean_cross_batch_fraction"] for row in valid_rows]
        # Normalize effective batches by the total batch count so it sits on the same 0-1 scale
        # as the cross-batch fraction and can share the left axis.
        normalized_effective = [row["mean_effective_batches"] / row["n_batches_total"] for row in valid_rows]
        cluster_counts = [row["n_clusters"] for row in valid_rows]

        fig = Figure(figsize=(7.4, 4.4), dpi=150, facecolor="white")
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(neighbor_values, mixing_fractions, marker="o", color="#378ADD", label="Mean cross-batch fraction")
        line3, = ax1.plot(neighbor_values, normalized_effective, marker="^", linestyle=":", color="#993C1D", label="Effective batches / total")
        ax1.set_xlabel("Neighbors/Batch", fontsize=9, color="black")
        ax1.set_ylabel("Fraction (0-1)", fontsize=9, color="black")
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis="y", labelsize=8, colors="black")
        ax1.tick_params(axis="x", labelsize=8, colors="black")

        ax2 = ax1.twinx()
        line2, = ax2.plot(neighbor_values, cluster_counts, marker="s", linestyle="--", color="#1D9E75", label=f"Clusters at resolution={resolution}")
        ax2.set_ylabel("Number of clusters", fontsize=9, color="#1D9E75")
        ax2.tick_params(axis="y", labelcolor="#1D9E75", labelsize=8)

        ax1.set_title(f"BBKNN neighbors sweep: {node.label}", fontsize=10, color="black")
        for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
            spine.set_color("black")
        ax1.legend(handles=[line1, line3, line2], fontsize=7, loc="upper left")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        canvas.get_tk_widget().pack(expand=True, fill="both")
        self._attach_plot_toolbar(canvas, plot_frame)

    def open_scvi_neighbors_sweep_dialog(self):
        """Tools > SCVI Neighbors Sweep...: scVI counterpart to the BBKNN Neighbors Sweep - try
        several `n_neighbors` values on the selected node's scVI latent embedding
        (obsm['X_scVI']), rebuilding only the neighbor graph itself for each value rather than
        retraining scVI (training is the expensive step and n_neighbors doesn't affect it, so
        this is much cheaper per value than the BBKNN sweep, which must rebuild its whole
        pipeline from scratch each time). Reports the same diagnostics as the BBKNN sweep, but
        the cross-batch mixing fraction is a more genuine signal here - scVI's neighbor graph has
        no forced per-batch quota the way BBKNN's does, so a high fraction means scVI actually
        learned to place different batches near each other, not just an artifact of how the
        graph was constructed (see run_scvi_neighbors_sweep's docstring for the full reasoning)."""
        if self.busy:
            return
        if not self.selected_node_id:
            messagebox.showwarning("No selection", "Select a node first.")
            return
        node = self.model.nodes[self.selected_node_id]
        # Prefer a fresh, not-yet-clustered scVI correction result if one is sitting on the
        # node; otherwise fall back to the scVI latent embedding baked into its last clustering
        # result. Both carry obsm['X_scVI'] the sweep needs, without requiring scVI to be
        # retrained here.
        graph_adata = node.integration_adata if node.integration_adata is not None else node.umap_adata
        if graph_adata is None or "X_scVI" not in graph_adata.obsm:
            messagebox.showerror(
                "SCVI Neighbors Sweep",
                "This node needs an scVI latent embedding first. Select 'SCVI' in the 'Batch "
                "Correction' dropdown above and click Process (or Batch Correction + Clustering "
                "together), then try the sweep again.",
            )
            return

        default_neighbor_values = "5, 10, 15, 20, 30, 50"
        neighbors_text = simpledialog.askstring(
            "SCVI Neighbors Sweep",
            f"n_neighbors values to try on '{node.label}' ({graph_adata.n_obs} cells), comma-separated:",
            initialvalue=default_neighbor_values,
            parent=self,
        )
        if neighbors_text is None:
            return
        try:
            neighbor_values = sorted({int(part.strip()) for part in neighbors_text.split(",") if part.strip()})
            if not neighbor_values or any(v < 1 for v in neighbor_values):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a comma-separated list of whole numbers >= 1, e.g. 5, 10, 15, 30")
            return

        default_resolution = self.resolution_var.get().strip() or "0.8"
        resolution_text = simpledialog.askstring(
            "SCVI Neighbors Sweep",
            "Fixed resolution to use for comparing cluster counts across n_neighbors values:",
            initialvalue=default_resolution,
            parent=self,
        )
        if resolution_text is None:
            return
        try:
            resolution = float(resolution_text.strip())
        except ValueError:
            messagebox.showerror("Invalid resolution", f"'{resolution_text}' is not a valid resolution number.")
            return

        try:
            self.set_busy(True, f"Running SCVI neighbors sweep on {node.label}...")
            # Sync the toolbar's "Batch Key" textbox into the service before running - the
            # mixing-fraction calculation inside run_scvi_neighbors_sweep reads self.batch_key
            # directly, and without this sync it would silently keep whatever batch key the
            # service last had instead of whatever's actually in the box (the same bug fixed
            # earlier for the BBKNN sweep - this dialog needs the identical fix since it never
            # goes through run_bbknn_correction to pick it up automatically).
            self.service.batch_key = self.batch_key_var.get().strip()
            self.append_log(
                f"=== SCVI neighbors sweep requested for node {node.label} ({node.node_id}): "
                f"batch key='{self.service.batch_key}', values={neighbor_values}, fixed resolution={resolution} ==="
            )
            results = self.service.run_scvi_neighbors_sweep(node, graph_adata, neighbor_values, resolution, logger=self.append_log)
            self._show_scvi_neighbors_sweep_results(node, results, resolution)
            self.set_status(f"SCVI neighbors sweep complete for {node.label} ({len(neighbor_values)} values tried).")
        except ImportError as exc:
            messagebox.showerror("SCVI Neighbors Sweep", str(exc))
            self.set_status("SCVI neighbors sweep failed: missing dependency.")
        except Exception as exc:
            self.append_log("SCVI neighbors sweep failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("SCVI neighbors sweep error", str(exc))
            self.set_status("SCVI neighbors sweep failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def _show_scvi_neighbors_sweep_results(self, node: ClusterNode, results: List[Dict[str, object]], resolution: float):
        """Popup showing the scVI neighbors sweep results as a table plus a dual-axis chart: two
        mixing measures (mean cross-batch neighbor fraction, and normalized effective batches
        per neighborhood) on the left axis, cluster count at the fixed resolution on the right,
        both against n_neighbors - same layout as the BBKNN sweep's popup, but labeled for
        n_neighbors rather than neighbors_within_batch."""
        popup = tk.Toplevel(self)
        popup.title(f"SCVI Neighbors Sweep - {node.label}")
        popup.geometry("820x680")
        popup.configure(bg=BG_PANEL)

        header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        header.pack(fill="x")
        ttk.Label(
            header, text=f"SCVI neighbors sweep for '{node.label}' ({node.adata.n_obs} cells), fixed resolution={resolution}",
            style="Panel.TLabel",
        ).pack(side="left")

        guidance = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 6, 10, 0))
        guidance.pack(fill="x")
        ttk.Label(
            guidance,
            text=(
                "Unlike the BBKNN sweep, scVI's neighbor graph has no forced per-batch quota, so "
                "both mixing measures here are genuine signals. Cross-batch fraction just asks "
                "'same batch as me or not' per neighbor, so it can look high even if all a "
                "cell's cross-batch neighbors come from just one other batch. Effective batches "
                "(an inverse-Simpson/iLISI-style measure) checks HOW those neighbors are spread "
                "across batches - close to the total batch count means neighborhoods draw "
                "broadly from every batch, not just one or two. Connected components should stay "
                "at 1. High ARI vs the previous value means the final clustering isn't sensitive "
                "to this choice. Read-only; nothing here has been applied to the node."
            ),
            style="Panel.TLabel", wraplength=780, justify="left",
        ).pack(anchor="w")

        table_frame = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        table_frame.pack(fill="x")
        columns = ("neighbors", "mixing", "effective", "components", "n_clusters", "ari")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=min(len(results), 8))
        tree.heading("neighbors", text="n_neighbors")
        tree.heading("mixing", text="Cross-batch fraction")
        tree.heading("effective", text="Effective batches")
        tree.heading("components", text="Connected components")
        tree.heading("n_clusters", text="Clusters")
        tree.heading("ari", text="ARI vs previous")
        tree.column("neighbors", width=100, anchor="center")
        tree.column("mixing", width=130, anchor="center")
        tree.column("effective", width=120, anchor="center")
        tree.column("components", width=140, anchor="center")
        tree.column("n_clusters", width=80, anchor="center")
        tree.column("ari", width=130, anchor="center")
        for row in results:
            mixing_text = f"{row['mean_cross_batch_fraction']:.3f}"
            effective_text = f"{row['mean_effective_batches']:.2f} / {row['n_batches_total']}"
            components_text = str(row["n_connected_components"])
            ari_text = f"{row['ari_vs_previous']:.3f}" if row["ari_vs_previous"] is not None else "-"
            tree.insert("", "end", values=(row["n_neighbors"], mixing_text, effective_text, components_text, row["n_clusters"], ari_text))
        tree.pack(fill="x")

        plot_frame = ttk.Frame(popup, style="Panel.TFrame", padding=10)
        plot_frame.pack(fill="both", expand=True)

        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(plot_frame, text="Matplotlib is not installed; showing the table only.", style="Panel.TLabel").pack(anchor="w")
            return

        neighbor_values = [row["n_neighbors"] for row in results]
        mixing_fractions = [row["mean_cross_batch_fraction"] for row in results]
        normalized_effective = [row["mean_effective_batches"] / row["n_batches_total"] for row in results]
        cluster_counts = [row["n_clusters"] for row in results]

        fig = Figure(figsize=(7.4, 4.4), dpi=150, facecolor="white")
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(neighbor_values, mixing_fractions, marker="o", color="#378ADD", label="Mean cross-batch fraction")
        line3, = ax1.plot(neighbor_values, normalized_effective, marker="^", linestyle=":", color="#993C1D", label="Effective batches / total")
        ax1.set_xlabel("n_neighbors", fontsize=9, color="black")
        ax1.set_ylabel("Fraction (0-1)", fontsize=9, color="black")
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis="y", labelsize=8, colors="black")
        ax1.tick_params(axis="x", labelsize=8, colors="black")

        ax2 = ax1.twinx()
        line2, = ax2.plot(neighbor_values, cluster_counts, marker="s", linestyle="--", color="#1D9E75", label=f"Clusters at resolution={resolution}")
        ax2.set_ylabel("Number of clusters", fontsize=9, color="#1D9E75")
        ax2.tick_params(axis="y", labelcolor="#1D9E75", labelsize=8)

        ax1.set_title(f"SCVI neighbors sweep: {node.label}", fontsize=10, color="black")
        for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
            spine.set_color("black")
        ax1.legend(handles=[line1, line3, line2], fontsize=7, loc="upper left")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        canvas.get_tk_widget().pack(expand=True, fill="both")
        self._attach_plot_toolbar(canvas, plot_frame)

    def _mark_stage_status(self, node: ClusterNode, text: str):
        node.last_stage_status = f"{text} {datetime.now().strftime('%H:%M:%S')}"

    def run_bbknn_for_node(self, node_id: str):
        """Row button: run BBKNN batch correction (preprocessing + BBKNN) on this node, storing
        the result on the node so the Leiden button can cluster it next."""
        self.select_node(node_id)
        if self.busy:
            return
        node = self.model.nodes.get(node_id)
        if node is None:
            return
        neighbors_within_batch_text = self.neighbors_within_batch_var.get().strip()
        if not neighbors_within_batch_text:
            messagebox.showerror("Invalid neighbors/batch", "Neighbors/batch cannot be blank. Enter a whole number (e.g. 3).")
            return
        try:
            neighbors_within_batch = int(neighbors_within_batch_text)
            if neighbors_within_batch < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid neighbors/batch", f"'{neighbors_within_batch_text}' is not a valid whole number >= 1.")
            return
        try:
            self.set_busy(True, f"Running BBKNN correction on {node.label}...")
            self.append_log(f"=== BBKNN correction requested for node {node.label} ({node.node_id}) ===")
            self.service.batch_key = self.batch_key_var.get().strip()
            self.append_log(f"Using batch key '{self.service.batch_key}', neighbors/batch {neighbors_within_batch}")
            corrected = self.service.run_bbknn_correction(node, neighbors_within_batch=neighbors_within_batch, logger=self.append_log)
            node.integration_adata = corrected
            node.integration_method = "bbknn"
            node.child_neighbors_within_batch = neighbors_within_batch
            self._mark_stage_status(node, "BBKNN done")
            self.append_log(f"BBKNN correction complete for {node.label}; ready for Leiden clustering ({corrected.n_obs} cells).")
            self.set_status(f"BBKNN correction complete for {node.label}. Click Leiden to create subclusters.")
            self.refresh_node_list()
        except Exception as exc:
            self._mark_stage_status(node, "BBKNN failed")
            self.append_log("BBKNN correction failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("BBKNN correction error", str(exc))
            self.set_status("BBKNN correction failed.")
            self.refresh_node_list()
        finally:
            self.set_busy(False, self.status_var.get())

    def run_scvi_for_node(self, node_id: str):
        """Row button: run scVI batch correction on this node. run_scvi_correction() is a
        placeholder - fill in your own scVI code there; this handler wires it up so the result
        (once implemented) flows into the same Leiden step BBKNN uses."""
        self.select_node(node_id)
        if self.busy:
            return
        node = self.model.nodes.get(node_id)
        if node is None:
            return
        neighbors_within_batch_text = self.neighbors_within_batch_var.get().strip()
        if not neighbors_within_batch_text:
            messagebox.showerror("Invalid neighbors/batch", "Neighbors/batch cannot be blank. Enter a whole number (e.g. 25).")
            return
        try:
            neighbors_within_batch = int(neighbors_within_batch_text)
            if neighbors_within_batch < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid neighbors/batch", f"'{neighbors_within_batch_text}' is not a valid whole number >= 1.")
            return
        max_epochs = None
        max_epochs_text = self.max_epochs_var.get().strip()
        if max_epochs_text:
            try:
                max_epochs = int(max_epochs_text)
                if max_epochs < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid max epochs", f"'{max_epochs_text}' is not a valid whole number >= 1. Leave blank to use scvi-tools' automatic default.")
                return
        try:
            self.set_busy(True, f"Running SCVI correction on {node.label}...")
            self.append_log(f"=== SCVI correction requested for node {node.label} ({node.node_id}) ===")
            corrected = self.service.run_scvi_correction(node, max_epochs=max_epochs, neighbors_within_batch=neighbors_within_batch, logger=self.append_log)
            node.integration_adata = corrected
            node.integration_method = "scvi"
            node.child_neighbors_within_batch = neighbors_within_batch
            self._mark_stage_status(node, "SCVI done")
            self.append_log(f"SCVI correction complete for {node.label}; ready for Leiden clustering ({corrected.n_obs} cells).")
            self.set_status(f"SCVI correction complete for {node.label}. Click Leiden to create subclusters.")
            self.refresh_node_list()
        except NotImplementedError as exc:
            self._mark_stage_status(node, "SCVI not implemented")
            self.append_log(f"SCVI correction is not implemented yet: {exc}")
            messagebox.showinfo(
                "SCVI Correction",
                "SCVI correction is a placeholder - implement your own code in "
                "H5ADClusterService.run_scvi_correction().",
            )
            self.set_status("SCVI correction is not implemented yet.")
            self.refresh_node_list()
        except Exception as exc:
            self._mark_stage_status(node, "SCVI failed")
            self.append_log("SCVI correction failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("SCVI correction error", str(exc))
            self.set_status("SCVI correction failed.")
            self.refresh_node_list()
        finally:
            self.set_busy(False, self.status_var.get())

    def run_leiden_for_node(self, node_id: str):
        """Row button: run UMAP + Leiden clustering on this node's BBKNN/SCVI-corrected data
        (from run_bbknn_for_node / run_scvi_for_node) and partition it into child nodes."""
        self.select_node(node_id)
        if self.busy:
            return
        node = self.model.nodes.get(node_id)
        if node is None:
            return
        integration_adata = node.integration_adata
        if integration_adata is None:
            integration_adata = node.adata
            self.append_log(
                f"No correction result recorded for {node.label} in this session - using its own "
                f"data as-is for clustering (assuming it was already batch-corrected)."
            )
        resolution_text = self.resolution_var.get().strip()
        if not resolution_text:
            messagebox.showerror("Invalid resolution", "Resolution cannot be blank. Enter a numeric value (e.g. 0.5).")
            return
        try:
            resolution = float(resolution_text)
        except ValueError:
            messagebox.showerror("Invalid resolution", f"'{resolution_text}' is not a valid resolution number.")
            return
        if node.children:
            old_resolution = node.child_resolution
            descendant_count = self.model.count_descendants(node.node_id)
            confirm_message = (
                f"This node already has children created with resolution {old_resolution}. "
                f"Re-clustering with resolution {resolution} will delete {descendant_count} existing node(s) "
                f"(its children and all their descendants) and replace them. Continue?"
            )
            if not messagebox.askyesno("Already clustered", confirm_message):
                return
        try:
            self.set_busy(True, f"Running Leiden clustering on {node.label}...")
            self.append_log(f"=== Leiden clustering requested for node {node.label} ({node.node_id}) via {node.integration_method or 'unknown'} correction ===")
            umap_kwargs = {"umap_min_dist": 0.1, "umap_spread": 1.0} if node.integration_method == "scvi" else {}
            result = self.service.run_leiden_clustering(node, integration_adata, resolution=resolution, logger=self.append_log, **umap_kwargs)
            new_cluster_count = len(result["children"])
            if new_cluster_count <= 1:
                reason = (
                    f"Leiden clustering '{node.label}' at resolution {resolution} found only {new_cluster_count} "
                    f"group(s). This node's cells already look homogeneous at this resolution, so no new "
                    f"subclusters were created \u2014 a single subcluster would just duplicate the parent "
                    f"node and add no information.\n\nTry raising the resolution, or check whether this "
                    f"node has enough distinct cells to justify splitting further."
                )
                self._mark_stage_status(node, "Leiden found 1 group (no split)")
                self.append_log(f"Skipped subcluster creation for '{node.label}': {reason}")
                messagebox.showinfo("Cannot subcluster further", reason)
                self.set_status(f"No new subclusters created for {node.label} (only {new_cluster_count} group found).")
                self.refresh_node_list()
                return
            if node.children:
                removed_count = len(node.children)
                self.append_log(
                    f"Removing {removed_count} old child branch(es) created with resolution {node.child_resolution} "
                    f"before attaching new resolution {resolution} results."
                )
                self.model.remove_children(node.node_id)
                if self.selected_node_id not in self.model.nodes:
                    self.selected_node_id = node.node_id
            node.umap_adata = result["clustered_adata"]
            node.last_cluster_key = result["cluster_key"]
            node.child_resolution = resolution
            self.append_log(f"Stored clustered UMAP and labels in selected node using key {node.last_cluster_key}")
            for cluster_value, child_adata in result["children"].items():
                if node.parent_id is None:
                    label = str(cluster_value)
                else:
                    label = f"{node.label}.{cluster_value}"
                child_id = self.model.add_child(self.selected_node_id, child_adata, label=label, cluster_value=cluster_value, resolution=resolution, neighbors_within_batch=node.child_neighbors_within_batch)
                self.append_log(f"Attached child node {child_id} as {label}")
            # This node just gained children, so it's no longer a leaf - clear any manually
            # assigned cell type, and clear the consumed integration result so a future
            # BBKNN/SCVI + Leiden run on this node starts fresh.
            node.assigned_cell_type = None
            node.integration_adata = None
            node.integration_method = None
            self._mark_stage_status(node, f"Leiden done ({new_cluster_count} subclusters)")
            self.refresh_views()
            self._maybe_predict_cell_types(node)
            self._refresh_embedded_plots(node)
            self.set_status(f"Created {len(result['children'])} subclusters under {node.label} via Leiden.")
        except Exception as exc:
            self._mark_stage_status(node, "Leiden failed")
            self.append_log("Leiden clustering failed.")
            self.append_log(f"Exception type: {type(exc).__name__}")
            self.append_log(f"Exception detail: {str(exc)}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Leiden clustering error", str(exc))
            self.set_status("Leiden clustering failed.")
            self.refresh_node_list()
        finally:
            self.set_busy(False, self.status_var.get())

    def toggle_save_selected(self):
        if self.busy:
            return
        if not self.selected_node_id:
            messagebox.showwarning("No selection", "Select a node first.")
            return
        node = self.model.nodes[self.selected_node_id]
        self.model.mark_saved(self.selected_node_id, saved=not node.saved)
        self.refresh_views()
        state = "saved" if self.model.nodes[self.selected_node_id].saved else "unsaved"
        self.append_log(f"Node {node.label} ({node.node_id}) marked as {state}")
        self.set_status(f"Node {node.label} marked as {state}.")

    def export_selected_nodes(self):
        if self.busy:
            return
        nodes = self.model.iter_saved_nodes()
        if not nodes:
            messagebox.showwarning("Nothing to export", "Save at least one node first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.set_busy(True, "Exporting selected nodes...")
            self.append_log("=== Export requested ===")
            df = self.service.export_saved_nodes(nodes, path, logger=self.append_log)
            self.set_status(f"Exported {len(df)} rows from {len(nodes)} saved nodes to {path}.")
            messagebox.showinfo("Export complete", f"Saved {len(df)} rows to:\n{path}")
        except Exception as exc:
            self.append_log(f"Export failed: {exc}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Export error", str(exc))
            self.set_status("Export failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def _collect_leaf_cell_type_series(self) -> pd.Series:
        """Walk every leaf in the current cluster tree and build a full-population pandas Series
        (indexed by cell barcode) of assigned cell types, using the same 'Not assigned' label
        shown in the Cell Type combo box for any leaf that hasn't been given one yet. Leaves
        fully partition the root's cells (every further round of clustering only splits a leaf
        into new leaves), so walking to leaves covers the whole dataset with no gaps."""
        if not self.model.root_id:
            raise ValueError("No h5ad is loaded.")
        barcodes: List[str] = []
        labels: List[str] = []
        leaf_count = 0
        assigned_leaf_count = 0

        def walk(node_id: str):
            nonlocal leaf_count, assigned_leaf_count
            node = self.model.nodes.get(node_id)
            if node is None:
                # Should not happen (a parent's children list referencing a node id that's no
                # longer in the model), but if it ever does, log it loudly rather than silently
                # dropping those cells from the export with no trace of why.
                self.append_log(f"[Collect cell types] Child node id '{node_id}' referenced but not found in the model - its cells are excluded from this export.")
                return
            if node.children:
                for child_id in node.children:
                    walk(child_id)
                return
            leaf_count += 1
            source = node.adata if node.adata is not None else node.umap_adata
            if source is None or source.n_obs == 0:
                self.append_log(f"[Collect cell types] Leaf '{node.label}' ({node.node_id}) has no data attached - its cells are excluded from this export.")
                return
            label = node.assigned_cell_type if node.assigned_cell_type else NOT_ASSIGNED_CELL_TYPE_LABEL
            if node.assigned_cell_type:
                assigned_leaf_count += 1
            barcodes.extend(source.obs_names.tolist())
            labels.extend([label] * source.n_obs)

        walk(self.model.root_id)
        if not barcodes:
            raise ValueError("No cells were found under the current cluster tree.")
        self.append_log(
            f"[Collect cell types] Walked {leaf_count} leaf node(s) covering {len(barcodes)} cell(s): "
            f"{assigned_leaf_count} leaf/leaves had a cell type assigned, "
            f"{leaf_count - assigned_leaf_count} still '{NOT_ASSIGNED_CELL_TYPE_LABEL}'."
        )
        series = pd.Series(labels, index=barcodes, name="cell_type")
        duplicate_count = int(series.index.duplicated().sum())
        if duplicate_count:
            # Two leaves claiming the same barcode means a stale node (e.g. left over from a
            # re-clustering run that didn't fully clean up) is still being walked alongside its
            # replacement. Rather than let this silently corrupt the export (or raise deep inside
            # a later reindex), keep only the last-walked assignment per barcode and say so loudly.
            self.append_log(
                f"[Collect cell types] WARNING: {duplicate_count} cell barcode(s) were claimed by more "
                f"than one leaf (likely a stale node from an earlier clustering run still present in "
                f"the tree) - keeping only the last one walked for each. If assignments you made "
                f"aren't showing up in a save, this is worth checking - use Annotation > (session/tree "
                f"tools) to confirm no orphaned branches remain, or reload the session fresh."
            )
            series = series[~series.index.duplicated(keep="last")]
        return series

    def _find_unassigned_leaves(self) -> List[Dict[str, object]]:
        """Walk the current cluster tree and return every leaf that has no cell type assigned
        yet, with its label, node id, and cell count - used to warn (and let the user cancel)
        before an authoritative save, so an incomplete annotation is never written out silently."""
        if not self.model.root_id:
            return []
        unassigned: List[Dict[str, object]] = []

        def walk(node_id: str):
            node = self.model.nodes.get(node_id)
            if node is None:
                return
            if node.children:
                for child_id in node.children:
                    walk(child_id)
                return
            if node.assigned_cell_type:
                return
            source = node.adata if node.adata is not None else node.umap_adata
            n_cells = int(source.n_obs) if source is not None else 0
            unassigned.append({"label": node.label, "node_id": node.node_id, "n_cells": n_cells})

        walk(self.model.root_id)
        return unassigned

    def _confirm_unassigned_leaves_ok_to_save(self) -> bool:
        """Show a blocking confirmation if any leaf still has no cell type assigned. Returns True
        if it's fine to proceed (either everything is assigned, or the user explicitly chose to
        save anyway), False if the save should be aborted. Called before either save path writes
        anything, so an incomplete annotation is never exported without an explicit choice."""
        unassigned = self._find_unassigned_leaves()
        if not unassigned:
            return True
        total_cells = sum(int(u["n_cells"]) for u in unassigned)
        preview = ", ".join(str(u["label"]) for u in unassigned[:15])
        if len(unassigned) > 15:
            preview += f", and {len(unassigned) - 15} more"
        proceed = messagebox.askyesno(
            "Unassigned Cell Types",
            f"{len(unassigned)} leaf node(s) covering {total_cells} cell(s) have no cell type "
            f"assigned yet and would be saved as '{NOT_ASSIGNED_CELL_TYPE_LABEL}':\n\n{preview}\n\n"
            f"Continue saving anyway?",
        )
        if not proceed:
            self.append_log(
                f"Save cancelled: {len(unassigned)} leaf node(s) ({total_cells} cells) still have "
                f"no cell type assigned. Assign a type to every row in the Tree Nodes list first, "
                f"then save again."
            )
        return proceed

    def save_cell_type_annotations(self):
        """Collect the cell type assigned to every leaf (via the Tree Nodes list combo boxes),
        map it back onto every cell in the full h5ad object by barcode, and save the result as a
        new h5ad file - the original file on disk is never modified."""
        if self.busy:
            return
        if not self.model.root_id:
            messagebox.showwarning("Save Cell Type Annotations", "Load an h5ad file first.")
            return
        root_node = self.model.nodes[self.model.root_id]
        if root_node.adata is None:
            messagebox.showwarning("Save Cell Type Annotations", "The root node has no data loaded.")
            return

        column_name = simpledialog.askstring(
            "Cell Type Column Name",
            "Name for the new .obs column that will hold the cell type annotations:",
            initialvalue="cell_type",
            parent=self,
        )
        if column_name is None:
            return
        column_name = column_name.strip()
        if not column_name:
            messagebox.showwarning("Save Cell Type Annotations", "Column name cannot be blank.")
            return

        try:
            annotation_series = self._collect_leaf_cell_type_series()
        except Exception as exc:
            messagebox.showerror("Save Cell Type Annotations", f"Could not collect cell type annotations:\n{exc}")
            return

        if not self._confirm_unassigned_leaves_ok_to_save():
            return

        path = filedialog.asksaveasfilename(
            title="Save Annotated H5AD As",
            defaultextension=".h5ad",
            initialfile="annotated.h5ad",
            filetypes=[("H5AD files", "*.h5ad"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.set_busy(True, "Saving annotated h5ad...")
            annotated = root_node.adata.copy()
            mapped = annotation_series.reindex(annotated.obs_names)
            missing = int(mapped.isna().sum())
            mapped = mapped.fillna(NOT_ASSIGNED_CELL_TYPE_LABEL)
            annotated.obs[column_name] = pd.Categorical(mapped.astype(str))
            annotated.write_h5ad(path)
            if missing:
                self.append_log(
                    f"Note: {missing} cell(s) in the full h5ad had no matching leaf under the "
                    f"current tree and were marked '{NOT_ASSIGNED_CELL_TYPE_LABEL}'."
                )
            self.append_log(f"Saved annotated h5ad (column '{column_name}') to {path}")
            self.set_status(f"Saved annotated h5ad to {os.path.basename(path)}.")
            messagebox.showinfo("Save Cell Type Annotations", f"Saved annotated h5ad to:\n{path}")
        except Exception as exc:
            self.append_log(f"Failed to save annotated h5ad: {exc}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Save Cell Type Annotations", f"Could not save file:\n{exc}")
            self.set_status("Saving annotated h5ad failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def save_cell_type_annotations_csv(self):
        """Export current cell type assignments as a plain CSV covering the FULL loaded dataset -
        unlike most other actions in this app, this doesn't depend on which node is selected at
        all, since _collect_leaf_cell_type_series() always walks the whole tree from the root.
        Two columns: 'donor_barcode' (a chosen .obs column's value, concatenated with the cell
        barcode via an underscore) and 'cell_type'."""
        if self.busy:
            return
        if not self.model.root_id:
            messagebox.showwarning("Save Cell Type Annotations (CSV)", "Load an h5ad file first.")
            return
        root_node = self.model.nodes[self.model.root_id]
        if root_node.adata is None:
            messagebox.showwarning("Save Cell Type Annotations (CSV)", "The root node has no data loaded.")
            return

        default_donor_column = "donor" if "donor" in root_node.adata.obs.columns else self.batch_key_var.get().strip()
        donor_column = simpledialog.askstring(
            "Donor / Sample Column",
            "Name of the .obs column identifying each cell's donor/sample.\n"
            "The CSV's 'donor_barcode' column will be <this column's value>_<cell barcode>:",
            initialvalue=default_donor_column,
            parent=self,
        )
        if donor_column is None:
            return
        donor_column = donor_column.strip()
        if not donor_column:
            messagebox.showwarning("Save Cell Type Annotations (CSV)", "Column name cannot be blank.")
            return
        if donor_column not in root_node.adata.obs.columns:
            messagebox.showerror(
                "Save Cell Type Annotations (CSV)",
                f"'{donor_column}' was not found in adata.obs.\nAvailable columns: {list(root_node.adata.obs.columns)}",
            )
            return

        try:
            annotation_series = self._collect_leaf_cell_type_series()
        except Exception as exc:
            messagebox.showerror("Save Cell Type Annotations (CSV)", f"Could not collect cell type annotations:\n{exc}")
            return

        if not self._confirm_unassigned_leaves_ok_to_save():
            return

        path = filedialog.asksaveasfilename(
            title="Save Cell Type Annotations CSV As",
            defaultextension=".csv",
            initialfile="cell_type_annotations.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.set_busy(True, "Saving cell type annotations CSV...")
            all_barcodes = root_node.adata.obs_names
            mapped_types = annotation_series.reindex(all_barcodes)
            missing = int(mapped_types.isna().sum())
            mapped_types = mapped_types.fillna(NOT_ASSIGNED_CELL_TYPE_LABEL)
            donor_values = root_node.adata.obs[donor_column].astype(str)
            donor_barcode = donor_values + "_" + pd.Series(all_barcodes, index=all_barcodes).astype(str)
            out_df = pd.DataFrame({
                "donor_barcode": donor_barcode.to_numpy(),
                "cell_type": mapped_types.astype(str).to_numpy(),
            })
            out_df.to_csv(path, index=False)
            if missing:
                self.append_log(
                    f"Note: {missing} cell(s) in the full h5ad had no matching leaf under the "
                    f"current tree and were marked '{NOT_ASSIGNED_CELL_TYPE_LABEL}' in the CSV."
                )
            self.append_log(f"Saved cell type annotations CSV ({len(out_df)} cells, donor column '{donor_column}') to {path}")
            self.set_status(f"Saved cell type annotations CSV to {os.path.basename(path)}.")
            messagebox.showinfo("Save Cell Type Annotations (CSV)", f"Saved {len(out_df)} cell(s) to:\n{path}")
        except Exception as exc:
            self.append_log(f"Failed to save cell type annotations CSV: {exc}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Save Cell Type Annotations (CSV)", f"Could not save file:\n{exc}")
            self.set_status("Saving cell type annotations CSV failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def visualize_umap_by_annotation(self):
        """Show a UMAP (of the full loaded dataset) colored by the cell type annotations
        currently set in the Tree Nodes list combo boxes. If the h5ad object doesn't have a
        'cell_type' column yet, add and populate it in memory first (this does not touch the
        file on disk - use 'Save Cell Type Annotations...' for that)."""
        if self.busy:
            return
        if not self.model.root_id:
            messagebox.showwarning("Visualize UMAP by Annotation", "Load an h5ad file first.")
            return
        root_node = self.model.nodes[self.model.root_id]
        umap_source = root_node.umap_adata if root_node.umap_adata is not None else root_node.adata
        if umap_source is None or "X_umap" not in getattr(umap_source, "obsm", {}):
            messagebox.showwarning(
                "Visualize UMAP by Annotation",
                "No UMAP coordinates are available yet for the full dataset.\n\n"
                "Process the root node at least once (check 'Clustering' and click the root's "
                "'Process' button) to generate a UMAP embedding, then try this again.",
            )
            return

        column_name = "cell_type"
        if column_name not in umap_source.obs.columns:
            try:
                annotation_series = self._collect_leaf_cell_type_series()
            except Exception as exc:
                messagebox.showerror("Visualize UMAP by Annotation", f"Could not collect cell type annotations:\n{exc}")
                return
            mapped = annotation_series.reindex(umap_source.obs_names).fillna(NOT_ASSIGNED_CELL_TYPE_LABEL)
            umap_source.obs[column_name] = pd.Categorical(mapped.astype(str))
            self.append_log(f"Added in-memory '{column_name}' column to the loaded h5ad from the current combo box assignments.")
        else:
            self.append_log(f"'{column_name}' column already present on the h5ad object; using it as-is.")

        self._open_annotation_umap_popup(umap_source, column_name)

    def _open_annotation_umap_popup(self, adata, column_name: str):
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Visualize UMAP by Annotation", "Matplotlib is not installed.")
            return

        popup = getattr(self, "annotation_umap_popup_window", None)
        if popup is not None and popup.winfo_exists():
            popup.destroy()

        popup = tk.Toplevel(self)
        popup.title("UMAP by Cell Type Annotation")
        popup.geometry("1000x760")
        popup.configure(bg=BG_PANEL)
        self.annotation_umap_popup_window = popup

        header = ttk.Frame(popup, style="Panel.TFrame", padding=(10, 10, 10, 0))
        header.pack(fill="x")
        ttk.Label(header, text=f"UMAP colored by '{column_name}'", style="Panel.TLabel").pack(side="left")

        body = ttk.Frame(popup, style="Panel.TFrame", padding=10)
        body.pack(fill="both", expand=True)

        coords = np.asarray(adata.obsm["X_umap"])
        categories = adata.obs[column_name].astype(str)
        cat_values = categories.values
        unique_labels = sorted(categories.unique())
        color_lookup = {label: color for label, color in zip(unique_labels, self._qualitative_colors(len(unique_labels)))}

        # --- Left: a checkbox per cell type, to toggle which ones are plotted -----------------
        checklist_outer = ttk.Frame(body, style="Panel.TFrame")
        checklist_outer.pack(side="left", fill="y", padx=(0, 10))
        toggle_row = ttk.Frame(checklist_outer, style="Panel.TFrame")
        toggle_row.pack(fill="x", pady=(0, 4))
        ttk.Label(checklist_outer, text="Cell types shown:", style="Panel.TLabel").pack(anchor="w")

        type_vars: Dict[str, tk.BooleanVar] = {label: tk.BooleanVar(value=True) for label in unique_labels}

        checklist_canvas = tk.Canvas(checklist_outer, bg="white", highlightthickness=0, width=200)
        checklist_scroll = ttk.Scrollbar(checklist_outer, orient="vertical", command=checklist_canvas.yview)
        checklist_canvas.configure(yscrollcommand=checklist_scroll.set)
        checklist_canvas.pack(side="left", fill="y", expand=True)
        checklist_scroll.pack(side="left", fill="y")
        checklist_inner = ttk.Frame(checklist_canvas, style="Panel.TFrame")
        checklist_window = checklist_canvas.create_window((0, 0), window=checklist_inner, anchor="nw")
        checklist_inner.bind(
            "<Configure>",
            lambda _e: checklist_canvas.configure(scrollregion=checklist_canvas.bbox("all")),
        )
        checklist_canvas.bind(
            "<Configure>",
            lambda e: checklist_canvas.itemconfigure(checklist_window, width=e.width),
        )

        def redraw(*_args):
            ax.clear()
            shown = 0
            for label in unique_labels:
                if not type_vars[label].get():
                    continue
                mask = cat_values == label
                if not mask.any():
                    continue
                ax.scatter(coords[mask, 0], coords[mask, 1], s=6, color=[color_lookup[label]], label=label, linewidths=0)
                shown += 1
            ax.set_xlabel("UMAP1", fontsize=9, color="black")
            ax.set_ylabel("UMAP2", fontsize=9, color="black")
            ax.set_title(f"Cell type annotation ({shown} of {len(unique_labels)} categories shown)", fontsize=10, color="black")
            ax.tick_params(labelsize=7, colors="black")
            for spine in ax.spines.values():
                spine.set_color("black")
            if shown:
                ax.legend(fontsize=6, markerscale=2, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
            canvas.draw()

        for label in unique_labels:
            swatch_color = color_lookup[label]
            row = ttk.Frame(checklist_inner, style="Panel.TFrame")
            row.pack(fill="x", anchor="w")
            swatch = tk.Canvas(row, width=10, height=10, highlightthickness=0, bg=BG_PANEL)
            swatch.create_oval(1, 1, 9, 9, fill=swatch_color, outline=swatch_color)
            swatch.pack(side="left", padx=(2, 4), pady=2)
            ttk.Checkbutton(
                row, text=label, variable=type_vars[label], style="Card.TCheckbutton", command=redraw,
            ).pack(side="left", anchor="w")

        def _set_all(value: bool):
            for var in type_vars.values():
                var.set(value)
            redraw()

        ttk.Button(toggle_row, text="All", width=6, command=lambda: _set_all(True)).pack(side="left", padx=(0, 4))
        ttk.Button(toggle_row, text="None", width=6, command=lambda: _set_all(False)).pack(side="left")

        # --- Right: the plot itself -------------------------------------------------------------
        content_wrap = ttk.Frame(body, style="Panel.TFrame")
        content_wrap.pack(side="left", fill="both", expand=True)
        content_frame = self._make_scrollable_plot_area(content_wrap)

        fig = Figure(figsize=(8.0, 6.5), dpi=150, facecolor="white")
        ax = fig.add_subplot(111)
        fig.subplots_adjust(right=0.72)

        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        canvas.get_tk_widget().pack(expand=True, padx=4, pady=4)
        self._attach_plot_toolbar(canvas, content_frame)
        self.annotation_umap_figure = fig
        self.annotation_umap_canvas = canvas

        redraw()

    def _session_data_dir(self, json_path: str) -> str:
        base, _ext = os.path.splitext(json_path)
        return base + "_nodes"

    def save_session(self):
        if self.busy:
            return
        if not self.model.root_id:
            messagebox.showwarning("Nothing to save", "Load an h5ad file before saving a session.")
            return
        json_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Session JSON", "*.json"), ("All files", "*")],
            title="Save Session As",
        )
        if not json_path:
            return
        try:
            self.set_busy(True, "Saving session...")
            self.append_log("=== Save session requested ===")
            session_dir = self._session_data_dir(json_path)
            os.makedirs(session_dir, exist_ok=True)
            nodes_payload = []
            for node_id, node in self.model.nodes.items():
                h5ad_filename = f"{node_id}.h5ad"
                h5ad_path = os.path.join(session_dir, h5ad_filename)
                node.adata.write_h5ad(h5ad_path)
                self.append_log(f"Saved data matrix for node {node.label} ({node_id}) to {h5ad_path}")
                umap_h5ad_filename = None
                if node.umap_adata is not None:
                    umap_h5ad_filename = f"{node_id}_umap.h5ad"
                    umap_h5ad_path = os.path.join(session_dir, umap_h5ad_filename)
                    node.umap_adata.write_h5ad(umap_h5ad_path)
                    self.append_log(f"Saved UMAP preview matrix for node {node.label} ({node_id}) to {umap_h5ad_path}")
                nodes_payload.append({
                    "node_id": node.node_id,
                    "label": node.label,
                    "parent_id": node.parent_id,
                    "cluster_value": node.cluster_value,
                    "depth": node.depth,
                    "children": list(node.children),
                    "saved": node.saved,
                    "last_cluster_key": node.last_cluster_key,
                    "child_resolution": node.child_resolution,
                    "resolution": node.resolution,
                    "child_neighbors_within_batch": node.child_neighbors_within_batch,
                    "neighbors_within_batch": node.neighbors_within_batch,
                    "assigned_cell_type": node.assigned_cell_type,
                    "h5ad_file": h5ad_filename,
                    "umap_h5ad_file": umap_h5ad_filename,
                })

            selected_depth = 0
            if self.selected_node_id and self.selected_node_id in self.model.nodes:
                selected_depth = self.model.nodes[self.selected_node_id].depth

            ui_state = {
                "selected_node_id": self.selected_node_id,
                "current_clustering_level": selected_depth,
                "node_list_scroll_position": self.node_list_canvas.yview()[0],
                "available_cell_types": sorted(self.available_cell_types),
                "batch_key": self.batch_key_var.get(),
                "resolution": self.resolution_var.get(),
                "neighbors_within_batch": self.neighbors_within_batch_var.get(),
                "max_epochs": self.max_epochs_var.get(),
                "correction_method": self.correction_method_var.get(),
                "clustering_method": self.clustering_method_var.get(),
                "annotation_tool": self.annotation_tool_var.get(),
                "auto_assign_threshold": self.auto_assign_threshold_var.get(),
                "not_clear_threshold": self.not_clear_threshold_var.get(),
            }

            session_payload = {
                "root_id": self.model.root_id,
                "saved_node_ids": list(self.model.saved_node_ids),
                "nodes": nodes_payload,
                "ui_state": ui_state,
            }

            with open(json_path, "w") as f:
                json.dump(session_payload, f, indent=2)
            self.append_log(f"Session UI state and tree manifest written to {json_path}")
            self.set_status(f"Session saved to {json_path}.")
        except Exception as exc:
            self.append_log(f"Save session failed: {exc}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Save session error", str(exc))
            self.set_status("Save session failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def load_session(self):
        if self.busy:
            return
        json_path = filedialog.askopenfilename(
            filetypes=[("Session JSON", "*.json"), ("All files", "*")],
            title="Load Session",
        )
        if not json_path:
            return
        try:
            self.set_busy(True, "Loading session...")
            self.append_log("=== Load session requested ===")
            if sc is None:
                raise ImportError("scanpy is required to load .h5ad files. Install with: pip install scanpy")
            with open(json_path, "r") as f:
                session_payload = json.load(f)
            session_dir = self._session_data_dir(json_path)

            self.model.clear()
            for node_info in session_payload["nodes"]:
                h5ad_path = os.path.join(session_dir, node_info["h5ad_file"])
                self.append_log(f"Loading data matrix for node {node_info['label']} from {h5ad_path}")
                adata = sc.read_h5ad(h5ad_path)
                umap_adata = None
                umap_h5ad_filename = node_info.get("umap_h5ad_file")
                if umap_h5ad_filename:
                    umap_h5ad_path = os.path.join(session_dir, umap_h5ad_filename)
                    if os.path.exists(umap_h5ad_path):
                        self.append_log(f"Loading UMAP preview matrix for node {node_info['label']} from {umap_h5ad_path}")
                        umap_adata = sc.read_h5ad(umap_h5ad_path)
                node = ClusterNode(
                    node_id=node_info["node_id"],
                    label=node_info["label"],
                    adata=adata,
                    parent_id=node_info.get("parent_id"),
                    cluster_value=node_info.get("cluster_value"),
                    depth=node_info.get("depth", 0),
                    children=list(node_info.get("children", [])),
                    saved=node_info.get("saved", False),
                    last_cluster_key=node_info.get("last_cluster_key"),
                    child_resolution=node_info.get("child_resolution"),
                    resolution=node_info.get("resolution"),
                    child_neighbors_within_batch=node_info.get("child_neighbors_within_batch"),
                    neighbors_within_batch=node_info.get("neighbors_within_batch"),
                    umap_adata=umap_adata,
                    assigned_cell_type=node_info.get("assigned_cell_type"),
                )
                self.model.nodes[node.node_id] = node

            self.model.root_id = session_payload.get("root_id")
            self.model.saved_node_ids = list(session_payload.get("saved_node_ids", []))

            ui_state = session_payload.get("ui_state", {})
            restored_selection = ui_state.get("selected_node_id")
            if restored_selection and restored_selection in self.model.nodes:
                self.selected_node_id = restored_selection
            else:
                self.selected_node_id = self.model.root_id
            if ui_state.get("batch_key"):
                self.batch_key_var.set(ui_state["batch_key"])
                self.service.batch_key = ui_state["batch_key"]
            if ui_state.get("resolution"):
                self.resolution_var.set(ui_state["resolution"])
            if ui_state.get("neighbors_within_batch"):
                self.neighbors_within_batch_var.set(ui_state["neighbors_within_batch"])
            if ui_state.get("max_epochs"):
                self.max_epochs_var.set(ui_state["max_epochs"])
            if ui_state.get("correction_method") in ("None", "BBKNN", "SCVI"):
                self.correction_method_var.set(ui_state["correction_method"])
            if ui_state.get("clustering_method") in ("None", "Leiden"):
                self.clustering_method_var.set(ui_state["clustering_method"])
            if ui_state.get("annotation_tool") in ("None", "CellTypist"):
                self.annotation_tool_var.set(ui_state["annotation_tool"])
            if ui_state.get("auto_assign_threshold"):
                self.auto_assign_threshold_var.set(ui_state["auto_assign_threshold"])
            if ui_state.get("not_clear_threshold"):
                self.not_clear_threshold_var.set(ui_state["not_clear_threshold"])
            self.available_cell_types = set(ui_state.get("available_cell_types", []))
            # Also pick up any assigned/predicted cell types from the restored nodes themselves,
            # in case they predate the "available_cell_types" ui_state key being saved.
            for restored_node in self.model.nodes.values():
                if restored_node.assigned_cell_type:
                    self.available_cell_types.add(restored_node.assigned_cell_type)

            self.refresh_views()

            node_list_scroll = ui_state.get("node_list_scroll_position")
            self.after(50, lambda: self._restore_scroll_positions(node_list_scroll))

            level = ui_state.get("current_clustering_level", 0)
            self.append_log(f"Restored session with {len(self.model.nodes)} nodes; current clustering level {level}.")
            self.set_status(f"Session loaded from {json_path}.")
        except Exception as exc:
            self.append_log(f"Load session failed: {exc}")
            self.append_log(traceback.format_exc())
            messagebox.showerror("Load session error", str(exc))
            self.set_status("Load session failed.")
        finally:
            self.set_busy(False, self.status_var.get())

    def _restore_scroll_positions(self, node_list_scroll):
        try:
            if node_list_scroll is not None:
                self.node_list_canvas.yview_moveto(node_list_scroll)
        except Exception as exc:
            self.append_log(f"Could not fully restore scroll position: {exc}")

    def refresh_views(self):
        self.refresh_node_list()
        if self.selected_node_id and self.selected_node_id in self.model.nodes:
            self._maybe_predict_cell_types(self.model.nodes[self.selected_node_id])
        self._refresh_all_plot_views()

    def _row_color_state(self, node: "ClusterNode") -> str:
        """Which color a row should use: 'green' if this exact node has an assigned cell type,
        'grey' if it's a leaf that still needs review (no assignment yet), or 'default' for
        non-leaf nodes (cell types only apply to leaves)."""
        if node.assigned_cell_type:
            return "green"
        if not node.children:
            return "grey"
        return "default"

    def _get_row_style(self, color_state: str, is_selected: bool) -> tuple:
        """Lazily create (and cache) the ttk styles for a Tree Nodes list row: green if this
        node has an assigned cell type, light grey if it's a leaf still awaiting review, plus a
        thicker accent-colored outline if this is the currently-selected node. Returns
        (frame_style, label_style, checkbutton_style)."""
        bg = {
            "green": GREEN_HIGHLIGHT_ROW_COLOR,
            "grey": UNASSIGNED_LEAF_ROW_COLOR,
        }.get(color_state, BG_PANEL_ALT)
        fg = HIGHLIGHT_FG if color_state in ("green", "grey") else FG_MAIN
        cache_key = (bg, is_selected)
        cached = self._row_style_cache.get(cache_key)
        if cached is not None:
            return cached
        safe_key = bg.lstrip("#").upper()
        suffix = "Sel" if is_selected else ""
        frame_style = f"Row{safe_key}{suffix}.TFrame"
        label_style = f"Row{safe_key}{suffix}.TLabel"
        checkbutton_style = f"Row{safe_key}{suffix}.TCheckbutton"
        if is_selected:
            self.style.configure(frame_style, background=bg, relief="solid", borderwidth=SELECTED_BORDER_WIDTH, bordercolor=SELECTED_BORDER_COLOR)
        else:
            self.style.configure(frame_style, background=bg, relief="flat", borderwidth=0)
        row_font = ("TkDefaultFont", 9)
        self.style.configure(label_style, background=bg, foreground=fg, font=row_font)
        self.style.configure(checkbutton_style, background=bg, foreground=fg, font=row_font)
        self.style.map(checkbutton_style, background=[("active", bg)])
        styles = (frame_style, label_style, checkbutton_style)
        self._row_style_cache[cache_key] = styles
        return styles

    def _sync_node_list_size(self):
        """Resize the Tree Nodes list's canvas window item to fit its content: at least the
        visible viewport (so short lists still fill the pane), but never smaller than the rows
        actually need - that's what lets the horizontal scrollbar engage instead of the columns
        getting clipped/squished when the list is wider than the pane. Called both from
        <Configure> bindings (window/pane resize) and explicitly after refresh_node_list rebuilds
        the rows, since swapping in new rows of a similar or larger footprint doesn't reliably
        re-trigger a <Configure> event on its own once the canvas item's size has been pinned."""
        viewport_w = self.node_list_canvas.winfo_width()
        viewport_h = self.node_list_canvas.winfo_height()
        target_width = max(viewport_w, self.node_list_container.winfo_reqwidth())
        target_height = max(viewport_h, self.node_list_container.winfo_reqheight())
        self.node_list_canvas.itemconfigure(self.node_list_window, width=target_width, height=target_height)
        self.node_list_canvas.configure(scrollregion=self.node_list_canvas.bbox("all"))

    def _compute_node_name_column_width(self, min_width: int = 8, max_width: int = 40) -> int:
        """Widest 'indent + label (+ [saved])' text across the whole current tree, so the Node
        column always fits the longest cluster identifier/indentation instead of truncating it
        as nodes get deeper or their identifiers grow (e.g. '0.1.2.3')."""
        if not self.model.root_id:
            return self._node_col_widths.get("name", min_width)
        longest = 0

        def walk(node_id: str):
            nonlocal longest
            node = self.model.nodes[node_id]
            indent = "    " * node.depth
            text = f"{indent}{node.label}" + (" [saved]" if node.saved else "")
            longest = max(longest, len(text))
            for child_id in node.children:
                walk(child_id)

        walk(self.model.root_id)
        return max(min_width, min(longest + 2, max_width))

    def refresh_node_list(self):
        """Rebuild the Tree Nodes list from scratch: a header row followed by one row per node
        (indented by depth), showing cell count, resolution, a cell-type combo box (leaves
        only), and a Process action button. The header and every data row are placed
        directly into ONE shared grid (node_list_container's own grid, not each row's own
        independent grid) - this is what keeps every column pixel-aligned between the header and
        the rows even as content changes (e.g. the Node column's width growing with deeper
        nesting), since Tk sizes each grid column to the widest widget assigned to it across
        every row sharing that grid. A row is green if that exact node has an assigned cell
        type, light grey if it's a leaf still awaiting review, or the default color for non-leaf
        nodes."""
        for child in self.node_list_container.winfo_children():
            child.destroy()
        self._node_col_widths["name"] = self._compute_node_name_column_width()
        self._render_node_list_header()
        if not self.model.root_id:
            self.node_list_container.update_idletasks()
            self._sync_node_list_size()
            return
        self._next_node_row_index = 1
        self._render_node_row(self.model.root_id)
        self.node_list_container.update_idletasks()
        self._sync_node_list_size()

    def _render_node_list_header(self):
        w = self._node_col_widths
        c = self.node_list_container
        ttk.Label(c, text="Node", style="TreeNodesHeader.TLabel", width=w["name"], anchor="w").grid(row=0, column=0, sticky="w", padx=(4, 0), pady=(2, 4))
        ttk.Label(c, text="Cells", style="TreeNodesHeader.TLabel", width=w["cells"], anchor="w").grid(row=0, column=1, sticky="w", pady=(2, 4))
        ttk.Label(c, text="Resolution", style="TreeNodesHeader.TLabel", width=w["resolution"], anchor="w").grid(row=0, column=2, sticky="w", pady=(2, 4))
        ttk.Label(c, text="Neighbors", style="TreeNodesHeader.TLabel", width=w["neighbors"], anchor="w").grid(row=0, column=3, sticky="w", pady=(2, 4))
        ttk.Label(c, text="Cell Type", style="TreeNodesHeader.TLabel", width=w["cell_type"], anchor="w").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=(2, 4))
        ttk.Label(c, text="Action", style="TreeNodesHeader.TLabel", width=w["action"], anchor="w").grid(row=0, column=5, sticky="w", pady=(2, 4))

    def _render_node_row(self, node_id: str):
        node = self.model.nodes[node_id]
        color_state = self._row_color_state(node)
        is_selected = node_id == self.selected_node_id
        frame_style, label_style, _checkbutton_style = self._get_row_style(color_state, is_selected)
        w = self._node_col_widths
        c = self.node_list_container
        row_index = self._next_node_row_index
        self._next_node_row_index += 1

        # Full-width colored band for this row, gridded first so the cell widgets (created next,
        # and thus stacked above it by Tk's default creation-order z-ordering) render on top of
        # it - this keeps the colored-row look while every widget still lives in the ONE shared
        # grid that keeps columns aligned with the header.
        band = ttk.Frame(c, style=frame_style)
        band.grid(row=row_index, column=0, columnspan=5, sticky="nsew")
        band.bind("<Button-1>", lambda _e, nid=node_id: self.select_node(nid))

        indent = "    " * node.depth
        name_text = f"{indent}{node.label}" + (" [saved]" if node.saved else "")
        name_label = ttk.Label(c, text=name_text, style=label_style, width=w["name"], anchor="w")
        name_label.grid(row=row_index, column=0, sticky="w", padx=(4, 0), pady=3)

        cells_label = ttk.Label(c, text=str(node.n_cells()), style=label_style, width=w["cells"], anchor="w")
        cells_label.grid(row=row_index, column=1, sticky="w", pady=3)

        resolution_text = f"{node.resolution:g}" if node.resolution is not None else "—"
        resolution_label = ttk.Label(c, text=resolution_text, style=label_style, width=w["resolution"], anchor="w")
        resolution_label.grid(row=row_index, column=2, sticky="w", pady=3)

        neighbors_text = str(node.neighbors_within_batch) if node.neighbors_within_batch is not None else "—"
        neighbors_label = ttk.Label(c, text=neighbors_text, style=label_style, width=w["neighbors"], anchor="w")
        neighbors_label.grid(row=row_index, column=3, sticky="w", pady=3)

        is_leaf = not node.children
        combo_var = tk.StringVar(value=node.assigned_cell_type or NOT_ASSIGNED_CELL_TYPE_LABEL)
        combo = ttk.Combobox(
            c, textvariable=combo_var, values=[NOT_ASSIGNED_CELL_TYPE_LABEL] + sorted(self.available_cell_types),
            state=("readonly" if is_leaf else "disabled"), width=w["cell_type"],
            style="TreeNodesRow.TCombobox", font=("TkDefaultFont", 9),
        )
        combo.grid(row=row_index, column=4, sticky="w", padx=(0, 6), pady=3)
        combo.bind("<<ComboboxSelected>>", lambda _e, nid=node_id, var=combo_var: self._assign_cell_type(nid, var.get()))
        # Also set the dropdown listbox's font directly - it's a separate Tk widget the ttk
        # style layer doesn't reach, so without this the popup list still shows the theme's
        # default (larger) font even when the closed field itself looks right.
        self.option_add("*TCombobox*Listbox.font", ("TkDefaultFont", 9))

        ttk.Button(
            c, text="Process", width=w["action"], style="TreeNodesRow.TButton",
            command=lambda nid=node_id: self.select_and_process(nid),
        ).grid(row=row_index, column=5, sticky="w", pady=3)

        for widget in (name_label, cells_label, resolution_label, neighbors_label):
            widget.bind("<Button-1>", lambda _e, nid=node_id: self.select_node(nid))

        for child_id in node.children:
            self._render_node_row(child_id)

    def _assign_cell_type(self, node_id: str, cell_type: str):
        """Called when the user picks a value in a leaf row's cell-type combo box. Highlights
        just that row green (recomputed on refresh via _row_color_state). Selecting the
        'Not assigned' sentinel clears any existing assignment - lets the user undo a mistaken
        annotation directly, without needing to recluster the node."""
        node = self.model.nodes.get(node_id)
        if node is None:
            return
        if cell_type and cell_type != NOT_ASSIGNED_CELL_TYPE_LABEL:
            node.assigned_cell_type = cell_type
            self.append_log(f"Assigned cell type '{node.assigned_cell_type}' to node {node.label}")
        else:
            node.assigned_cell_type = None
            self.append_log(f"Cleared cell type assignment for node {node.label}")
        self.refresh_node_list()


    def _color_array_for_clusters(self, labels: List[str]) -> List[str]:
        unique_labels = sorted(set(labels))
        color_map = {label: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, label in enumerate(unique_labels)}
        return [color_map[label] for label in labels]

    def draw_umap_preview(self, node: Optional[ClusterNode], target_frame=None, big: bool = False, figsize_override=None):
        frame = target_frame if target_frame is not None else getattr(self, "umap_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(frame, text="Matplotlib is not installed.", style="Panel.TLabel").pack(anchor="w")
            return
        if node is None:
            ttk.Label(frame, text="No node selected.", style="Panel.TLabel").pack(anchor="w")
            return
        adata = node.umap_adata if node.umap_adata is not None else node.adata
        if adata is None or adata.n_obs == 0:
            ttk.Label(frame, text="Selected node has no cells.", style="Panel.TLabel").pack(anchor="w")
            return
        if "X_umap" not in adata.obsm.keys():
            ttk.Label(frame, text="UMAP preview will appear after this node is clustered.", style="Panel.TLabel").pack(anchor="w")
            return
        coords = np.asarray(adata.obsm["X_umap"])
        if coords.shape[1] < 2:
            ttk.Label(frame, text="UMAP coordinates are incomplete.", style="Panel.TLabel").pack(anchor="w")
            return

        if figsize_override is not None:
            figsize = figsize_override
        elif big:
            figsize = (7.5, 6.2)
        else:
            figsize = self._infer_embedded_figsize(frame)
        fig = Figure(figsize=figsize, dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        if node.last_cluster_key and node.last_cluster_key in adata.obs.columns:
            cluster_labels = adata.obs[node.last_cluster_key].astype(str).tolist()
            point_colors = self._color_array_for_clusters(cluster_labels)
            ax.scatter(coords[:, 0], coords[:, 1], s=7, alpha=0.85, c=point_colors)
            labels_array = np.asarray(cluster_labels)
            for cluster_id in sorted(set(cluster_labels)):
                mask = labels_array == cluster_id
                if np.any(mask):
                    centroid_x = float(np.mean(coords[mask, 0]))
                    centroid_y = float(np.mean(coords[mask, 1]))
                    ax.text(
                        centroid_x, centroid_y, cluster_id,
                        fontsize=9, fontweight="bold", color="black",
                        ha="center", va="center", zorder=5,
                    )
            if node.child_resolution is not None:
                ax.set_title(f"UMAP clusters r={node.child_resolution}", fontsize=9, color="black")
            else:
                ax.set_title(f"UMAP clusters: {node.label}", fontsize=9, color="black")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.8, c=ACCENT)
            ax.set_title(f"UMAP: {node.label}", fontsize=9, color="black")

        ax.set_xlabel("UMAP1", fontsize=8, color="black")
        ax.set_ylabel("UMAP2", fontsize=8, color="black")
        ax.tick_params(labelsize=7, colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.grid(alpha=0.2, color="#d1d5db")
        # Fixed, tight margins instead of tight_layout() - tight_layout tends to leave a
        # generous, inconsistent border around small preview figures. These fractions hug
        # the axis/tick labels closely while still leaving room for them.
        if big:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.09)
        else:
            fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.15)
        if big:
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
            canvas.get_tk_widget().pack(expand=True, padx=4, pady=4)
            self._attach_plot_toolbar(canvas, frame)
            self.umap_canvas = canvas
            self.umap_figure = fig
        else:
            # Rasterize once at the generous native size above, then display a copy scaled down
            # to fit the pane's current width - see _display_figure_scaled_to_fit for why this
            # (rather than embedding the live canvas) is what keeps dots from overlapping.
            self._display_figure_scaled_to_fit(fig, frame, cache_attr="umap")

    def load_markers_csv(self):
        path = filedialog.askopenfilename(
            title="Select markers CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            marker_dict: Dict[str, List[str]] = {}
            with open(path, newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                for row_index, row in enumerate(reader):
                    if not row:
                        continue
                    cell_type = str(row[0]).strip()
                    if not cell_type:
                        continue
                    genes = [str(g).strip() for g in row[1:] if str(g).strip()]
                    if not genes:
                        self.append_log(f"Markers CSV row {row_index + 1} ('{cell_type}') has no gene markers; skipped.")
                        continue
                    marker_dict[cell_type] = genes
            if not marker_dict:
                raise ValueError("No cell types with marker genes were found in this file.")
            existing = next((m for m in self.marker_file_sets if m["path"] == path), None)
            total_genes = sum(len(v) for v in marker_dict.values())
            if existing is not None:
                existing["marker_dict"] = marker_dict
                self.append_log(f"Reloaded marker genes from {path} (updating its existing dot plot)")
            else:
                self.marker_file_sets.append({
                    "id": self._next_marker_set_id,
                    "path": path,
                    "name": os.path.basename(path),
                    "marker_dict": marker_dict,
                })
                self._next_marker_set_id += 1
                self.append_log(f"Loaded marker genes from {path} (added as a new stacked dot plot)")
            self.append_log(f"Marker file contains {len(marker_dict)} cell types and {total_genes} gene markers total")
            self.set_status(f"Loaded markers for {len(marker_dict)} cell types from {os.path.basename(path)}. {len(self.marker_file_sets)} marker file(s) now stacked.")
            node = None
            if self.selected_node_id and self.selected_node_id in self.model.nodes:
                node = self.model.nodes[self.selected_node_id]
            self._refresh_dotplot_stack(node)
        except Exception as exc:
            self.append_log(f"Failed to load markers CSV: {exc}")
            messagebox.showerror("Load Markers CSV", f"Could not load markers file:\n{exc}")

    def _build_normalized_node_copy(self, node: ClusterNode, cluster_key: str, umap_adata, common_barcodes, stage_label: str):
        """Build a dedicated copy of this node's RAW counts (all genes), aligned to the cluster
        labels currently stored on node.umap_adata, then normalize/log1p it fresh from those raw
        counts. Used by both the marker dot plot and CellTypist prediction so that neither
        node.adata nor node.umap_adata (HVG-subsetted/scaled) is ever mutated by either feature.
        """
        plot_adata = node.adata[common_barcodes].copy()
        plot_adata.raw = None  # avoid scanpy/celltypist silently reading a stale .raw slot
        plot_adata.obs[cluster_key] = (
            umap_adata.obs.loc[common_barcodes, cluster_key].astype(str).astype("category").values
        )
        self.append_log(f"[{stage_label}] Built working copy with {plot_adata.n_obs} cells (raw counts, all genes)")
        sc.pp.normalize_total(plot_adata, target_sum=1e4)
        self.append_log(f"[{stage_label}] Step: normalize_total (target_sum=10000.0)")
        sc.pp.log1p(plot_adata)
        self.append_log(f"[{stage_label}] Step: log1p transform")
        return plot_adata

    def draw_dot_plot(self, node: Optional[ClusterNode], marker_dict: Dict[str, List[str]], title_suffix: str = "", target_frame=None, big: bool = False, figsize_override=None, cache_key: str = "dotplot", font_scale: float = 0.65, padding_scale: float = 1.3, largest_dot: float = 200.0, return_fig: bool = False):
        """Render one marker-gene dot plot for `node`, using `marker_dict` (one loaded markers
        CSV's cell_type -> genes mapping). `title_suffix` (typically that CSV's filename) is
        used in log/error messages so multiple stacked dot plots stay distinguishable in the
        log. `cache_key` namespaces the cached canvas/figure attributes this call stores on
        self, so multiple simultaneously-rendered dot plots (one per loaded markers file) don't
        overwrite each other's cache. `font_scale` scales all text in the plot; `padding_scale`
        scales the empty space scanpy reserves around each dot (>1 shrinks dots relative to
        their grid cell, which is the direct fix for overlapping dots); `largest_dot` is the
        point-size of the biggest dot scanpy will ever draw (scanpy's own default is 200).
        `return_fig=True` skips displaying the figure entirely and just returns the built
        matplotlib Figure to the caller instead - used by the high-resolution zoomable viewer
        (open_dotplot_zoomable_view), which needs a fresh, undisplayed Figure it can re-render
        at its own chosen DPI rather than whatever the embedded panel or popup already used."""
        frame = target_frame if target_frame is not None else getattr(self, "dotplot_frame", None)
        if not return_fig and frame is None:
            return
        if not return_fig:
            for child in frame.winfo_children():
                child.destroy()
        if node is not None:
            label_tag = f"Dot plot | {node.label}" + (f" | {title_suffix}" if title_suffix else "")
        else:
            label_tag = "Dot plot"

        def show_message(text: str):
            # When return_fig is True, the caller owns all UI - never touch `frame` (which,
            # when target_frame wasn't given, resolves to the SHARED self.dotplot_frame) or an
            # error message would land there directly instead of wherever the caller actually
            # wants it shown.
            if return_fig or frame is None:
                return
            ttk.Label(frame, text=text, style="Panel.TLabel", wraplength=260, justify="left").pack(anchor="w", pady=6, padx=4)

        if Figure is None or FigureCanvasTkAgg is None or plt is None:
            show_message("Matplotlib is not installed; cannot render the dot plot.")
            return
        if sc is None:
            show_message("Scanpy is not installed; cannot render the dot plot.")
            return
        if node is None:
            show_message("No node selected.")
            return
        if not marker_dict:
            show_message(f"No marker genes loaded for '{title_suffix}'." if title_suffix else "Load a markers CSV file to display a dot plot.")
            return
        cluster_key = node.last_cluster_key
        umap_adata = node.umap_adata
        if not cluster_key or umap_adata is None or cluster_key not in umap_adata.obs.columns:
            show_message("Dot plot will appear after this node is clustered.")
            return
        if node.adata is None or node.adata.n_obs == 0:
            show_message("Selected node has no cells.")
            return

        try:
            # Build a dedicated copy of this node's RAW counts (all genes) purely for plotting, so
            # neither node.adata nor node.umap_adata (HVG-subsetted/scaled) are ever touched here.
            common_barcodes = node.adata.obs_names.intersection(umap_adata.obs_names)
            if len(common_barcodes) == 0:
                show_message("No matching cells between raw counts and cluster labels.")
                return
            plot_adata = self._build_normalized_node_copy(
                node, cluster_key, umap_adata, common_barcodes, stage_label=label_tag
            )

            # Filter marker genes down to ones actually present in this dataset, to avoid KeyErrors.
            available_genes = set(plot_adata.var_names)
            filtered_marker_dict: Dict[str, List[str]] = {}
            missing_genes: List[str] = []
            for cell_type, genes in marker_dict.items():
                present = [g for g in genes if g in available_genes]
                missing_genes.extend(g for g in genes if g not in available_genes)
                if present:
                    filtered_marker_dict[cell_type] = present
            if missing_genes:
                self.append_log(
                    f"[{label_tag}] {len(missing_genes)} marker genes not found in this dataset and were skipped: "
                    f"{', '.join(sorted(set(missing_genes)))}"
                )
            if not filtered_marker_dict:
                show_message("None of the loaded marker genes were found in this dataset.")
                return

            all_marker_genes = sorted({gene for genes in filtered_marker_dict.values() for gene in genes})
            plot_adata = plot_adata[:, all_marker_genes].copy()
            self.append_log(f"[{label_tag}] Filtered plotting copy down to {plot_adata.n_vars} marker genes present in this dataset")

            num_groups = len(set(plot_adata.obs[cluster_key]))
            if figsize_override is not None:
                fig_width, fig_height = figsize_override
            else:
                # Default to generous per-gene/group spacing so dots never overlap. The size
                # controls in the popup let the user override this with something smaller
                # (e.g. 6x6) at the cost of possible crowding with many genes/groups.
                fig_width = max(11.0, 0.55 * len(all_marker_genes) + 3.5)
                fig_height = max(6.0, 0.5 * num_groups + 3.2)

            plt.close("all")
            # Scanpy's dotplot doesn't take an explicit dpi kwarg - bump the rcParam so the
            # rendered figure is higher-resolution (sharper both embedded and when saved), then
            # restore it so it doesn't leak into other plots. font.size drives every text
            # element scanpy draws (axis labels, tick labels, legend) - scaling it is the direct
            # way to make labels bigger/smaller without touching the figure's physical size.
            _prev_dpi = matplotlib.rcParams.get("figure.dpi", 100)
            _prev_font_size = matplotlib.rcParams.get("font.size", 10.0)
            matplotlib.rcParams["figure.dpi"] = 200
            matplotlib.rcParams["font.size"] = max(4.0, _prev_font_size * font_scale)
            try:
                dot_plot = sc.pl.dotplot(
                    plot_adata,
                    filtered_marker_dict,
                    groupby=cluster_key,
                    standard_scale="var",
                    use_raw=False,
                    show=False,
                    return_fig=True,
                    figsize=(fig_width, fig_height),
                )
                # x_padding/y_padding (scanpy defaults 0.8/1.0) are the fraction of each grid
                # cell reserved as empty space around its dot - raising them shrinks every dot
                # relative to its cell, which is the direct fix for dots overlapping their
                # neighbors. largest_dot caps the biggest dot's size in points outright.
                dot_plot.style(
                    largest_dot=largest_dot,
                    x_padding=0.8 * padding_scale,
                    y_padding=1.0 * padding_scale,
                )
                dot_plot.show()
                fig = dot_plot.fig
            finally:
                matplotlib.rcParams["figure.dpi"] = _prev_dpi
                matplotlib.rcParams["font.size"] = _prev_font_size

            if return_fig:
                return fig

            if big:
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
                canvas.get_tk_widget().pack(expand=True, padx=4, pady=4)
                self._attach_plot_toolbar(canvas, frame)
                setattr(self, cache_key + "_canvas", canvas)
                setattr(self, cache_key + "_figure", fig)
            else:
                # Embedded dashboard panel: several dot plots can be stacked in one shared,
                # vertically-scrolling area (self.dotplot_frame), so every plot must fit the
                # SAME outer viewport's width (measure_frame), not whatever width its own small
                # placement holder (frame) reports - and only width should ever constrain it
                # (fit_height=False), since the panel scrolls vertically to fit however tall the
                # result ends up being rather than needing it to fit one screen's height too.
                self._display_figure_scaled_to_fit(
                    fig, frame, cache_attr=cache_key,
                    measure_frame=getattr(self, "dotplot_frame", None), fit_height=False,
                )
            plt.close(fig)
        except Exception as exc:
            self.append_log(f"[{label_tag}] Failed to render dot plot: {exc}")
            show_message(f"Could not render dot plot: {exc}")

    def _draw_dot_plot_for_marker_id(self, marker_id: int, node: Optional[ClusterNode], target_frame=None, big: bool = False, figsize_override=None, font_scale: float = 0.65, padding_scale: float = 1.3, largest_dot: float = 200.0, return_fig: bool = False):
        """Look up a specific loaded markers file by id and render its dot plot. Looking this up
        fresh on every call (rather than capturing marker_dict in a closure at popup-open time)
        means re-loading that same CSV later transparently updates any already-open popup too."""
        entry = next((m for m in self.marker_file_sets if m["id"] == marker_id), None)
        if entry is None:
            if return_fig:
                return None
            frame = target_frame if target_frame is not None else getattr(self, "dotplot_frame", None)
            if frame is not None:
                for child in frame.winfo_children():
                    child.destroy()
                ttk.Label(frame, text="This marker set has been removed.", style="Panel.TLabel").pack(anchor="w", pady=6, padx=4)
            return
        return self.draw_dot_plot(
            node, entry["marker_dict"], entry["name"], target_frame=target_frame, big=big,
            figsize_override=figsize_override, cache_key=f"dotplot_{marker_id}",
            font_scale=font_scale, padding_scale=padding_scale, largest_dot=largest_dot,
            return_fig=return_fig,
        )

    def _refresh_dotplot_stack(self, node: Optional[ClusterNode]):
        """Rebuild the embedded 'Markers Dot Plots' panel: one stacked block per currently
        loaded markers CSV (in load order), each showing its dot plot directly as an
        interactive, high-resolution zoom/pan canvas (see _render_dotplot_embedded) - not a
        static shrink-to-fit thumbnail - so dots never overlap and the person can zoom in on
        any part of it right there in the dashboard, no separate window required. Each block's
        header also has a "Full Window" button (the same interactive canvas, just bigger, for
        closer inspection than the compact embedded height allows) and a remove button. All
        blocks share the single outer scrollable area created for this panel, so the whole
        stack scrolls together."""
        frame = getattr(self, "dotplot_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        if not self.marker_file_sets:
            ttk.Label(
                frame, text="Load a markers CSV file (Annotation menu) to display a dot plot. "
                "Loading more than one file stacks their dot plots here.",
                style="Panel.TLabel", wraplength=260, justify="left",
            ).pack(anchor="w", pady=6, padx=4)
            return
        for index, entry in enumerate(self.marker_file_sets):
            marker_id = entry["id"]
            block = ttk.Frame(frame, style="Panel.TFrame")
            block.pack(fill="x", padx=2, pady=(0, 4 if index == len(self.marker_file_sets) - 1 else 0), anchor="n")
            header = ttk.Frame(block, style="Panel.TFrame")
            header.pack(fill="x", pady=(4, 1))
            ttk.Label(header, text=f"Markers Dot Plot \u2014 {entry['name']}", style="Panel.TLabel").pack(side="left")
            ttk.Button(
                header, text="Full Window", width=11,
                command=lambda mid=marker_id: self.open_dotplot_zoomable_view(mid),
            ).pack(side="right", padx=(0, 6))
            ttk.Button(header, text="Remove", width=8, command=lambda mid=marker_id: self.remove_marker_set(mid)).pack(side="right", padx=(0, 6))
            plot_holder = ttk.Frame(block, style="PlotArea.TFrame")
            plot_holder.pack(fill="x")
            self._render_dotplot_embedded(marker_id, node, entry, plot_holder)
            if index != len(self.marker_file_sets) - 1:
                ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8, padx=2)

    def _render_dotplot_embedded(self, marker_id: int, node: Optional[ClusterNode], entry: dict, plot_holder):
        """Render this marker set's dot plot at a genuinely high, fixed resolution (300 dpi -
        the same as the "Full Window" view) and embed it directly in the dashboard panel as an
        interactive zoom/pan canvas (_build_zoomable_canvas), capped at a compact fixed height
        (several plots can be stacked, so no single one should be allowed to take over the
        whole panel - the "Full Window" button is there for when more room is actually needed).
        Nothing here is ever written to disk; the rendered image stays purely in memory, same
        as the plain thumbnail this replaces, so there is no temporary file to manage."""
        fig = self._draw_dot_plot_for_marker_id(marker_id, node, return_fig=True)
        if fig is None:
            ttk.Label(
                plot_holder, text="Dot plot will appear after this node is clustered, or check "
                "the Processing Log if this markers CSV failed to load for this dataset.",
                style="Panel.TLabel", wraplength=260, justify="left",
            ).pack(anchor="w", pady=6, padx=4)
            return
        if Image is None or ImageTk is None or FigureCanvasAgg is None:
            ttk.Label(
                plot_holder, text="Pillow is not installed; cannot render the dot plot.",
                style="Panel.TLabel",
            ).pack(anchor="w", pady=6, padx=4)
            plt.close(fig)
            return
        render_dpi = 300.0
        fig.set_dpi(render_dpi)
        agg_canvas = FigureCanvasAgg(fig)
        agg_canvas.draw()
        width_px, height_px = agg_canvas.get_width_height()
        native_image = Image.frombuffer(
            "RGBA", (width_px, height_px), agg_canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        ).copy()
        plt.close(fig)
        self._build_zoomable_canvas(
            plot_holder, native_image, cache_attr=f"dotplot_{marker_id}",
            default_filename=f"dotplot_{entry['name']}", render_dpi=render_dpi,
            fixed_height=460, compact=True,
        )

    def _build_zoomable_canvas(self, parent, native_image, cache_attr: str, default_filename: Optional[str] = None,
                                render_dpi: float = 300.0, fixed_height: Optional[int] = None, compact: bool = False):
        """Build, inside `parent`, an interactive viewer for an already-rendered `native_image`
        (a PIL Image - callers rasterize at whatever DPI they want before calling this): click
        Zoom In/Out/Fit buttons (the primary, always-discoverable way to zoom - a scroll wheel
        alone isn't reliable across every mouse/trackpad, so it's offered as a bonus, not the
        only path) plus click-and-drag panning. If `default_filename` is given, also adds a
        "Save as PNG..." button to export `native_image` to a real, permanent, user-chosen file
        - that's the only thing here that ever touches disk; everything else stays purely
        in-memory, the same way the plain embedded thumbnail already worked, so there's no
        temporary file to manage or clean up for ordinary viewing.

        `fixed_height`, if given, caps the canvas at that height (e.g. one block within a
        vertical stack of several plots, where the outer panel - not this inner canvas - is
        what scrolls between blocks); otherwise the canvas expands to fill whatever space
        `parent` gives it. `compact=True` trims the header/hint text down for a small embedded
        block instead of a full popup window.
        """
        chrome_pad = 4 if compact else 8
        header = ttk.Frame(parent, style="Panel.TFrame", padding=(chrome_pad, chrome_pad, chrome_pad, 2))
        header.pack(fill="x")

        zoom_label_var = tk.StringVar(value="100%")

        def _save_as_png():
            path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG image", "*.png")],
                initialfile=f"{default_filename}.png" if default_filename else "dotplot.png",
            )
            if not path:
                return
            try:
                native_image.convert("RGB").save(path, dpi=(render_dpi, render_dpi))
                self.append_log(f"Saved high-resolution image to {path}")
                self.set_status(f"Saved {os.path.basename(path)}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))

        if default_filename is not None:
            ttk.Button(header, text="Save as PNG...", width=13, command=_save_as_png).pack(side="right")
        ttk.Label(header, textvariable=zoom_label_var, style="Panel.TLabel").pack(side="right", padx=(0, 10))

        canvas_frame = ttk.Frame(parent, style="PlotArea.TFrame")
        canvas_frame.pack(fill="both", expand=not bool(fixed_height))
        if fixed_height:
            canvas_frame.configure(height=fixed_height)
            canvas_frame.pack_propagate(False)
        canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        state = {"zoom": 1.0, "photo": None, "image_id": None, "user_interacted": False}

        def _render_at_zoom():
            z = state["zoom"]
            w = max(1, int(native_image.width * z))
            h = max(1, int(native_image.height * z))
            resized = native_image.resize((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            state["photo"] = photo  # keep a reference so Tk doesn't garbage-collect it
            setattr(self, cache_attr + "_zoom_photo", photo)  # extra safety against gc
            if state["image_id"] is None:
                state["image_id"] = canvas.create_image(0, 0, anchor="nw", image=photo)
            else:
                canvas.itemconfigure(state["image_id"], image=photo)
            canvas.configure(scrollregion=(0, 0, w, h))
            zoom_label_var.set(f"{int(z * 100)}%")

        def _fit_to_window():
            # Pressing "Fit" is also how the person resets back to an auto-fitting view after
            # zooming/panning manually, so it clears user_interacted too - see
            # _on_canvas_configure below, which re-fits automatically on resize only while this
            # flag is False.
            state["user_interacted"] = False
            canvas.update_idletasks()
            avail_w = canvas.winfo_width()
            avail_h = canvas.winfo_height()
            if avail_w > 1 and avail_h > 1:
                state["zoom"] = min(avail_w / native_image.width, avail_h / native_image.height, 1.0)
                _render_at_zoom()
                return True
            # Canvas hasn't been laid out to a real size yet (e.g. still 1x1 on first show) -
            # nothing sensible to fit to yet. _schedule_initial_fit below retries until it is.
            return False

        def _zoom_by(factor, center_x=None, center_y=None):
            state["user_interacted"] = True
            old_zoom = state["zoom"]
            new_zoom = max(0.05, min(old_zoom * factor, 10.0))
            if new_zoom == old_zoom:
                return
            # Default to zooming on the middle of the currently visible area (used by the
            # +/- buttons); mouse-wheel zoom passes the actual cursor position instead.
            if center_x is None:
                center_x = canvas.winfo_width() / 2
            if center_y is None:
                center_y = canvas.winfo_height() / 2
            cx = canvas.canvasx(center_x)
            cy = canvas.canvasy(center_y)
            rel_x = cx / max(1.0, native_image.width * old_zoom)
            rel_y = cy / max(1.0, native_image.height * old_zoom)
            state["zoom"] = new_zoom
            _render_at_zoom()
            new_cx = rel_x * native_image.width * new_zoom
            new_cy = rel_y * native_image.height * new_zoom
            total_w = max(1, int(native_image.width * new_zoom))
            total_h = max(1, int(native_image.height * new_zoom))
            canvas.xview_moveto(max(0.0, (new_cx - center_x)) / total_w)
            canvas.yview_moveto(max(0.0, (new_cy - center_y)) / total_h)

        def _on_mousewheel(event):
            # Windows/Mac report event.delta directly; Linux sends Button-4/5 instead (bound
            # separately below), so this handler only needs the delta-based convention.
            _zoom_by(1.1 if event.delta > 0 else (1 / 1.1), event.x, event.y)

        def _on_wheel_linux(event):
            _zoom_by(1.1 if event.num == 4 else (1 / 1.1), event.x, event.y)

        def _on_drag_start(event):
            state["user_interacted"] = True
            canvas.scan_mark(event.x, event.y)

        def _on_drag_move(event):
            canvas.scan_dragto(event.x, event.y, gain=1)

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_wheel_linux)
        canvas.bind("<Button-5>", _on_wheel_linux)
        canvas.bind("<ButtonPress-1>", _on_drag_start)
        canvas.bind("<B1-Motion>", _on_drag_move)

        # Zoom In / Zoom Out / Fit buttons - the primary, always-discoverable way to zoom,
        # since relying on the scroll wheel alone wasn't easy to use for everyone.
        ttk.Button(header, text="Fit", width=5, command=_fit_to_window).pack(side="right", padx=(0, 6))
        ttk.Button(header, text="\u2212", width=3, command=lambda: _zoom_by(1 / 1.3)).pack(side="right", padx=(0, 2))
        ttk.Button(header, text="+", width=3, command=lambda: _zoom_by(1.3)).pack(side="right", padx=(0, 2))

        if not compact:
            hint = ttk.Frame(parent, style="Panel.TFrame", padding=(chrome_pad, 2, chrome_pad, 4))
            hint.pack(fill="x")
            ttk.Label(
                hint, text="+/- or scroll to zoom \u00b7 click and drag to pan \u00b7 Fit to reset",
                style="Panel.TLabel",
            ).pack(anchor="w")

        def _on_canvas_configure(_event):
            # Whenever this canvas actually changes size - including the very first time it
            # gets laid out to its real size, which a single fixed-delay after() call can miss
            # or catch too early depending on how busy the rest of the UI is - re-fit to it,
            # but only while the person hasn't manually zoomed/panned (Fit clears that flag).
            if not state["user_interacted"]:
                _fit_to_window()

        canvas.bind("<Configure>", _on_canvas_configure)

        def _schedule_initial_fit(attempts_left=20):
            # Belt-and-suspenders for the rare case <Configure> never fires with a usable size
            # (e.g. an already-realized frame that doesn't resize again): keep retrying briefly
            # on a short timer rather than gambling on one fixed delay.
            if state["user_interacted"]:
                return
            if not _fit_to_window() and attempts_left > 0:
                parent.after(50, lambda: _schedule_initial_fit(attempts_left - 1))

        _schedule_initial_fit()
        return {"canvas": canvas, "fit": _fit_to_window, "zoom_by": _zoom_by}

    def open_dotplot_zoomable_view(self, marker_id: int):
        """Render this marker set's dot plot fresh at a genuinely high, fixed resolution
        (300 dpi, independent of screen size or the embedded panel's shrink-to-fit), then open
        it in a dedicated window built from _build_zoomable_canvas: it rasterizes once, then
        zooms/pans that single saved image, which stays responsive regardless of how many dots
        it contains."""
        if self.selected_node_id is None or self.selected_node_id not in self.model.nodes:
            messagebox.showwarning("High-Res Zoom", "Select a node first.")
            return
        node = self.model.nodes[self.selected_node_id]
        entry = next((m for m in self.marker_file_sets if m["id"] == marker_id), None)
        if entry is None:
            messagebox.showerror("High-Res Zoom", "This marker set has been removed.")
            return
        fig = self._draw_dot_plot_for_marker_id(marker_id, node, return_fig=True)
        if fig is None:
            messagebox.showerror(
                "High-Res Zoom",
                "Could not render this dot plot (check the Processing Log for details) - it "
                "may need this node to be clustered first, or the markers CSV may have no "
                "genes present in this dataset.",
            )
            return
        if Image is None or ImageTk is None or FigureCanvasAgg is None:
            messagebox.showerror("High-Res Zoom", "Pillow is required for this feature.")
            plt.close(fig)
            return
        render_dpi = 300.0
        fig.set_dpi(render_dpi)
        agg_canvas = FigureCanvasAgg(fig)
        agg_canvas.draw()
        width_px, height_px = agg_canvas.get_width_height()
        native_image = Image.frombuffer(
            "RGBA", (width_px, height_px), agg_canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        ).copy()
        plt.close(fig)

        title = f"High-Res Dot Plot \u2014 {entry['name']}"
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("1100x800")
        popup.configure(bg=BG_PANEL)
        self._build_zoomable_canvas(
            popup, native_image, cache_attr=f"dotplot_popup_zoom_{marker_id}",
            default_filename=f"dotplot_{entry['name']}", render_dpi=render_dpi,
        )

    def remove_marker_set(self, marker_id: int):
        entry = next((m for m in self.marker_file_sets if m["id"] == marker_id), None)
        if entry is None:
            return
        self.marker_file_sets = [m for m in self.marker_file_sets if m["id"] != marker_id]
        attr_name = f"dotplot_popup_window_{marker_id}"
        popup = getattr(self, attr_name, None)
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        if hasattr(self, attr_name):
            delattr(self, attr_name)
        self.append_log(f"Removed marker set '{entry['name']}'.")
        node = None
        if self.selected_node_id and self.selected_node_id in self.model.nodes:
            node = self.model.nodes[self.selected_node_id]
        self._refresh_dotplot_stack(node)

    def clear_marker_plots(self):
        if not self.marker_file_sets:
            self.set_status("No marker files are currently loaded.")
            return
        if not messagebox.askyesno("Clear Marker Plots", "Remove all loaded marker gene files and their dot plots?"):
            return
        for entry in list(self.marker_file_sets):
            attr_name = f"dotplot_popup_window_{entry['id']}"
            popup = getattr(self, attr_name, None)
            if popup is not None and popup.winfo_exists():
                popup.destroy()
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.marker_file_sets = []
        self.append_log("Cleared all loaded marker gene files.")
        self.set_status("Cleared all marker gene files.")
        node = None
        if self.selected_node_id and self.selected_node_id in self.model.nodes:
            node = self.model.nodes[self.selected_node_id]
        self._refresh_dotplot_stack(node)

    def _load_default_celltypist_model(self):
        """Auto-load the default CellTypist model (Human_IPF_Lung.pkl) at startup. CellTypist
        downloads it to a local cache on first use if it isn't already present. Users can switch
        to a different model at any time with the 'Load CellTypist Model' button."""
        if celltypist is None:
            self.append_log(f"CellTypist is not installed; skipping default model load ({DEFAULT_CELLTYPIST_MODEL}).")
            return
        try:
            self.set_status(f"Loading default CellTypist model '{DEFAULT_CELLTYPIST_MODEL}' (downloading if needed)...")
            self.update_idletasks()
            model = celltypist.models.Model.load(model=DEFAULT_CELLTYPIST_MODEL)
            self.celltypist_model = model
            self.celltypist_model_path = DEFAULT_CELLTYPIST_MODEL
            cell_types = getattr(model, "cell_types", None)
            n_types = len(cell_types) if cell_types is not None else "unknown"
            self.append_log(f"Loaded default CellTypist model '{DEFAULT_CELLTYPIST_MODEL}' ({n_types} cell types).")
            self.append_log("Use 'Load CellTypist Model' to switch to a different model at any time.")
            self.set_status(f"Default CellTypist model loaded: {DEFAULT_CELLTYPIST_MODEL}.")
        except Exception as exc:
            self.append_log(f"Could not load default CellTypist model '{DEFAULT_CELLTYPIST_MODEL}': {exc}")
            self.append_log("You can still load a model manually via 'Load CellTypist Model'.")
            self.set_status("Default CellTypist model unavailable; load one manually if needed.")

    def load_celltypist_model(self):
        if celltypist is None:
            messagebox.showerror(
                "CellTypist",
                "The celltypist package is not installed.\nInstall with: pip install celltypist",
            )
            return
        path = filedialog.askopenfilename(
            title="Select a CellTypist model file (.pkl)",
            filetypes=[("CellTypist model", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            model = celltypist.models.Model.load(path)
            self.celltypist_model = model
            self.celltypist_model_path = path
            cell_types = getattr(model, "cell_types", None)
            n_types = len(cell_types) if cell_types is not None else "unknown"
            self.append_log(f"Loaded CellTypist model from {path}")
            self.append_log(f"Model recognizes {n_types} cell types")
            self.set_status(f"CellTypist model loaded: {os.path.basename(path)}. Predictions will refresh automatically.")
            # A new model invalidates any cached predictions made with the previous model.
            for other_node in self.model.nodes.values():
                other_node.predicted_labels = None
            self.append_log("Cleared cached cell type predictions for all nodes; they will be recomputed with the new model.")
            if self.selected_node_id and self.selected_node_id in self.model.nodes:
                selected_node = self.model.nodes[self.selected_node_id]
                self._maybe_predict_cell_types(selected_node)
                self._refresh_all_plot_views()
        except Exception as exc:
            self.append_log(f"Failed to load CellTypist model: {exc}")
            messagebox.showerror("Load CellTypist Model", f"Could not load model file:\n{exc}")

    def load_cell_types_csv(self):
        """Populate the Cell Type combo box options from a plain CSV file listing cell type
        names - every non-empty cell across every row/column in the file is treated as one
        candidate name (so this works whether the file is a single column, one row of many
        names, or a whole grid of names), in first-seen order, deduplicated. Unlike
        load_cell_types_for_annotation() this doesn't need any CellTypist model at all."""
        path = filedialog.askopenfilename(
            title="Select a CSV file listing cell type names",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            names: List[str] = []
            seen = set()
            with open(path, newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    for cell in row:
                        name = str(cell).strip()
                        if not name or name in seen:
                            continue
                        seen.add(name)
                        names.append(name)
            names = [n for n in names if n != NOT_ASSIGNED_CELL_TYPE_LABEL]
            if not names:
                raise ValueError("No cell type names were found in this file.")
            before = len(self.available_cell_types)
            self.available_cell_types.update(names)
            added = len(self.available_cell_types) - before
            self.refresh_node_list()
            self.append_log(f"Loaded {len(names)} cell type name(s) from {path} ({added} new) for the annotation combo boxes.")
            self.set_status(f"Loaded {len(names)} cell type(s) from {os.path.basename(path)} ({added} new).")
        except Exception as exc:
            self.append_log(f"Failed to load cell types CSV: {exc}")
            messagebox.showerror("Load Cell Types CSV", f"Could not load cell types file:\n{exc}")

    def add_custom_cell_type(self):
        """Manually add one cell type name to the Cell Type combo box options - for labels that
        aren't in any loaded CellTypist model (e.g. a custom annotation scheme), since the
        dropdown otherwise only ever offers what a model's cell_types list or its predictions
        already contain."""
        name = simpledialog.askstring(
            "Add Custom Cell Type",
            "Cell type name to add to the dropdown list:",
            parent=self,
        )
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showwarning("Add Custom Cell Type", "Cell type name cannot be blank.")
            return
        if name == NOT_ASSIGNED_CELL_TYPE_LABEL:
            messagebox.showwarning("Add Custom Cell Type", f"'{NOT_ASSIGNED_CELL_TYPE_LABEL}' is reserved and can't be added.")
            return
        if name in self.available_cell_types:
            self.set_status(f"'{name}' is already in the cell type list.")
            return
        self.available_cell_types.add(name)
        self.refresh_node_list()
        self.append_log(f"Added custom cell type '{name}' to the annotation combo boxes.")
        self.set_status(f"Added '{name}' to the cell type list.")

    def load_cell_types_for_annotation(self):
        """Populate the Cell Type combo box options in the Tree Nodes list. By default this uses
        whichever CellTypist model is already loaded (e.g. the default model auto-loaded at
        startup) - reading its full `cell_types` list, not just whatever's been predicted so far.
        If no model is loaded yet, prompts for a .pkl to read the list from (without necessarily
        making it the active prediction model)."""
        if celltypist is None:
            messagebox.showerror(
                "Load Cell Types",
                "The celltypist package is not installed.\nInstall with: pip install celltypist",
            )
            return

        model = self.celltypist_model
        model_source = self.celltypist_model_path

        if model is None:
            path = filedialog.askopenfilename(
                title="Select a CellTypist model (.pkl) to load cell types from",
                filetypes=[("CellTypist model", "*.pkl"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                model = celltypist.models.Model.load(path)
            except Exception as exc:
                self.append_log(f"Failed to load model for cell types: {exc}")
                messagebox.showerror("Load Cell Types", f"Could not load model file:\n{exc}")
                return
            model_source = path

        cell_types = getattr(model, "cell_types", None)
        if cell_types is None or len(cell_types) == 0:
            messagebox.showwarning("Load Cell Types", "The selected model does not expose a list of cell types.")
            return

        before = len(self.available_cell_types)
        self.available_cell_types.update(str(c) for c in cell_types)
        added = len(self.available_cell_types) - before
        self.refresh_node_list()
        source_label = os.path.basename(str(model_source)) if model_source else "the loaded model"
        self.append_log(f"Loaded {len(cell_types)} cell types from '{source_label}' ({added} new) for the annotation combo boxes.")
        self.set_status(f"Loaded {len(cell_types)} cell types for annotation from {source_label}.")

    def _maybe_predict_cell_types(self, node: Optional[ClusterNode]):
        """Auto-run CellTypist prediction for `node` using whichever model is currently loaded
        (self.celltypist_model), if it hasn't been predicted yet. Mirrors the caching pattern
        used for node.umap_adata: only recompute when node.predicted_labels is None. Respects
        the 'Cell Type Prediction Tool' dropdown - if the user has explicitly set it to 'None', this
        convenience auto-run is skipped too, not just the main Process button's own annotation
        stage."""
        if self.annotation_tool_var.get().strip() == "None":
            return
        if node is None:
            return
        if celltypist is None or self.celltypist_model is None:
            return
        if node.predicted_labels is not None:
            return
        cluster_key = node.last_cluster_key
        umap_adata = node.umap_adata
        if not cluster_key or umap_adata is None or cluster_key not in umap_adata.obs.columns:
            return
        self._run_celltypist_prediction(node)

    def _run_celltypist_prediction(self, node: ClusterNode):
        cluster_key = node.last_cluster_key
        umap_adata = node.umap_adata
        stage_label = f"CellTypist | {node.label}"
        try:
            common_barcodes = node.adata.obs_names.intersection(umap_adata.obs_names)
            if len(common_barcodes) == 0:
                raise ValueError("No matching cells between raw counts and cluster labels.")
            plot_adata = self._build_normalized_node_copy(node, cluster_key, umap_adata, common_barcodes, stage_label=stage_label)
            self.append_log(f"[{stage_label}] Running celltypist.annotate on {plot_adata.n_obs} cells x {plot_adata.n_vars} genes")
            predictions = celltypist.annotate(plot_adata, model=self.celltypist_model, majority_voting=False)
            predicted = predictions.predicted_labels["predicted_labels"]
            node.predicted_labels = predicted
            new_types = set(predicted.astype(str).unique().tolist())
            if not new_types.issubset(self.available_cell_types):
                self.available_cell_types.update(new_types)
                self.refresh_node_list()
            counts = predicted.value_counts()
            preview = ", ".join(f"{k}: {v}" for k, v in counts.head(10).items())
            self.append_log(f"[{stage_label}] Predicted {predicted.nunique()} distinct cell types across {len(predicted)} cells")
            self.append_log(f"[{stage_label}] Top counts -> {preview}")
            self._auto_assign_child_cell_types(node, stage_label=stage_label)
        except Exception as exc:
            self.append_log(f"[{stage_label}] Prediction failed: {exc}")

    def _auto_assign_child_cell_types(self, node: ClusterNode, stage_label: str):
        """After a fresh CellTypist prediction on `node`, check each of its existing child
        nodes - each one corresponds to a single Leiden cluster from node's last clustering,
        identified by child.cluster_value - and automatically set that child's assigned_cell_type
        based on how dominant its top predicted cell type is (the same bar-dominance the stacked
        bar chart visualizes):
          - top type >= "Auto-assign at" threshold (self.auto_assign_threshold_var): assign that
            type directly.
          - top type < "Not clear below" threshold (self.not_clear_threshold_var): the bar is
            fragmented enough that no single type is really dominant, so assign the sentinel
            NOT_CLEAR_CELL_TYPE_LABEL instead of guessing - this is a real answer ("this cluster's
            identity is genuinely ambiguous"), not a placeholder, and it's picked the same way any
            other cell type would be.
          - in between the two thresholds: leave the child unassigned for manual review, exactly
            as before this feature existed.
        Never overwrites a child that already has an assigned_cell_type (whether set manually,
        by an earlier auto-assign run, or previously marked Not_clear) - only fills in children
        that are still unassigned, so this is always safe to re-run after a fresh prediction
        without clobbering deliberate choices. To let a child be re-evaluated on the next run,
        set it back to 'Not assigned' first via its row's combo box.

        Also refuses to run at all - no auto-assigned types, no Not_clear either - if the 'Cell
        Type Prediction Tool' dropdown is set to 'None'. Both call sites into this already check
        that, but the guard lives here too so no future call path can bypass it."""
        if self.annotation_tool_var.get().strip() == "None":
            return
        if not node.children:
            return
        cluster_key = node.last_cluster_key
        umap_adata = node.umap_adata
        if not cluster_key or umap_adata is None or cluster_key not in umap_adata.obs.columns:
            return
        if node.predicted_labels is None:
            return

        auto_assign_text = self.auto_assign_threshold_var.get().strip()
        not_clear_text = self.not_clear_threshold_var.get().strip()
        try:
            auto_assign_fraction = float(auto_assign_text) / 100.0
            not_clear_fraction = float(not_clear_text) / 100.0
            if not (0 < auto_assign_fraction <= 1) or not (0 <= not_clear_fraction < 1):
                raise ValueError
            if not_clear_fraction >= auto_assign_fraction:
                self.append_log(
                    f"[{stage_label}] 'Not clear below' ({not_clear_text}%) must be lower than "
                    f"'Auto-assign at' ({auto_assign_text}%) - skipping auto-assignment for this run."
                )
                return
        except ValueError:
            self.append_log(
                f"[{stage_label}] '{auto_assign_text}%' / '{not_clear_text}%' are not valid "
                f"thresholds (enter numbers between 0 and 100, with 'Not clear below' smaller "
                f"than 'Auto-assign at') - skipping auto-assignment for this run."
            )
            return

        common_barcodes = umap_adata.obs_names.intersection(node.predicted_labels.index)
        if len(common_barcodes) == 0:
            return
        clusters = umap_adata.obs.loc[common_barcodes, cluster_key].astype(str)
        predicted = node.predicted_labels.loc[common_barcodes].astype(str)

        assigned_count = 0
        for child_id in node.children:
            child = self.model.nodes.get(child_id)
            if child is None or child.cluster_value is None:
                continue
            if child.assigned_cell_type:
                continue  # don't override an existing manual or earlier auto-assignment
            mask = clusters == child.cluster_value
            cluster_cell_count = int(mask.sum())
            if cluster_cell_count == 0:
                continue
            counts = predicted[mask].value_counts()
            top_type = str(counts.index[0])
            top_fraction = float(counts.iloc[0]) / cluster_cell_count
            if top_fraction >= auto_assign_fraction:
                child.assigned_cell_type = top_type
                self.available_cell_types.add(top_type)
                assigned_count += 1
                self.append_log(
                    f"[{stage_label}] Auto-assigned '{top_type}' to {child.label} - "
                    f"{top_fraction:.0%} of its {cluster_cell_count} predicted cells match "
                    f"(threshold {auto_assign_fraction:.0%})."
                )
            elif top_fraction < not_clear_fraction:
                child.assigned_cell_type = NOT_CLEAR_CELL_TYPE_LABEL
                self.available_cell_types.add(NOT_CLEAR_CELL_TYPE_LABEL)
                assigned_count += 1
                self.append_log(
                    f"[{stage_label}] Marked {child.label} as '{NOT_CLEAR_CELL_TYPE_LABEL}' - "
                    f"top predicted type '{top_type}' only covers {top_fraction:.0%} of its "
                    f"{cluster_cell_count} cells, below the {not_clear_fraction:.0%} threshold "
                    f"(no type is clearly dominant)."
                )
            else:
                self.append_log(
                    f"[{stage_label}] Left {child.label} unassigned - top predicted type "
                    f"'{top_type}' covers {top_fraction:.0%} of its {cluster_cell_count} cells, "
                    f"between the {not_clear_fraction:.0%} and {auto_assign_fraction:.0%} "
                    f"thresholds (review manually via its row's combo box)."
                )
        if assigned_count:
            self.refresh_node_list()


    def draw_stacked_bar_plot(self, node: Optional[ClusterNode], target_frame=None, big: bool = False, figsize_override=None):
        frame = target_frame if target_frame is not None else getattr(self, "stackedbar_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()

        def show_message(text: str):
            ttk.Label(frame, text=text, style="Panel.TLabel", wraplength=260, justify="left").pack(anchor="w", pady=6, padx=4)

        if Figure is None or FigureCanvasTkAgg is None or plt is None:
            show_message("Matplotlib is not installed; cannot render the cell type distribution.")
            return
        if node is None:
            show_message("No node selected.")
            return
        cluster_key = node.last_cluster_key
        umap_adata = node.umap_adata
        if not cluster_key or umap_adata is None or cluster_key not in umap_adata.obs.columns:
            show_message("Cell type distribution will appear after this node is clustered.")
            return
        if node.predicted_labels is None:
            show_message("Load a CellTypist model and click 'Predict Cell Types' to see the distribution.")
            return

        try:
            common_barcodes = umap_adata.obs_names.intersection(node.predicted_labels.index)
            if len(common_barcodes) == 0:
                show_message("No matching cells between cluster labels and CellTypist predictions.")
                return
            clusters = umap_adata.obs.loc[common_barcodes, cluster_key].astype(str)
            predicted = node.predicted_labels.loc[common_barcodes].astype(str)
            crosstab = pd.crosstab(clusters, predicted, normalize="index")
            try:
                crosstab = crosstab.reindex(sorted(crosstab.index, key=lambda x: int(x)))
            except (ValueError, TypeError):
                crosstab = crosstab.sort_index()

            # Order the stacked segments by each predicted cell type's TOTAL size across the
            # whole node (not the per-cluster fraction), largest first. pandas draws stacked
            # bars bottom-to-top in column order, so putting the biggest type first here is
            # what puts it at the bottom of every bar.
            type_totals = predicted.value_counts()  # descending by count already
            ordered_types = [t for t in type_totals.index if t in crosstab.columns]
            crosstab = crosstab.reindex(columns=ordered_types)

            # Default to generous spacing so labels/legend stay readable; the popup's size
            # controls let the user override this with something smaller (e.g. 6x6).
            if figsize_override is not None:
                fig_width, fig_height = figsize_override
            else:
                fig_width = max(8.0, 0.75 * len(crosstab.index) + 3.5)
                fig_height = 6.2
            fig = Figure(figsize=(fig_width, fig_height), dpi=200, facecolor="white")
            ax = fig.add_subplot(111)
            colors = self._qualitative_colors(len(crosstab.columns))
            crosstab.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Fraction of cells", fontsize=8, color="black")
            ax.set_xlabel(f"{cluster_key} cluster", fontsize=8, color="black")
            ax.set_title(f"Predicted cell types: {node.label}", fontsize=9, color="black")
            ax.tick_params(labelsize=7, colors="black", rotation=0)
            for spine in ax.spines.values():
                spine.set_color("black")
            ax.legend(fontsize=6, title="Predicted type", title_fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            fig.tight_layout()

            if big:
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
                canvas.get_tk_widget().pack(expand=True, padx=4, pady=4)
                self._attach_plot_toolbar(canvas, frame)
                self.stackedbar_canvas = canvas
                self.stackedbar_figure = fig
            else:
                self._display_figure_scaled_to_fit(fig, frame, cache_attr="stackedbar")
        except Exception as exc:
            self.append_log(f"[Stacked bar | {node.label}] Failed to render cell type distribution: {exc}")
            show_message(f"Could not render cell type distribution: {exc}")

    def select_node(self, node_id: str):
        self.selected_node_id = node_id
        self.refresh_node_list()
        self._maybe_predict_cell_types(self.model.nodes[node_id])
        self._refresh_all_plot_views()
        node = self.model.nodes[node_id]
        self.set_status(f"Selected {node.label} with {node.n_cells()} cells.")

    def select_and_process(self, node_id: str):
        self.select_node(node_id)
        self.process_selected_node()

    def select_and_toggle_save(self, node_id: str):
        self.select_node(node_id)
        self.toggle_save_selected()


if __name__ == "__main__":
    app = ClusterTreeApp()
    app.mainloop()