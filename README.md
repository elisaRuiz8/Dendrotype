# Cell Type Annotation GUI

A desktop application for annotating cell types in single-cell RNA-seq (scRNA-seq) data,
built around a hierarchical, tree-based clustering workflow that mirrors how this kind of
analysis actually gets done in practice: coarse split first, then drill into whatever still
looks mixed, one branch at a time.

## The problem this solves

Clustering rarely separates cell types cleanly on the first try, and the problem gets worse
as cell counts climb into the hundreds of thousands: some clusters mix multiple types
together, while others split one real type across several clusters due to batch effects.
Fixing this normally means manually rewriting scripts to subset, re-correct, and re-cluster
the messy parts over and over.

This app turns that workflow into an interactive tree. Each node holds a subset of cells;
clicking **Process** runs batch correction, clustering, and cell-type annotation on it in
whatever combination you check, in that fixed order. Clustering splits a node into child
nodes you can drill into again if they still look mixed - all the way down until each
remaining cluster is either confidently a single cell type or clearly ambiguous and flagged
as such, at which point the annotations get saved back into the h5ad file or exported as a
CSV covering every cell.

## Key features

- **Hierarchical clustering tree** - every node is an independent subset of cells that can be
  batch-corrected, clustered, and annotated on its own, any number of levels deep.
- **Two batch-correction paths** - BBKNN and scVI, selectable per run.
- **Leiden clustering** with per-node resolution, automatically re-numbered so cluster `0` is
  always the largest (rather than Leiden's own arbitrary numbering).
- **CellTypist integration** for automated cell-type prediction, with a two-threshold
  auto-assignment system: a cluster gets auto-labeled if one predicted type clears a
  configurable confidence bar, gets flagged `Not_clear` if nothing comes close, or is left for
  manual review in between.
- **Diagnostic sweep tools** - Resolution Sweep and separate BBKNN/scVI Neighbors Sweeps, each
  reporting cluster-count stability via Adjusted Rand Index across a parameter range, plus
  proper batch-mixing diagnostics (including an inverse-Simpson/iLISI-style "effective
  batches" measure, not just a simple cross-batch neighbor fraction, which turns out to be a
  much weaker signal than it first appears - see [Notable engineering decisions](#notable-engineering-decisions)).
  These exist because picking clustering parameters by eyeballing a UMAP is not a substitute
  for actually checking whether a result is stable.
- **Marker gene dot plots** - loadable from CSV, rendered through scanpy, with an interactive
  high-resolution zoom/pan viewer (mouse wheel or +/-/Fit buttons) for inspecting dense marker
  panels without dots overlapping.
- **Non-destructive resolution preview** - "New at Resolution..." on the UMAP and Predicted
  Cell Types panels lets you look at what a different resolution's clustering would actually
  look like, without touching the node's real, committed clustering - useful for sanity
  checking a candidate resolution (e.g. one the Resolution Sweep flagged as stable) before
  committing to it via Process.
- **HiDPI-aware rendering** - plots are rasterized at whatever pixel density the actual screen
  needs (detected at runtime), not a fixed guess, so they stay crisp on Retina/scaled displays
  without being redrawn from scratch on every resize.
- **Session save/load**, CSV/h5ad export, and a running processing log for every operation.

## Architecture

The analysis logic and the GUI are intentionally kept separate:

- `H5ADClusterService` - all the actual scRNA-seq logic (preprocessing, BBKNN/scVI
  correction, Leiden clustering, the sweep diagnostics). Takes an AnnData in, returns results
  out; has no Tkinter dependency at all.
- `ClusterTreeApp` (Tkinter) - the GUI layer: the tree view, the Process button, the plot
  panels, the popups. Calls into `H5ADClusterService` for everything scientific.

This split is what makes the sweep tools and diagnostics straightforward to reason about (and
would make it possible to put a different UI - a web frontend, for instance - in front of the
same service layer without touching the analysis code).

## Installation

```bash
git clone <this-repo-url>
cd <repo-directory>
pip install -r requirements.txt
python ann_cellType_app_fixed.py
```

`torch` should be installed with whichever build (CPU-only or a specific CUDA version)
matches your hardware - see the [PyTorch install guide](https://pytorch.org/get-started/locally/)
if you need a GPU build. On Linux, `tkinter` may need to be installed separately at the OS
level (`sudo apt-get install python3-tk`).

## Running in a container

`requirements.txt` is still the right way to declare dependencies even if this ends up in a
Dockerfile - a container doesn't replace it, it just gives you a reliable place to *run*
`pip install -r requirements.txt` once, consistently, instead of on every machine
individually. A minimal Dockerfile for this project would look roughly like:

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y python3-tk && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "ann_cellType_app_fixed.py"]
```

Note this app is a Tkinter desktop GUI, not a web app - a container built this way has no
display of its own, so running it remotely needs either X11 forwarding, a VNC setup, or (for
genuine multi-user, browser-based access) rewriting the UI layer with a web framework on top
of the existing `H5ADClusterService` analysis code, which has no Tkinter dependency itself.

## Usage

1. Load an `.h5ad` file.
2. Set the batch key, resolution, and neighbor count in the toolbar.
3. Select a node (starting with the root), check whichever of Batch Correction / Clustering /
   Cell Type Tool you want to run, and click **Process**.
4. If clustering ran, drill into any child node that still looks mixed and repeat.
5. Use the Resolution/Neighbors sweeps (Tools menu) if you're not confident a parameter choice
   is stable before committing to it.
6. Load a markers CSV to cross-check clusters against known marker genes.
7. Once every leaf is annotated (or deliberately left `Not_clear`), save the annotations to a
   new h5ad or export to CSV.

## Notable engineering decisions

A few things worth calling out, since they came from real debugging rather than being
designed in up front:

- **A subtle scVI correctness bug**: the batch-correction step was calling
  `SCVI.setup_anndata(..., categorical_covariate_keys=[batch_key])` instead of
  `batch_key=batch_key`. These are genuinely different mechanisms in scVI's API -
  `categorical_covariate_keys` is meant for *secondary* nuisance covariates layered on top of
  an already-specified `batch_key`, not a substitute for it. The model was never actually told
  what to correct for. This was found by building the Neighbors Sweep diagnostic first,
  noticing batch mixing was suspiciously weak and unresponsive to the neighbor parameter, and
  tracing it back to the setup call - a good example of a diagnostic tool surfacing a bug that
  code review alone hadn't caught.
- **Cross-batch fraction vs. effective batches**: an early batch-mixing metric (fraction of a
  cell's neighbors from a different batch) turned out to have a real blind spot - a cell whose
  cross-batch neighbors all come from just *one* other batch scores identically to a cell
  whose neighbors are evenly spread across every other batch, even though only the second case
  is genuinely well-mixed. Fixed by adding an inverse-Simpson (iLISI-style) "effective batches"
  measure alongside it, which distinguishes the two cases correctly.
- **A staleness bug in the annotation pipeline**: re-clustering a node replaced its UMAP/graph
  data but didn't invalidate its previously-computed CellTypist predictions, which are indexed
  by cell barcode. Any cluster whose cells didn't happen to overlap with the *old* prediction's
  barcodes would silently vanish from the stacked bar chart - and, worse, could have caused the
  auto-assignment logic to label a cluster based on a tiny, unrepresentative leftover sliver of
  cells rather than its true composition. Fixed by clearing predictions whenever a node is
  re-clustered, forcing a fresh, correctly-scoped prediction before any auto-assignment runs
  again.

## Status

Actively developed, iterated through repeated cycles of testing on real datasets and fixing
what broke. Not all clusters are expected to resolve to a single confident cell type - some
genuinely are ambiguous, and the app is built to surface that (`Not_clear`) rather than force
a guess.
