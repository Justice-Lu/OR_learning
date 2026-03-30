# OR_learning

> **Structure-guided olfactory receptor learning with ESM2, AlphaFold3, and spatial deep learning**

<p align="center">
  <img src="./img/Model_design.png" alt="Graphical abstract of the OR_learning pipeline" width="100%" />
</p>

## Overview

`OR_learning` contains the notebooks, scripts, and helper functions used to study how **olfactory receptor (OR) sequence and structure relate to odor recognition**. The core idea is to combine:

- **ESM2** protein language model embeddings from OR amino acid sequences
- **AlphaFold3**-based OR structural models
- **pyKVFinder**-style binding cavity identification
- **Spatial voxel encoding** of cavity residues and features
- **CNN-based classifiers** for odor and receptor-level prediction

This repository is organized both as a **research workspace** and a **replication guide** for the spatial-LLM / “SmellLM” pipeline.

---

## Abstract in brief

A major challenge in olfaction is that we still lack a clear map connecting **olfactory receptors** to the **odorants** they detect. This project addresses that gap by integrating sequence-derived representations from **ESM2** with **AlphaFold3** structural models of ORs. Predicted binding cavities are identified, aligned, converted into voxelized spatial features, and then used to train deep learning models that classify odor-related response patterns. In the associated study, this framework achieved strong predictive performance across multiple odor categories and helped reveal cavity features likely to contribute to receptor selectivity.

---

## Pipeline at a glance

1. **Encode OR sequences with ESM2**  
   Generate residue- or sequence-level embeddings from primary amino acid sequence.

2. **Predict OR structures with AlphaFold3**  
   Build receptor–miniG-protein structural models and collect the resulting PDB files.

3. **Identify and align binding cavities**  
   Use cavity detection and structural alignment to define a shared **canonical binding cavity** across receptors.

4. **Build spatial feature tensors**  
   Map ESM-reduced residue features into a **3D voxel grid** centered on the cavity.

5. **Train downstream models**  
   Run CNN-based models for odor classification, chemical class prediction, OR classification, or binary tasks.

6. **Interpret latent structure**  
   Visualize learned embeddings, feature maps, and publication figures from saved outputs.

---

## Repository layout

| Path | Purpose | Typical use |
|---|---|---|
| `notebooks/binding_cavity/` | Stepwise exploratory and reproducible notebooks for cavity definition, AF3 processing, spatial encoding, and model setup | Start here to understand or rerun the pipeline |
| `scripts/` | Scripted versions of preprocessing and model training workflows | Use for batch runs or reproducible training |
| `utils/` | Shared helper functions for PDB parsing, voxelization, plotting, alignment, and analysis | Imported by both notebooks and scripts |
| `img/` | Static figures and graphical abstracts | README / documentation assets |
| `files/` | Local input data, metadata, embeddings, cavity pickles, screening tables | Required for replication; mostly not tracked in git |
| `output/` | Model outputs, figures, intermediate results, and evaluation artifacts | Inspect trained results and generated plots |
| `temp/` | Scratch space for temporary processing | Optional / disposable |

### Important data locations

For users trying to rerun the pipeline, the most relevant subfolders are:

- `files/ESM/` – saved sequence and residue embeddings, plus reduced ESM representations
- `files/binding_cavity/` – canonical cavity coordinates, residue-coordinate dictionaries, and reference centers
- `files/pS6IP/` – odor response / screening tables used for supervised learning
- `files/OR_seq/` – receptor sequence-related resources
- `output/spatialESM_OdorClass/` – odor classification runs and evaluation outputs
- `output/Features/` – downstream feature analyses and visualization outputs

---

## Recommended order for replication

If you want to **follow the project from raw inputs to trained models**, this is the most useful navigation path:

### 1) Sequence features
- `notebooks/binding_cavity/esm_embeddings.ipynb`
- `scripts/esm_embeddings.py`

Use these to generate or inspect **ESM2 embeddings** for mouse and human OR sequences.

### 2) Binding cavity definition and alignment
- `notebooks/binding_cavity/bc_0_binding_cavity.ipynb`
- `notebooks/binding_cavity/bc_01_Canonical_binding_cavity.ipynb`
- `notebooks/binding_cavity/bc_AF3_CBC.ipynb`
- `scripts/bc_cav_dump_AF3.py`
- `scripts/tmalign_pdb_dump_AF3.py`

These files define the **canonical binding cavity**, align structures, and extract cavity / residue coordinates.

### 3) Spatial encoding and voxel generation
- `notebooks/binding_cavity/bc_spatial_encoding_AF3.ipynb`
- `notebooks/binding_cavity/bc_spatial_feature_AF3.ipynb`
- `scripts/bc_Cbc_voxel_esm_dump.py`

These steps combine **cavity geometry + reduced ESM residue embeddings** into the spatial voxel representation used for learning.

### 4) Model training
- `scripts/cnn_spatialESM_Odorclassification_model_AF3.py`
- `scripts/cnn_spatialESM_Chemcalssification_model_AF3.py`
- `scripts/cnn_spatialESM_ORclassification_model_AF3.py`
- `scripts/cnn_spatialESM_binary_model_AF3.py`

These are the main training scripts for odor-related prediction tasks.

### 5) Figures and interpretation
- `notebooks/figures_code/Figures.ipynb`
- `output/pub_Figures/`
- `output/spatialESM_OdorClass/`
- `output/Features/`

Use these to review saved results, ROC curves, latent maps, and figure-generation code.

---

## Practical notes before running

### Data availability
Large inputs and intermediates are stored under `files/` and `output/`, and many of them are intentionally **git-ignored**. Full replication therefore requires access to the local datasets used in development, including:

- pS6 screening / response tables
- OR metadata and mappings
- ESM embedding files
- cavity coordinate pickles
- AF3-derived structure outputs

### Path assumptions
Several notebooks and scripts currently use **absolute local paths** such as:

```python
/mnt/data2/Justice/OR_learning/...
```

If you run this repository on another machine, update those paths or define a shared project root before execution.

### Archived material
Older exploratory work is kept in:

- `notebooks/old/`
- `scripts/old/`

These are useful for history and troubleshooting, but the **AF3-based notebooks and scripts** are the best place to start for current replication.

---

## Key idea of the project

Instead of treating ORs only as linear sequences, this repository builds a **structure-aware representation of the receptor binding cavity**. By placing reduced language-model features into a shared 3D cavity frame, the models can learn spatial patterns associated with odor responsiveness. In that sense, the project aims to move toward a **structural map of smell**.

---

## Suggested starting point for new users

If you are opening this repository for the first time, the quickest path is:

1. Read this README
2. Open `notebooks/binding_cavity/bc_01_Canonical_binding_cavity.ipynb`
3. Then review `notebooks/binding_cavity/bc_spatial_encoding_AF3.ipynb`
4. Finally run or inspect `scripts/cnn_spatialESM_Odorclassification_model_AF3.py`

That sequence gives the clearest view of **how structures are prepared, encoded, and used for prediction**.
