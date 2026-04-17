# TopoTENet

Topology-aware **E(3)-equivariant** neural network for physically consistent piezoelectric tensor prediction.

> Accurate prediction of piezoelectric tensors in crystals remains challenging because valid responses must satisfy both continuous geometric symmetries and discrete crystallographic constraints under periodic boundary conditions. TopoTENet integrates explicit periodic-topology encoding from SLICES with topology-conditioned E(3)-equivariant message passing, followed by a deterministic point-group projection step that removes residual symmetry-forbidden components.

---

## Key Features

- **Complete SLICES Integration** — Full tokenization of all 27 periodic boundary categories
- **Topology-conditioned E(3)-equivariant Attention** — Learns to weight cross-boundary vs intra-cell edges via angular/cutoff modulation, attention bias, and FiLM gating
- **Deterministic Symmetry Enforcement** — Hard crystallographic constraints via point-group averaging
- **State-of-the-art Performance** — MAE of **0.132 C/m²** and RMSE of **0.302 C/m²** on 369 test crystals spanning 79 space groups

---

## Architecture Overview

<p align="center">
  <img src="assets/model.png" width="85%" />
</p>

**TopoTENet** combines SLICES edge labels and space-group information with radial basis functions, spherical harmonics, and node features. The interaction block fuses topology with geometry through:

- Angular and cutoff modulation gated by topology
- Additive topological bias in attention logits
- MoE-gated key/value generation
- Edge-to-node FiLM modulation

The final output is passed through a point-group projection layer that guarantees exact crystallographic compatibility.

---

## Repository Structure

```
TopoTENet/
├── TopoTENet.py                  # Main training script
├── symmetry.py                   # Point-group symmetry projection utilities
├── utils.py                      # Helper functions for data loading and preprocessing
├── network_classes.py            # Auxiliary network modules
├── generate_sllices_from_dataset.py  # Generate SLICES representations from crystal dataset
├── assets/                       # Figures and visualizations
│   ├── model.pdf
│   ├── nature_level_performance.png
│   ├── projection_effects_single.png
│   ├── fig_attn.png
│   └── nature_case_panels_combined.png
└── README.md
```

---

## Installation

```bash
git clone https://github.com/future3317/TopoTENet.git
cd TopoTENet
pip install -r requirements.txt
```

### Install SLICES

TopoTENet depends on the [SLICES](https://github.com/xiaohang007/SLICES.git) package for periodic-topology encoding. Please install it separately:

```bash
git clone https://github.com/xiaohang007/SLICES.git
cd SLICES/SLICES-3.1.0
pip install -e .
```

### Core Dependencies

- `torch`
- `e3nn`
- `spglib`
- `pymatgen`
- `numpy`, `scipy`, `matplotlib`

> You can generate `requirements.txt` from your environment via `pip freeze > requirements.txt`.

---

## Usage

### 1. Prepare SLICES representations

```bash
python generate_sllices_from_dataset.py
```

This script processes raw crystal structures (e.g., from the Materials Project) and generates SLICES token sequences used by TopoTENet.

### 2. Train TopoTENet

```bash
python TopoTENet.py
```

`TopoTENet.py` is the main training script. It imports the model architecture from `network_classes.py` and data utilities from `utils.py`. Hyperparameters (e.g., learning rate, batch size, cutoff radius) can be adjusted at the top of the file or via command-line arguments if added.

### 3. Run symmetry projection

The symmetry projection utilities in `symmetry.py` can be imported to enforce point-group constraints on predicted tensors:

```python
from symmetry import apply_pointgroup_projection

projected_tensor = apply_pointgroup_projection(tensor, space_group_number)
```

---

## Main Results

<p align="center">
  <img src="assets/nature_level_performance.png" width="70%" />
</p>

| Method | MAE (C/m²) | RMSE (C/m²) |
|--------|------------|-------------|
| **TopoTENet (Ours)** | **0.132** | **0.302** |
| GMTNet | 0.158 | 0.335 |
| EATGNN | 0.141 | 0.391 |
| SEGNN | 0.172 | 0.405 |
| GoeCTP | 0.189 | 0.389 |
| MatTen | 0.189 | 0.495 |
| ALIGNN | 0.251 | 0.482 |

### Physical Consistency

- **Rotational equivariance**: Mean absolute error of $3.2 \times 10^{-6}$ C/m² across 1000 random rotations
- **Supercell invariance**: Mean absolute error of $2.1 \times 10^{-4}$ C/m² between unit cells and $2 \times 2 \times 2$ supercells
- **Symmetry projection**: Suppresses symmetry-forbidden tensor components by ~8 orders of magnitude

<p align="center">
  <img src="assets/projection_effects_single.png" width="60%" />
</p>

### Attention Analysis

TopoTENet selectively emphasizes specific periodic-topological contexts. Mixed boundary-crossing edges receive the highest attention weights, while the balance between intra-cell and cross-boundary contributions is material-dependent.

<p align="center">
  <img src="assets/fig_attn.png" width="85%" />
</p>

<p align="center">
  <img src="assets/nature_case_panels_combined.png" width="80%" />
</p>

---

## Acknowledgements

- **SLICES** — This repository includes the `SLICES-3.1.0` package for encoding periodic crystal topology. The original SLICES code is available at [https://github.com/xiaohang007/SLICES.git](https://github.com/xiaohang007/SLICES.git).
  - Xiao, H. et al. *SLICES: Sliced invariant canonical encoding for crystal structures.* (2023)
- Crystal structures and piezoelectric tensor data were obtained from the [Materials Project](https://materialsproject.org/).

---

## License

This project is released under the MIT License.
