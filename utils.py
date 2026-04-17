import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
import json
import warnings
from torch_geometric.data import DataLoader
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atom
from ase.data import atomic_numbers
from jarvis.core import specie
from sklearn.preprocessing import OneHotEncoder
from jarvis.core.specie import Specie
from e3nn import o3, nn
from torch.nn import Linear
from e3nn.nn import FullyConnectedNet
from torch_scatter import scatter
import math
import torch.nn.functional as F
from e3nn.o3 import Irreps
from e3nn.nn import Gate
from e3nn.util.jit import compile_mode
from typing import Union, Dict
from torch_geometric.data import Data
from contextlib import contextmanager
import time

# 常量定义
LABEL_CHAR2INT = {'o': 0, '+': 1, '-': -1}
CHARS = ['-', 'o', '+']
EDGE_LABEL_VOCAB = [''.join(t) for t in (a+b+c for a in CHARS for b in CHARS for c in CHARS)]
EDGE_LABEL_TO_ID = {label: idx for idx, label in enumerate(EDGE_LABEL_VOCAB)}
EDGE_LABEL_EMBED_DIM = 16
SPACE_GROUP_EMBED_DIM = 8
TOPO_FEATURE_DIM = EDGE_LABEL_EMBED_DIM + SPACE_GROUP_EMBED_DIM

# 全局变量
TENSOR_NORM_MEAN = None
TENSOR_NORM_STD = None


def tensor_stats(t):
    t = t.detach()
    # Handle NaN values by replacing them with zeros for statistics calculation
    t_clean = t.clone()
    t_clean[torch.isnan(t_clean)] = 0
    t_clean[torch.isinf(t_clean)] = 0

    return {
        "mean": float(t_clean.mean()),
        "std":  float(t_clean.std()),
        "min":  float(t_clean.min()),
        "max":  float(t_clean.max()),
        "nan":  int(torch.isnan(t).sum()),
        "inf":  int(torch.isinf(t).sum()),
    }


def denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    # No need for denormalization, return original tensor directly
    return t


@contextmanager
def epoch_timer():
    start = time.time()
    yield
    dur = time.time() - start
    print(f"[Time] epoch_seconds={dur:.2f}")


def safe_item(x):
    try:
        return float(x)
    except:
        return None


def robust_scale(x, q=0.90, dim=None, eps=1e-6):
    """Compute robust scaling using quantiles"""
    s = torch.quantile(x.abs().float(), q=q, dim=dim, keepdim=True)
    return torch.clamp(s, min=eps)


def calculate_tensor_normalization(dataset):
    """
    Compute per-component normalization parameters (after outlier filtering, before resampling).
    """
    component_vectors: list[torch.Tensor] = []

    print("Calculating normalization parameters for 27D tensors...")

    for data in tqdm(dataset, desc="Calculating normalization"):
        if hasattr(data, 'energy'):
            tensor_cpu = data.energy.detach().cpu().reshape(-1)
            component_vectors.append(tensor_cpu)

    if not component_vectors:
        print("No tensor data found for normalization, using default values")
        return (
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )

    stacked = torch.stack(component_vectors, dim=0)

    print(f"Total tensor elements collected: {stacked.numel()}")
    print(f"Data range: [{stacked.min().item():.6f}, {stacked.max().item():.6f}]")

    mean_val = stacked.mean(dim=0)
    std_val = stacked.std(dim=0).clamp_min(1e-6)

    print("Normalization parameters (per-component):")
    print(f"  Mean range: [{mean_val.min().item():.6f}, {mean_val.max().item():.6f}]")
    print(f"  Std range:  [{std_val.min().item():.6f}, {std_val.max().item():.6f}]")

    return mean_val, std_val


def apply_tensor_normalization(dataset, mean_val, std_val, clip_extremes=False):
    """
    Apply per-component normalization to the 27D piezoelectric tensors.

    Args:
        dataset: Dataset to normalize
        mean_val: Normalization mean (per component)
        std_val: Normalization std (per component)
        clip_extremes: Whether to clip extreme values (ignored, kept for compatibility)
    """
    print("Applying tensor normalization to dataset (27D tensors)...")

    mean_flat_cpu = mean_val.detach().clone()
    std_flat_cpu = std_val.detach().clone()

    for i, data in enumerate(dataset):
        if hasattr(data, 'energy'):
            energy = data.energy
            original_shape = energy.shape
            mean_device = mean_flat_cpu.to(device=energy.device, dtype=energy.dtype)
            std_device = std_flat_cpu.to(device=energy.device, dtype=energy.dtype)
            energy_flat = energy.reshape(-1)
            normalized_flat = (energy_flat - mean_device) / std_device
            dataset[i].energy = normalized_flat.reshape(original_shape)

    return mean_val, std_val


def robust_loss_fn(pred_tensor, target_tensor, q=0.90):
    """
    Compute loss with robust scaling to handle amplitude mismatch
    """
    if pred_tensor.shape != target_tensor.shape:
        if pred_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError(f"Prediction and target batch sizes differ: {pred_tensor.shape} vs {target_tensor.shape}")
        pred_tensor = pred_tensor.reshape(pred_tensor.shape[0], -1)
        target_tensor = target_tensor.reshape(target_tensor.shape[0], -1)
    else:
        pred_tensor = pred_tensor.reshape(pred_tensor.shape[0], -1)
        target_tensor = target_tensor.reshape(target_tensor.shape[0], -1)

    # Compute robust scaling factors
    s_pred = robust_scale(pred_tensor, q=q)
    s_target = robust_scale(target_tensor, q=q)

    # Normalize predictions and targets
    pred_n = pred_tensor / s_target
    target_n = target_tensor / s_target

    # Compute loss
    loss = F.smooth_l1_loss(pred_n, target_n, beta=0.5, reduction="mean")

    return loss


def parse_slices_2010(full_slices_string: str):
    """
    Parse SLICES 2.0.10 string with strict validation.

    Args:
        full_slices_string: Raw SLICES string

    Returns:
        Tuple of (edge_indices, edge_labels, to_jimages)

    Raises:
        ValueError: If SLICES string is invalid or cannot be parsed
    """
    if not full_slices_string or not full_slices_string.strip():
        raise ValueError("Empty SLICES string provided")

    tokens = full_slices_string.strip().split()
    if not tokens:
        raise ValueError("SLICES string tokenization failed")

    # Find start of edge information
    start = 0
    while start < len(tokens) and not tokens[start].isdigit():
        start += 1

    if start >= len(tokens):
        raise ValueError("No edge information found in SLICES string")

    edge_tokens = tokens[start:]
    if len(edge_tokens) == 0:
        raise ValueError("No edge tokens found in SLICES string")

    # Parse edge information with strict validation
    edge_indices, edge_labels, to_jimages = [], [], []

    # Validate that we have complete triplets
    if len(edge_tokens) % 3 != 0:
        raise ValueError(f"SLICES string has incomplete edge information. "
                        f"Expected multiple of 3 tokens, got {len(edge_tokens)} tokens")

    for idx in range(0, len(edge_tokens), 3):
        a, b, label = edge_tokens[idx], edge_tokens[idx + 1], edge_tokens[idx + 2]

        # Strict validation of edge tokens
        if not (a.isdigit() and b.isdigit()):
            raise ValueError(f"Invalid node indices in SLICES string: '{a}', '{b}'. "
                           f"Expected integer node indices.")

        if len(label) != 3:
            raise ValueError(f"Invalid edge label length: '{label}'. Expected 3 characters.")

        if not all(char in LABEL_CHAR2INT for char in label):
            raise ValueError(f"Invalid characters in edge label: '{label}'. "
                           f"Allowed characters: {list(LABEL_CHAR2INT.keys())}")

        node_a, node_b = int(a), int(b)
        edge_indices.append([node_a, node_b])
        edge_labels.append(label)
        to_jimages.append(tuple(LABEL_CHAR2INT[char] for char in label))

    # Validate that we actually parsed edges
    if not edge_indices:
        raise ValueError("No valid edges found in SLICES string")

    # Validate node indices are reasonable
    max_node_idx = max(max(idx) for idx in edge_indices)
    if max_node_idx < 0:
        raise ValueError(f"Invalid node indices found: negative values detected")

    return edge_indices, edge_labels, to_jimages


def build_edge_shift_from_jimages(jimgs):
    import torch

    if not jimgs:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.as_tensor(jimgs, dtype=torch.float32)


def r_cut2D(x, cell):
    structure = AseAtomsAdaptor.get_structure(cell)
    cell = structure.lattice.matrix
    r_cut = max(np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), x)
    return r_cut


def datatransform_from_slices(crystal, property_value, full_slices_string, space_group_number, fea, device):
    # Parse SLICES string - this must succeed
    edge_indices, edge_labels, jimages = parse_slices_2010(full_slices_string)

    # SLICES parsing must produce edges - no fallback allowed
    if not edge_indices:
        raise ValueError(f"No edges found in SLICES string for crystal with {len(crystal)} atoms. "
                        f"SLICES parsing failed - this indicates invalid SLICES data.")

    if not edge_labels:
        raise ValueError(f"No edge labels found in SLICES string for crystal with {len(crystal)} atoms. "
                        f"SLICES parsing failed - this indicates invalid SLICES data.")

    # Create edge tensors from parsed SLICES data
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_shift = build_edge_shift_from_jimages(jimages)
    # Strict check that each label is in the dictionary, no silent fallback to 0
    if any(label not in EDGE_LABEL_TO_ID for label in edge_labels):
        unknown_labels = [label for label in edge_labels if label not in EDGE_LABEL_TO_ID]
        raise ValueError(f"Unknown SLICES edge labels detected: {unknown_labels}. "
                        f"All 27 labels should be present in EDGE_LABEL_VOCAB.")
    edge_label_ids = torch.tensor([EDGE_LABEL_TO_ID[label] for label in edge_labels], dtype=torch.long)

    property_tensor = torch.as_tensor(property_value, dtype=torch.float32)
    # Fix lattice dimension issue - ensure [1, 3, 3] format
    lattice_tensor = torch.as_tensor(crystal.cell.array, dtype=torch.float32)
    if lattice_tensor.dim() == 2:
        lattice_tensor = lattice_tensor.unsqueeze(0)  # [3, 3] -> [1, 3, 3]

    data = torch_geometric.data.Data(
        pos=torch.as_tensor(crystal.get_positions(), dtype=torch.float32),
        lattice=lattice_tensor,
        x=torch.as_tensor([fea[atomic_numbers[atom] - 1] for atom in crystal.symbols], dtype=torch.float32),
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_label=edge_labels,
        edge_label_id=edge_label_ids,
        space_group_number=int(space_group_number),
        energy=property_tensor.unsqueeze(0).to(device),
    )
    return data


def filter_outliers_by_quantile(dataset, quantile=0.95):
    """
    Filter outliers by piezoelectric tensor modulus

    Args:
        dataset: Original dataset
        quantile: Quantile threshold, default 0.95

    Returns:
        Filtered dataset
    """
    # --- Step 1: Calculate moduli for all relevant samples at once ---
    # Use a list to store (modulus, original data index)
    moduli_with_indices = []

    print("Calculating piezoelectric tensor moduli...")
    for i, data in enumerate(tqdm(dataset, desc="Calculating moduli")):
        if hasattr(data, 'energy'):
            tensor_flat = data.energy.view(-1)
            modulus = torch.norm(tensor_flat).item()
            moduli_with_indices.append((modulus, i)) # Store modulus and its index in the original dataset

    if not moduli_with_indices:
        print("Warning: No valid tensor data found")
        return dataset

    # Extract all modulus values for statistical calculation
    all_moduli = [item[0] for item in moduli_with_indices]

    # --- Step 2: Calculate threshold and print statistics (same as original) ---
    threshold = np.quantile(all_moduli, quantile)

    min_modulus = min(all_moduli)
    max_modulus = max(all_moduli)
    mean_modulus = np.mean(all_moduli)
    median_modulus = np.median(all_moduli)

    print(f"Original dataset statistics (for {len(all_moduli)} samples with energy field):")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Modulus range: [{min_modulus:.3f}, {max_modulus:.3f}]")
    print(f"  Mean modulus: {mean_modulus:.3f}")
    print(f"  Median modulus: {median_modulus:.3f}")
    print(f"  {quantile*100:.0f}% quantile threshold: {threshold:.3f}")

    # --- Step 3: Efficient filtering ---
    # Create a set containing all sample indices to keep, more efficient
    indices_to_keep = set(range(len(dataset)))
    outliers_count = 0

    outliers_to_print = []

    for modulus, index in moduli_with_indices:
        if modulus > threshold:
            indices_to_keep.remove(index)
            outliers_count += 1
            if outliers_count <= 5: # Collect first 5 outlier information
                outliers_to_print.append(f"  Outlier sample #{outliers_count} (original index {index}): modulus = {modulus:.3f}")

    # Print outlier information
    for line in outliers_to_print:
        print(line)

    # Build filtered dataset based on indices
    filtered_dataset = [dataset[i] for i in sorted(list(indices_to_keep))]

    print(f"\nFiltering results:")
    print(f"  Outliers removed: {outliers_count}")
    print(f"  Samples retained: {len(filtered_dataset)}")
    print(f"  Filter ratio: {outliers_count/len(dataset)*100:.1f}%")

    return filtered_dataset


def compute_regression_sample_weights(train_data, high_value_threshold=1.42):
    """
    Construct per-sample weights to emphasise high-modulus tensors without duplicating data.

    Args:
        train_data: Training dataset before resampling.
        high_value_threshold: Threshold used only for reporting high-value coverage.
    """
    if not train_data:
        return torch.ones(0, dtype=torch.double), 0

    moduli = []
    indices = []
    for idx, data in enumerate(train_data):
        if hasattr(data, 'energy'):
            tensor_flat = data.energy.view(-1)
            modulus = torch.norm(tensor_flat).item()
            moduli.append(modulus)
            indices.append(idx)

    if not moduli:
        print("No energy tensors found in training data; using uniform sampling weights.")
        weights = torch.ones(len(train_data), dtype=torch.double)
        return weights, len(train_data)

    moduli = np.asarray(moduli)
    percentiles = [20, 40, 60, 80, 90, 95, 98]
    boundaries = [0.0] + [float(np.percentile(moduli, p)) for p in percentiles] + [float('inf')]
    oversample_factors = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]

    sample_factors = np.ones(len(train_data), dtype=float)
    bin_counts = np.zeros(len(oversample_factors), dtype=int)

    for local_idx, modulus in enumerate(moduli):
        for bin_idx in range(len(boundaries) - 1):
            lower, upper = boundaries[bin_idx], boundaries[bin_idx + 1]
            in_bin = lower <= modulus < upper or (bin_idx == len(boundaries) - 2 and modulus >= lower)
            if in_bin:
                sample_factors[indices[local_idx]] = oversample_factors[bin_idx]
                bin_counts[bin_idx] += 1
                break

    print("Regression-focused sampling strategy:")
    for bin_idx in range(len(oversample_factors)):
        lower = boundaries[bin_idx]
        upper = boundaries[bin_idx + 1]
        if np.isinf(upper):
            range_str = f"[{lower:.3f}, inf)"
        else:
            range_str = f"[{lower:.3f}, {upper:.3f})"
        print(f"  Range {range_str}: {bin_counts[bin_idx]} samples (weight: {oversample_factors[bin_idx]:.1f}x)")

    total_expected = int(np.round(sample_factors.sum()))
    total_expected = max(total_expected, len(train_data))

    weighted_high = float(
        sum(sample_factors[indices[i]] for i, modulus in enumerate(moduli) if modulus > high_value_threshold)
    )
    high_ratio = weighted_high / max(total_expected, 1)

    print("\nSampling weights summary:")
    print(f"  Original samples: {len(train_data)}")
    print(f"  Expected sampled instances: {total_expected} ({total_expected / max(len(train_data), 1):.2f}x)")
    print(f"  Weighted high-value samples (>{high_value_threshold:.2f}): {weighted_high:.1f}/{total_expected} ({high_ratio * 100:.1f}%)")

    weights = torch.as_tensor(sample_factors, dtype=torch.double)
    return weights, total_expected
