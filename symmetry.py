#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point group hard projection module based on pymatgen
Improved version: Combines accurate Hall number retrieval with comprehensive validation
"""

import torch
from functools import lru_cache
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from typing import Dict, Tuple
import numpy as np


def kronecker3(R: torch.Tensor) -> torch.Tensor:
    """
    Calculate R⊗R⊗R for third-order tensor symmetry operations

    Args:
        R: [3, 3] rotation matrix

    Returns:
        [27, 27] rotation operation in third-order tensor representation
    """
    return torch.kron(R, torch.kron(R, R))


def find_hall_number_for_sgnum(sgnum: int) -> int:
    """
    Find the first Hall number corresponding to space group number by searching

    Args:
        sgnum: space group number (1-230)

    Returns:
        corresponding Hall number
    """
    try:
        import spglib
    except ImportError:
        raise ImportError("Need to install spglib library: pip install spglib")

    # Search Hall numbers 1-530
    for hall_number in range(1, 531):
        try:
            sg_type = spglib.get_spacegroup_type(hall_number)
            if sg_type and getattr(sg_type, 'number', None) == sgnum:
                return hall_number
        except:
            continue

    raise ValueError(f"Could not find Hall number for space group {sgnum}")


@lru_cache(maxsize=None)
def build_pointgroup_projector_from_sgnum(sgnum: int, device: torch.device) -> torch.Tensor:
    """
    Build point group projector from space group number (improved spglib method)

    Args:
        sgnum: space group number (1-230)
        device: PyTorch device

    Returns:
        [27, 27] projection matrix P = (1/|G|)∑R⊗R⊗R
    """
    try:
        import spglib

        # Dynamically find the correct Hall number
        hall_number = find_hall_number_for_sgnum(sgnum)

        # Get symmetry operations using the correct Hall number
        symm = spglib.get_symmetry_from_database(hall_number)
        if symm is None:
            raise ValueError(f"Could not get symmetry operations for Hall number {hall_number} (space group {sgnum})")

        rotations = symm["rotations"]  # [n_ops, 3, 3]

        # Deduplicate to get point group rotation matrices
        seen = set()
        unique_rotations = []
        for R in rotations:
            key = tuple(R.reshape(-1).tolist())
            if key not in seen:
                seen.add(key)
                unique_rotations.append(R.astype(float))

        if not unique_rotations:
            raise ValueError(f"Space group {sgnum} has no valid rotation operations")

        # Verify if operation count is reasonable
        expected_orders = {
            1: 1,      # P1
            2: 2,      # P-1
            216: 24,   # F-43m (Td)
            225: 48,   # Fm-3m (Oh)
            186: 12,   # P6_3mc
            152: 6,    # P3_121 (32)
        }

        if sgnum in expected_orders:
            expected = expected_orders[sgnum]
            if len(unique_rotations) != expected:
                print(f"Warning: Space group {sgnum} expected {expected} operations, got {len(unique_rotations)}")

    except ImportError:
        raise ImportError("Need to install spglib library: pip install spglib")
    except Exception as e:
        raise RuntimeError(f"Failed to build point group projector (space group {sgnum}): {e}")

    # Build projection matrix P = (1/|G|)∑R⊗R⊗R
    Rs = [torch.tensor(R, dtype=torch.get_default_dtype(), device=device) for R in unique_rotations]
    Ds = [kronecker3(R) for R in Rs]
    P = torch.stack(Ds, dim=0).mean(dim=0)  # [27, 27]

    return P


def apply_pointgroup_projection(
    tensor: torch.Tensor,
    space_group_numbers: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Apply point group projection to tensors

    Args:
        tensor: [B, 27] or [B, 3, 3, 3] tensor
        space_group_numbers: [B] space group number for each sample
        device: PyTorch device

    Returns:
        Projected tensor with same shape as input
    """
    if tensor.dim() == 2:
        # [B, 27] -> [B, 3, 3, 3]
        tensor = tensor.reshape(-1, 3, 3, 3)

    B = tensor.shape[0]

    # Flatten for matrix operations
    tensor_flat = tensor.reshape(B, 27)

    # Apply corresponding point group projection to each sample
    projected_flat = torch.empty_like(tensor_flat)

    for b in range(B):
        sgnum = int(space_group_numbers[b].item())
        P = build_pointgroup_projector_from_sgnum(sgnum, device)  # [27, 27]
        projected_flat[b] = tensor_flat[b] @ P.T

    # Restore original shape
    if tensor.dim() == 4:
        return projected_flat.reshape(B, 3, 3, 3)
    else:
        return projected_flat


def get_pointgroup_statistics(
    tensor: torch.Tensor,
    space_group_numbers: torch.Tensor,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Get point group projection statistics

    Args:
        tensor: [B, 27] or [B, 3, 3, 3] tensor
        space_group_numbers: [B] space group numbers
        device: PyTorch device

    Returns:
        Dictionary containing statistical information
    """
    if tensor.dim() == 2:
        tensor_3d = tensor.reshape(-1, 3, 3, 3)
    else:
        tensor_3d = tensor

    B = tensor_3d.shape[0]

    # Original tensor statistics
    original_nonzero = (tensor_3d.abs() > 1e-10).float().sum(dim=(1, 2, 3))

    # Projected statistics
    projected = apply_pointgroup_projection(tensor_3d, space_group_numbers, device)
    projected_nonzero = (projected.abs() > 1e-10).float().sum(dim=(1, 2, 3))

    # Difference before and after projection
    diff = torch.norm(projected.reshape(B, 27) - tensor_3d.reshape(B, 27), dim=1)

    return {
        'original_nonzero': original_nonzero,
        'projected_nonzero': projected_nonzero,
        'projection_difference': diff,
        'nonzero_change': projected_nonzero - original_nonzero
    }


