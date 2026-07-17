import json
import torch
# from torch_scatter import scatter
from e3nn import o3, nn

import matplotlib.pyplot as plt

import ase.neighborlist
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import os
import pandas as pd
from datetime import datetime

from jarvis.core import specie
from sklearn.preprocessing import OneHotEncoder
from jarvis.core.specie import Specie,get_node_attributes


import warnings
import torch_geometric
from ase.atoms import Atom
from ase.data import atomic_numbers
# from ase.io import read

from e3nn.math import soft_one_hot_linspace,soft_unit_step
from torch_scatter import scatter

import torch_scatter
from e3nn.nn import Gate
from e3nn.util.jit import compile_mode

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch.utils.data import WeightedRandomSampler
from e3nn.io import CartesianTensor
from tqdm import tqdm
import numpy as np
from pandarallel import pandarallel
import math
import torch.nn.functional as F
from e3nn.o3 import Irreps
from typing import  Dict, Union
from e3nn.nn import BatchNorm
from pymatgen.core.tensors import Tensor
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Linear
from e3nn.nn import FullyConnectedNet
from torch.nn.utils import clip_grad_norm_
import time
from contextlib import contextmanager

# Import symmetry projection module
from symmetry import apply_pointgroup_projection

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

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
warnings.filterwarnings("ignore")
# torch.set_float32_matmul_precision("high")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pandarallel.initialize(progress_bar=True, verbose=True)
# torch.autograd.set_detect_anomaly(True)

TENSOR_NORM_MEAN: torch.Tensor | None = None
TENSOR_NORM_STD: torch.Tensor | None = None

n=-1
batch_size =128
epochs=50
#define the multihead attention
heads=2
lmax=4
STD_ALIGNMENT_LAMBDA = 5e-3


class EquivariantLayerNormFast(torch.nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
            self.affine_bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        '''
            Use torch layer norm for scalar features.
        '''

        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw:(iw + mul)]
                bias = self.affine_bias[ib:(ib + mul)]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
                continue

            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw:(iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim

        output = torch.cat(fields, dim=-1)
        return output






@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)

#from equiformer
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


#from matten
class ShiftedSoftPlus(torch.nn.Module):
    """
    Shifted softplus as defined in SchNet, NeurIPS 2017.

    :param beta: value for the a more general softplus, default = 1
    :param threshold: values above are linear function, default = 20
    """

    _log2: float

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x):
        """
        Evaluate shifted softplus

        :param x: torch.Tensor, input
        :return: torch.Tensor, ssp(x)
        """
        return self.softplus(x) - self._log2


ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus(),
        "silu": torch.nn.functional.silu,
        "sigmoid": torch.sigmoid,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


def find_positions_in_tensor_fast(tensor):
    """
    Optimized function to find positions of each unique element in a PyTorch tensor
    using advanced indexing and broadcasting, keeping outputs as tensors.

    Parameters:
    tensor (torch.Tensor): The input tensor to analyze.

    Returns:
    dict: A dictionary where each key is a unique element from the tensor,
          and the value is a tensor of indices where this element appears.
    """
    unique_elements, inverse_indices = torch.unique(tensor, sorted=True, return_inverse=True)
    positions = {}
    for i, element in enumerate(unique_elements):
        # Directly store tensors of positions
        positions[element.item()] = torch.nonzero(inverse_indices == i, as_tuple=True)[0]

    return positions


class Fromtensor(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.tensor = CartesianTensor(formula)
    def forward(self, data):
        return self.tensor.from_cartesian(data)


class Totensor(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.tensor = CartesianTensor(formula)

    def forward(self, data):
        return self.tensor.to_cartesian(data)





class TensorIrreps(torch.nn.Module):
    def __init__(self ,formula , conv_to_output_hidden_irreps_out):
        super().__init__()
        if formula is None:
            self.formula=formula
            self.irreps_in = conv_to_output_hidden_irreps_out
            # self.dropout=nn.Dropout(irreps=self.irreps_in,p=0.2)
            self.irreps_out = o3.Irreps('0e')
            self.extra_layers = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)
        else:
            self.formula=formula
            self.irreps_in = conv_to_output_hidden_irreps_out
            # self.dropout = nn.Dropout(irreps=self.irreps_in, p=0.2)
            self.irreps_out = CartesianTensor(formula=self.formula)

            self.extra_layers = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)

            # self.to_cartesian = Totensor(self.formula)

    def forward(self,data):
        # out=self.dropout(data)
        out=self.extra_layers(data)
        if self.formula is None:
            return out
        else:
            out = self.irreps_out.to_cartesian(out)
            return out


class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        node_attr:o3.Irreps,
        internal_and_share_weights: bool = False,
        # mlp_input_size: int = None,
        # mlp_hidden_size: int = 8,
        # mlp_num_hidden_layers: int = 1,
        # mlp_activation: Callable = ACTIVATION["e"]["ssp"],
    ):
        """
        UVU tensor product.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.

        """

        super().__init__()

        self.out=irreps_out
        self.node_attr=node_attr
        # self.dropout = nn.Dropout(irreps=irreps_in1,p=0.3)

        # uvu instructions for tensor product
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out or ir_out == o3.Irreps("0e"):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)

        assert irreps_mid.dim > 0, (
            f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
            f"instructions in irreps_out={irreps_out}"
        )

        # sort irreps_mid to let irreps of the same type be adjacent to each other
        self.irreps_mid, permutation, _ = irreps_mid.sort()

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.lin0=o3.FullyConnectedTensorProduct(irreps_in1, self.node_attr,irreps_in1)
        # self.dropout1=nn.Dropout(irreps=irreps_in1,p=0.2)
        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )
        self.dropout2 = nn.Dropout(irreps=self.irreps_mid, p=0.2)
        # self.lin=o3.Linear(irreps_in=self.irreps_mid,irreps_out=self.out)
        self.lin=o3.FullyConnectedTensorProduct(self.irreps_mid, self.node_attr,self.out)

        self.sc = o3.FullyConnectedTensorProduct(
            irreps_in1, self.node_attr, self.out
        )

    # def forward(
    #     self, data1: Tensor, data2: Tensor, data_weight: Tensor,data3:Tensor
    # ) -> Tensor:
    #     # if self.weight_nn is not None:
    #     #     assert data_weight is not None, "data for weight not provided"
    #     #     weight = self.weight_nn(data_weight)
    #     # else:
    #     #     weight = None
    #     x = self.tp(data1, data2, data_weight)
    #     x=self.lin(x)
    #
    #     return x


    def forward( self, data1: Tensor, data2: Tensor, data_weight: Tensor,data3:Tensor
    ) -> Tensor:
        node_feats = data1
        node_attrs = data3
        edge_attrs = data2
        # node_feats=self.dropout(node_feats)
        node_sc = self.sc(node_feats, node_attrs)
        # node_sc=self.dropout(node_sc)

        node_feats = self.lin0(node_feats, node_attrs)
        # node_feats=self.dropout1(node_feats)

        node_feats = self.tp(node_feats, edge_attrs, data_weight)
        node_feats=self.dropout2(node_feats)
        # node_feats=self.lin(node_feats,node_attrs)

        # update
        node_conv_out = self.lin(node_feats, node_attrs)
        # node_conv_out=self.dropout(node_conv_out)
        node_feats = node_sc + node_conv_out

        return node_feats

def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def _scalar_channel_mask(irreps):
    """Return a 1-D mask that is 1 on even-parity scalar (0e) channels and 0 elsewhere.

    Used to restrict FiLM additive biases to the equivariant scalar subspace.
    """
    mask = []
    for mul, ir in o3.Irreps(irreps):
        val = 1.0 if (ir.l == 0 and ir.p == 1) else 0.0
        mask.extend([val] * (mul * ir.dim))
    return torch.tensor(mask, dtype=torch.get_default_dtype())


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)

class ComposeWithTopo(torch.nn.Module):
    def __init__(self, conv, gate, norm) -> None:
        super().__init__()
        self.conv = conv
        self.gate = gate
        self.norm = norm

    def forward(self, node_attr, node_features, edge_src, edge_dst, edge_attr, edge_scalars, edge_length, topo_scalar, topo_bias, fpit):
        # Conv layer needs all topological parameters
        x = self.conv(node_attr, node_features, edge_src, edge_dst, edge_attr, edge_scalars, edge_length, topo_scalar, topo_bias, fpit)
        # Gate layer only needs the transformed features
        x = self.gate(x)
        # Norm layer only needs the gated features
        x = self.norm(x)
        return x

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


# Note: Target values now maintain original scale, no per-element normalization
# This resolves logical inconsistency between symmetry projection and normalization
p=pd.read_json("large_piezo_dataset_3x3x3_fixed.json")
# p=pd.read_json("pie20627.json")
with open("sllices_from_dataset_2.0.10.json", "r", encoding="utf-8") as f:
    slices_payload = json.load(f)
slices_results = slices_payload.get("results", [])
slices_by_mp = {}
for entry in slices_results:
    mp_id = entry.get("mp_id", "")
    if not mp_id:
        continue
    slices_by_mp.setdefault(mp_id, entry)

subset = p.iloc[:n] if n >= 0 else p

struct = []
dummy_energies = []
full_slices_strings = []
space_groups = []
for _, row in subset.iterrows():
    mp_id = row.get("mp_id")
    entry = slices_by_mp.get(mp_id)
    if entry is None:
        continue
    structure_dict = row["structure"]
    crystal = AseAtomsAdaptor.get_atoms(Structure.from_dict(structure_dict))
    struct.append(crystal)
    dummy_energies.append(row["total"])
    full_slices_strings.append(entry.get("full_slices_string", ""))
    space_groups.append(int(entry.get("space_group_number", row.get("space_group_number", 0))))

if not struct:
    raise ValueError("No structures matched between dataset and SLICES metadata.")

num_nodes=sum([len(i) for i in struct])/len([len(i) for i in struct])

radial_cutoff = 5
max_radius=7

try:
    encoder = OneHotEncoder(max_categories=6, sparse=False)
except:
    encoder = OneHotEncoder(max_categories=6,sparse_output=False)

fea = [Specie(Atom(i).symbol, source='magpie').get_descrp_arr for i in range(1, 102)]
fea = encoder.fit_transform(fea)

print(len(fea[0]))
dim=len(fea[0])
# fea=torch.as_tensor(fea)

# TR=Fromtensor('ijk=ikj')
dataset=[]

LABEL_CHAR2INT = {'o': 0, '+': 1, '-': -1}
# Generate complete 27 edge labels (3^3 = 27)
CHARS = ['-', 'o', '+']
EDGE_LABEL_VOCAB = [''.join(t) for t in (a+b+c for a in CHARS for b in CHARS for c in CHARS)]
EDGE_LABEL_TO_ID = {label: idx for idx, label in enumerate(EDGE_LABEL_VOCAB)}
EDGE_LABEL_EMBED_DIM = 16
SPACE_GROUP_EMBED_DIM = 8
TOPO_FEATURE_DIM = EDGE_LABEL_EMBED_DIM + SPACE_GROUP_EMBED_DIM


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


def r_cut2D(x,cell):
    structure=AseAtomsAdaptor.get_structure(cell)
    cell=structure.lattice.matrix
    r_cut = max(np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), x)
    #define the maximum r_cut values
    #r_cut=min(r_cut,9)
    return r_cut

def datatransform_from_slices(crystal, property_value, full_slices_string, space_group_number):
    # Parse SLICES string - this must succeed
    edge_indices, edge_labels, jimages = parse_slices_2010(full_slices_string)

    # SLICES parsing must produce edges - no fallback allowed
    if not edge_indices:
        raise ValueError(f"No edges found in SLICES string for crystal with {len(crystal)} atoms. "
                        f"SLICES parsing failed - this indicates invalid SLICES data.")

    if not edge_labels:
        raise ValueError(f"No edge labels found in SLICES string for crystal with {len(crystal)} atoms. "
                        f"SLICES parsing failed - this indicates invalid SLICES data.")

    # Node indices should match crystal structure exactly with new SLICES file
    # No validation needed as we've verified 100% consistency

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

for crystal, energy, slices_string, sg_number in zip(struct, dummy_energies, full_slices_strings, space_groups):
    data=datatransform_from_slices(crystal, energy, slices_string, sg_number)
    dataset.append(data)


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

# Apply outlier filtering
print("Starting outlier filtering...")
dataset = filter_outliers_by_quantile(dataset, quantile=0.95)


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

# print(dataset)

# dataset=shuffle(dataset)
train_ratio, valid_ratio = 0.8, 0.2

print("Splitting dataset into train/valid sets (8:2 split)...")
traindataset, validdataset = train_test_split(
    dataset,
    test_size=valid_ratio,
    random_state=42,
    shuffle=True,
)

print(f"  Train samples: {len(traindataset)}")
print(f"  Valid samples: {len(validdataset)}")

print("Computing regression-focused sampling weights for training set...")
train_sample_weights, expected_train_samples = compute_regression_sample_weights(traindataset)


# Skip per-element normalization of target values, keep original scale
print("Skipping tensor normalization - keeping original scale for target values")
TENSOR_NORM_MEAN = None
TENSOR_NORM_STD = None

# Print current train/valid piezoelectric tensor L2 norm range (original scale)
train_moduli = [torch.norm(data.energy.view(-1)).item() for data in traindataset if hasattr(data, 'energy')]
valid_moduli = [torch.norm(data.energy.view(-1)).item() for data in validdataset if hasattr(data, 'energy')]
if train_moduli:
    print(f"Training set tensor moduli range (original scale): [{min(train_moduli):.3f}, {max(train_moduli):.3f}]")
if valid_moduli:
    print(f"Validation set tensor moduli range (original scale): [{min(valid_moduli):.3f}, {max(valid_moduli):.3f}]")

print("=" * 50)

train_sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=expected_train_samples,
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)
train_dataloader = DataLoader(traindataset, batch_size=batch_size, sampler=train_sampler)
valid_dataloader = DataLoader(validdataset, batch_size=batch_size)

del dataset,traindataset,validdataset




# import torch.nn as nn

def infer_head_count(irreps_list, max_heads):
    max_heads = max(int(max_heads), 1)
    irreps_objects = [o3.Irreps(ir) for ir in irreps_list]
    for h in range(max_heads, 0, -1):
        divisible = True
        for irreps in irreps_objects:
            for mul, _ in irreps:
                if mul > 0 and mul % h != 0:
                    divisible = False
                    break
            if not divisible:
                break
        if divisible:
            return h
    return 1


def multiheadsplit(irreps, num_heads):
    irreps = o3.Irreps(irreps)
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    ll = []
    for mul, ir in irreps:
        if mul == 0:
            continue
        if mul % num_heads != 0:
            raise ValueError(f"Multiplicity {mul} of irrep {ir} is not divisible by num_heads={num_heads}")
        ll.append((mul // num_heads, ir))
    return o3.Irreps(ll)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way using PyTorch."""
    shiftx = x - torch.max(x)
    # exps = torch.exp(shiftx)
    return F.softmax(shiftx,dim=-1)
@compile_mode('script')
class Attention(torch.nn.Module):
    def __init__(self, node_attr,irreps_node_input, irreps_query, irreps_key, irreps_output, number_of_basis):
        super().__init__()
        # self.radial_cutoff = radial_cutoff
        self.node_attr = o3.Irreps(node_attr)
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.irreps_output = o3.Irreps(irreps_output)
        self.heads = infer_head_count(
            [self.irreps_query, self.irreps_key, self.irreps_output],
            heads,
        )
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        self.norm = EquivariantLayerNormFast(irreps=self.irreps_output)

        self.radial_layers = list(number_of_basis)
        self.topo_dim = TOPO_FEATURE_DIM
        self.num_relation_experts = 8
        self.h_q = o3.FullyConnectedTensorProduct(self.irreps_node_input, self.node_attr, self.irreps_query)
        # self.h_q = o3.Linear(irreps_node_input,self.irreps_query)


        self.tp_k=UVUTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_key,self.node_attr)
        # self.dropout1=nn.Dropout(irreps=self.irreps_node_input,p=0.2)
        hidden_layers = self.radial_layers[1:]
        input_dim = (self.radial_layers[0] if self.radial_layers else 0) + self.topo_dim
        fc_k_layers = [input_dim] + hidden_layers + [self.tp_k.tp.weight_numel]
        self.fc_k_experts = torch.nn.ModuleList(
            [FullyConnectedNet(fc_k_layers, act=torch.nn.functional.silu) for _ in range(self.num_relation_experts)]
        )

        self.tp_v=UVUTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_output,self.node_attr)
        # self.dropout2=nn.Dropout(irreps=self.irreps_output,p=0.2)
        fc_v_layers = [input_dim] + hidden_layers + [self.tp_v.tp.weight_numel]
        self.fc_v_experts = torch.nn.ModuleList(
            [FullyConnectedNet(fc_v_layers, act=torch.nn.functional.silu) for _ in range(self.num_relation_experts)]
        )
        self.expert_gate = torch.nn.Sequential(
            Linear(TOPO_FEATURE_DIM, 64),
            torch.nn.SiLU(),
            Linear(64, self.num_relation_experts)
        )
        self.angle_gate = torch.nn.Sequential(
            Linear(TOPO_FEATURE_DIM, 64),
            torch.nn.SiLU(),
            Linear(64, self.irreps_sh.dim)
        )
        self.cutoff_gate = torch.nn.Sequential(
            Linear(TOPO_FEATURE_DIM, 32),
            torch.nn.SiLU(),
            Linear(32, 1)
        )

        split_query = multiheadsplit(self.irreps_query, self.heads).simplify()
        split_key = multiheadsplit(self.irreps_key, self.heads).simplify()
        split_value = multiheadsplit(self.irreps_output, self.heads).simplify()
        self.vec2headsq = Vec2AttnHeads(split_query,self.heads)
        self.vec2headsk=Vec2AttnHeads(split_key,self.heads)
        self.vec2headsv=Vec2AttnHeads(split_value,self.heads)


        # self.heads2vecq = AttnHeads2Vec(multiheadsplit(self.irreps_query).simplify())
        # self.heads2veck = AttnHeads2Vec(multiheadsplit(self.irreps_key).simplify())
        self.heads2vecv = AttnHeads2Vec(split_value)

        self.lin = o3.FullyConnectedTensorProduct(self.irreps_output,self.node_attr,self.irreps_output)
        self.sc = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.node_attr, self.irreps_output
        )
    def forward(self, node_attr,node_input,  edge_src, edge_dst, edge_attr, edge_scalars,edge_length,topo_scalar,topo_bias,fpit) -> torch.Tensor:
        edge_length_embedded = edge_scalars
        # edge_length_embedded=self.dropout00(edge_length_embedded)
        edge_sh = edge_attr
        edge_weight_cutoff = edge_length
        # fpit = find_positions_in_tensor_fast(edge_dst)
        # node_input=self.dropout0(node_input)
        # print(node_input.shape)

        num_nodes = node_input.shape[0]

        node_input_sc = self.sc(node_input, node_attr)

        # q = self.h_q0(node_input,node_attr)
        q = self.h_q(node_input,node_attr)

        combined_edge_input = torch.cat([edge_length_embedded, topo_scalar], dim=-1)
        gate_logits = self.expert_gate(topo_scalar)
        expert_weights = torch.softmax(gate_logits, dim=-1)
        cutoff_scale = 1.0 + 0.4 * torch.tanh(self.cutoff_gate(topo_scalar)).squeeze(-1)
        edge_weight_cutoff = edge_weight_cutoff * cutoff_scale
        angle_scale = 1.0 + 0.45 * torch.tanh(self.angle_gate(topo_scalar))
        edge_sh = edge_sh * angle_scale

        weight0 = combined_edge_input.new_zeros(combined_edge_input.size(0), self.tp_k.tp.weight_numel)
        weight1 = combined_edge_input.new_zeros(combined_edge_input.size(0), self.tp_v.tp.weight_numel)
        for idx in range(self.num_relation_experts):
            gate_w = expert_weights[:, idx:idx + 1]
            weight0 = weight0 + gate_w * self.fc_k_experts[idx](combined_edge_input)
            weight1 = weight1 + gate_w * self.fc_v_experts[idx](combined_edge_input)

        k = self.tp_k(node_input[edge_src], edge_sh, weight0,node_attr[edge_src])


        v = self.tp_v(node_input[edge_src], edge_sh, weight1,node_attr[edge_src])


        q = self.vec2headsq(q)
        k = self.vec2headsk(k)
        v = self.vec2headsv(v)

        q_heads = q[edge_dst]
        k_heads = k
        v_heads = v

        head_dim = k_heads.size(-1)
        if head_dim == 0:
            raise RuntimeError('Attention head dimension is zero, cannot compute attention weights.')

        attn_scores = (q_heads * k_heads).sum(dim=-1) / head_dim ** 0.5
        if topo_bias.dim() == 1:
            topo_bias = topo_bias.unsqueeze(-1)
        attn_scores = attn_scores + topo_bias

        max_per_node, _ = torch_scatter.scatter_max(
            attn_scores, edge_dst, dim=0, dim_size=num_nodes
        )
        attn_scores = attn_scores - max_per_node[edge_dst]

        cutoff = edge_weight_cutoff.unsqueeze(-1) if edge_weight_cutoff.dim() == 1 else edge_weight_cutoff
        attn_weights = torch.exp(attn_scores) * cutoff

        denom = scatter(attn_weights, edge_dst, dim=0, dim_size=num_nodes, reduce='sum')
        denom = denom[edge_dst] + 1e-12
        alpha = attn_weights / denom

        messages = alpha.unsqueeze(-1) * v_heads
        sca = scatter(messages, edge_dst, dim=0, dim_size=num_nodes)
        sca=self.heads2vecv(sca)
        sca_conv_out=self.lin(sca,node_attr)
        sca=sca_conv_out+node_input_sc
        # sca=self.norm(sca)

        return sca

class EquivariantAttention(torch.nn.Module):
    def __init__(
        self,
            node_attr,
        irreps_node_input,
            irreps_query,
            irreps_key,
        irreps_node_hidden,
        irreps_node_output,
        irreps_edge_attr,
        layers,
        fc_neurons,

    ) -> None:
        super().__init__()

        self.attr=o3.Irreps(node_attr)
        self.irreps_node_input = o3.Irreps(irreps_node_input)

        self.irreps_query=o3.Irreps(irreps_query)
        self.irreps_key=o3.Irreps(irreps_key)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        # Topological prior modules
        self.edge_label_embed = torch.nn.Embedding(27, EDGE_LABEL_EMBED_DIM)
        self.sg_embed = torch.nn.Embedding(231, SPACE_GROUP_EMBED_DIM)
        self.topo_scalar_norm = torch.nn.LayerNorm(TOPO_FEATURE_DIM)
        self.topo_bias = torch.nn.Sequential(
            Linear(TOPO_FEATURE_DIM, 32),
            torch.nn.SiLU(),
            Linear(32, 1)
        )
        self.topo_bias_log_scale = torch.nn.Parameter(torch.log(torch.tensor(0.35)))



        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: ShiftedSoftPlus(),
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()


        # self.layer.append(self.embed)
        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, ir)
                ]
            ).simplify()

            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, ir)
                ]
            )
            # self.irreps_query1 = o3.Irreps(
            #     [(mul, ir) for mul, ir in o3.Irreps(self.irreps_query) if tp_path_exists(self.irreps_node_input, "0e", ir)])

            ir = "0e" if tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )

            conv = Attention(self.attr,
                self.irreps_node_input,  self.irreps_query,self.irreps_key, gate.irreps_in, fc_neurons
            )
            self.irreps_node_input = gate.irreps_out
            self.norm=EquivariantLayerNormFast(self.irreps_node_input)
            self.layers.append(ComposeWithTopo(conv, gate, self.norm))
        self.layers.append(Attention(self.attr,
                self.irreps_node_input, self.irreps_query, self.irreps_key, self.irreps_node_output, fc_neurons
            )
        )
        num_film_layers = len(self.layers)
        film_modules = []
        for _ in range(num_film_layers):
            film_modules.append(
                torch.nn.Sequential(
                    Linear(TOPO_FEATURE_DIM, 64),
                    torch.nn.SiLU(),
                    Linear(64, 2)
                )
            )
        self.film_mlps = torch.nn.ModuleList(film_modules)
        self.film_gamma_log_scale = torch.nn.Parameter(torch.full((num_film_layers,), math.log(0.5)))  # Increase from 0.35 to 0.5
        self.film_beta_log_scale = torch.nn.Parameter(torch.full((num_film_layers,), math.log(0.35)))

        # Restrict FiLM additive term to even-parity scalar (0e) channels.
        # Multiplicative gamma is a scalar and therefore equivariant on all irreps;
        # additive beta must not be added to non-scalar channels.
        for i, lay in enumerate(self.layers):
            target_irreps = lay.gate.irreps_out if isinstance(lay, ComposeWithTopo) else self.irreps_node_output
            self.register_buffer(f'film_beta_mask_{i}', _scalar_channel_mask(target_irreps))

    def forward(self,node_attr, node_features,  edge_src, edge_dst, edge_attr, edge_scalars,edge_length, edge_label_id, space_group_number, edge_graph) -> torch.Tensor:
        fpit = None  # No cached positions required; placeholder for legacy signature

        # Generate topological priors
        edge_label_emb = self.edge_label_embed(edge_label_id)  # (E, 16)

        sg_numbers = space_group_number.reshape(-1)
        if sg_numbers.device != edge_graph.device:
            edge_graph = edge_graph.to(sg_numbers.device)
        edge_graph = edge_graph.to(torch.long)
        sg_per_edge = sg_numbers[edge_graph]
        sg_emb = self.sg_embed(sg_per_edge.to(edge_label_id.device))  # (E, 8)
        topo_scalar = torch.cat([edge_label_emb, sg_emb], dim=-1)  # (E, 24)
        topo_scalar = self.topo_scalar_norm(topo_scalar)
        topo_bias_val = self.topo_bias(topo_scalar).squeeze(-1)  # (E,)
        topo_bias_val = torch.exp(self.topo_bias_log_scale) * topo_bias_val

        # Generate FiLM parameters per layer
        topo_scalar_dst = scatter(topo_scalar, edge_dst, dim=0, reduce='mean')  # (N, 24)
        film_gamma_list = []
        film_beta_list = []
        for idx, film_mlp in enumerate(self.film_mlps):
            film_params = film_mlp(topo_scalar_dst)  # (N, 2)
            gamma_scale = torch.exp(self.film_gamma_log_scale[idx])
            beta_scale = torch.exp(self.film_beta_log_scale[idx])
            gamma_scalar = torch.exp(gamma_scale * torch.tanh(film_params[:, 0:1]))
            beta_scalar = beta_scale * torch.tanh(film_params[:, 1:2])
            film_gamma_list.append(gamma_scalar)
            film_beta_list.append(beta_scalar)

        # Cache modulation parameters and topology bias statistics for reading at epoch end
        with torch.no_grad():
            if film_gamma_list:
                gamma_stats_tensor = torch.cat([g.detach().flatten() for g in film_gamma_list])
                beta_stats_tensor = torch.cat([b.detach().flatten() for b in film_beta_list])
                self.last_film_gamma_stats = tensor_stats(gamma_stats_tensor)
                self.last_film_beta_stats  = tensor_stats(beta_stats_tensor)
                # Save beta parameter list for L1 regularization
                self.last_film_beta_list = [b.detach().clone() for b in film_beta_list]
            else:
                self.last_film_gamma_stats = {}
                self.last_film_beta_stats  = {}
                self.last_film_beta_list = []
            self.last_topo_bias_stats  = tensor_stats(topo_bias_val)

        for idx, lay in enumerate(self.layers):
            node_features = lay(
                node_attr,
                node_features,
                edge_src,
                edge_dst,
                edge_attr,
                edge_scalars,
                edge_length,
                topo_scalar,
                topo_bias_val,
                fpit,
            )
            if idx < len(film_gamma_list):
                film_gamma_scalar = film_gamma_list[idx]
                film_beta_scalar = film_beta_list[idx]
                expand_shape = [node_features.size(0)] + [1] * (node_features.dim() - 1)
                film_gamma = film_gamma_scalar.reshape(expand_shape).expand_as(node_features)
                film_beta = film_beta_scalar.reshape(expand_shape).expand_as(node_features)
                # Mask beta so it is added only to 0e scalar channels
                film_beta = film_beta * getattr(self, f'film_beta_mask_{idx}').to(node_features.device).unsqueeze(0)
            else:
                film_gamma = torch.ones_like(node_features)
                film_beta = torch.zeros_like(node_features)
            node_features = film_gamma * node_features + film_beta
        return node_features


@compile_mode('script')
class Network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
            embedding_dim,
            irreps_query,
            irreps_key,
        irreps_out,
            formula,
        max_radius,
            num_nodes,
        mul=32,
        layers=2,
            number_of_basis=16,
        lmax=lmax,
        pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.pool_exponent_logit = torch.nn.Parameter(torch.tensor(0.0))
        self.formula = formula

        self.irreps_in=irreps_in
        self.embeding_dim=embedding_dim

        irreps_node_hidden = o3.Irreps([(int(mul/2**(l)), (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_node_hidden = irreps_node_hidden.simplify()
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key

        self.irreps_sh=o3.Irreps.spherical_harmonics(lmax)
        # self.dropout0 = nn.Dropout(irreps="{}x0e".format(self.embeding_dim),p=0.2)

        self.lin=o3.Linear(self.irreps_in,"{}x0e".format(self.embeding_dim))
        self.GAT=EquivariantAttention(
            node_attr=self.irreps_in,
        irreps_node_input="{}x0e".format(self.embeding_dim),
            irreps_query=irreps_query,
            irreps_key=irreps_key,
        irreps_node_hidden=self.irreps_node_hidden,
        irreps_node_output=irreps_out,
        irreps_edge_attr=self.irreps_sh,
        layers=layers,
        fc_neurons=[self.number_of_basis,128],
        )

        self.irreps_in = self.GAT.irreps_node_input
        self.irreps_out = self.GAT.irreps_node_output

        self.TI0=o3.Linear(self.irreps_out,self.irreps_out)
        self.TI1 = o3.Linear(self.irreps_out, self.irreps_out)
        self.TI = TensorIrreps(self.formula, self.irreps_out)
        # Mask graph-level FiLM beta to 0e scalar channels for equivariance
        self.register_buffer('graph_film_beta_mask', _scalar_channel_mask(self.irreps_out))

        self.dropout1 = nn.Dropout(irreps=self.irreps_out, p=0.25)
        self.graph_film_mlp = torch.nn.Sequential(
            Linear(self.irreps_out.dim, 128),
            torch.nn.SiLU(),
            Linear(128, 2)
        )
        self.graph_film_log_scale = torch.nn.Parameter(torch.tensor(math.log(0.35)))
        self.last_graph_film_gamma_stats = None
        self.last_graph_film_beta_stats = None

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if batch.dtype != torch.long:
            batch = batch.to(torch.long)

        edge_src = data['edge_index'][0]  # Edge source
        edge_dst = data['edge_index'][1]  # Edge destination

        # No need to check indices bounds - new SLICES file ensures 100% consistency
        edge_graph = batch[edge_src]

        # We need to compute this in the computation graph to backprop to positions
        # We are computing the relative distances + unit cell shifts from periodic boundaries
        edge_vec = (data['pos'][edge_dst]
                    - data['pos'][edge_src]
                    + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice'][edge_graph]))

        return batch, data['x'], edge_src, edge_dst, edge_vec, edge_graph

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, node_inputs, edge_src, edge_dst, edge_vec, edge_graph = self.preprocess(data)
        # Don't delete data yet, we need it for edge_label_id and space_group_number
        node_attr=node_inputs
        edge_attr = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization="component")

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis ** 0.5)

        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))
        node_features=self.lin(node_inputs)
        # node_features=self.dropout0(node_features)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        edge_graph = edge_graph.to(node_features.device, dtype=torch.long)
        edge_label_id = data.get('edge_label_id')
        if edge_label_id is None:
            edge_label_id = torch.zeros(edge_src.shape[0], dtype=torch.long, device=node_features.device)
        else:
            edge_label_id = edge_label_id.to(node_features.device, dtype=torch.long)
        raw_space_group = data.get('space_group_number')
        if raw_space_group is None:
            space_group_number = torch.zeros(num_graphs, dtype=torch.long, device=node_features.device)
        else:
            if not torch.is_tensor(raw_space_group):
                raw_space_group = torch.tensor(raw_space_group, dtype=torch.long, device=node_features.device)
            else:
                raw_space_group = raw_space_group.to(node_features.device, dtype=torch.long)
            if raw_space_group.dim() == 0 or raw_space_group.numel() == 1:
                space_group_number = raw_space_group.reshape(1).expand(num_graphs)
            elif raw_space_group.numel() < num_graphs:
                pad = num_graphs - raw_space_group.numel()
                padding = raw_space_group.new_zeros(pad)
                space_group_number = torch.cat([raw_space_group.reshape(-1), padding], dim=0)
            else:
                space_group_number = raw_space_group.reshape(-1)
        node_outputs = self.GAT(
            node_attr,
            node_features,
            edge_src,
            edge_dst,
            edge_attr,
            edge_length_embedding,
            edge_weight_cutoff,
            edge_label_id,
            space_group_number,
            edge_graph,
        )
        # node_outputs=self.dropout1(node_outputs)
        if self.pool_nodes:
            batch_index = batch.to(node_outputs.device, dtype=torch.long)
            dim_size = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
            graph_outputs = scatter(node_outputs, batch_index, dim=0, reduce="sum", dim_size=dim_size)
            ones = torch.ones(batch_index.size(0), device=node_outputs.device, dtype=node_outputs.dtype)
            node_counts = scatter(ones, batch_index, dim=0, reduce="sum", dim_size=dim_size).unsqueeze(-1)
            exponent = torch.sigmoid(self.pool_exponent_logit)
            norm = node_counts.pow(exponent).clamp_min(1.0)
            node_outputs = graph_outputs / norm
        else:
            pass
        graph_params = self.graph_film_mlp(node_outputs)
        graph_scale = torch.exp(self.graph_film_log_scale)
        graph_gamma = torch.exp(graph_scale * torch.tanh(graph_params[:, 0:1]))
        graph_beta = graph_scale * torch.tanh(graph_params[:, 1:2])
        # Apply beta only to 0e scalar channels; gamma scales all channels
        node_outputs = graph_gamma * node_outputs + graph_beta * self.graph_film_beta_mask.to(node_outputs.device).unsqueeze(0)
        with torch.no_grad():
            self.last_graph_film_gamma_stats = tensor_stats(graph_gamma)
            self.last_graph_film_beta_stats = tensor_stats(graph_beta)
            # Save GraphFiLM beta parameters for L1 regularization
            self.last_graph_film_beta = graph_beta.detach().clone()
        node_outputs1=self.TI0(node_outputs)

        node_outputs2=self.TI1(node_outputs1)
        node_outputs=node_outputs2+node_outputs
        node_outputs=self.TI(node_outputs)

        # Point group hard projection: Force tensor to satisfy crystal symmetry constraints
        node_outputs_flat = node_outputs.reshape(node_outputs.size(0), 27)  # [B, 27]
        # Use each sample's own space group number (need to get from data in batch)
        if hasattr(data, 'space_group_number'):
            sg_numbers = data.space_group_number
            # Ensure sg_numbers has correct shape
            if sg_numbers.numel() == 1:
                sg_numbers = sg_numbers.repeat(node_outputs.size(0))
            else:
                sg_numbers = sg_numbers
        else:
            sg_numbers = torch.zeros(node_outputs.size(0), dtype=torch.long, device=node_outputs.device)
        projected_outputs = apply_pointgroup_projection(node_outputs_flat, sg_numbers, node_outputs.device)  # Apply symmetry projection
        node_outputs = projected_outputs.reshape(node_outputs.size(0), 3, 3, 3)  # Restore [B, 3, 3, 3]

        if torch.isnan(node_outputs).any():
            print('nan after TI')
        # node_outputs=self.dropout1(node_outputs)
        return node_outputs


net = Network(
    irreps_in="{}x0e".format(dim),
    embedding_dim=64,
    irreps_query="32x0e+32x0o+16x1e+16x1o+12x2e+12x2o+8x3e+8x3o+4x4e+4x4o",
    irreps_key="32x0e+32x0o+16x1e+16x1o+12x2e+12x2o+8x3e+8x3o+4x4e+4x4o",
    irreps_out="16x1o+8x2o+4x3o+4x4o",
    formula="ijk=ikj",
    max_radius=max_radius, # Cutoff radius for convolution
    num_nodes=num_nodes,
    pool_nodes=True,  # We pool nodes to predict properties.
)

net=net.to(device)

# Print model parameter count
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total model parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

film_related_params = []
film_related_params.extend(list(net.GAT.film_mlps.parameters()))
film_related_params.extend(list(net.GAT.topo_bias.parameters()))
film_related_params.append(net.GAT.topo_bias_log_scale)
film_related_params.append(net.GAT.film_gamma_log_scale)
film_related_params.append(net.GAT.film_beta_log_scale)
film_related_param_ids = {id(p) for p in film_related_params}
base_params = [p for p in net.parameters() if p.requires_grad and id(p) not in film_related_param_ids]
base_lr = 2e-4
film_lr = base_lr * 2.0
optim=torch.optim.AdamW(
    [
        {"params": base_params, "weight_decay": 1e-4, "lr": base_lr},
        {"params": film_related_params, "weight_decay": 5e-5, "lr": film_lr}
    ],
    lr=base_lr
)
loss=torch.nn.MSELoss()
loss=loss.to(device)
# L2 norm loss parameters
L2_NORM_ALPHA = 1
scaler=GradScaler(enabled=False)
steps_per_epoch = max(len(train_dataloader), 1)
total_steps = max(epochs * steps_per_epoch, 1)
warmup_steps = min(4 * steps_per_epoch, total_steps)  # Extend warmup from 2→4 epochs
def lr_lambda(step_idx: int) -> float:
    if total_steps <= 1:
        return 1.0
    if step_idx < warmup_steps:
        return float(step_idx + 1) / max(1, warmup_steps)
    progress = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    min_lr_ratio = 3e-2
    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

# Add ReduceLROnPlateau scheduler
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim,
    mode='min',
    factor=0.8,
    patience=3,
    min_lr=1e-6
)

# Create timestamped result folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.join("result", f"run_{timestamp}")
os.makedirs(result_dir, exist_ok=True)

print(f"Results will be saved to: {result_dir}")

# Best model saver and update ratio tracker
best_val = float("inf")
best_path = os.path.join(result_dir, "best_model.pt")

def save_checkpoint(path, model, optim, scheduler, epoch, extra=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "plateau_sched_state": plateau_scheduler.state_dict() if 'plateau_scheduler' in globals() else None,
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)

# Used to calculate "parameter update ratio"
def param_update_ratio(model, prev_params):
    num, den = 0.0, 0.0
    with torch.no_grad():
        for (name, p), prev in zip(model.named_parameters(), prev_params):
            if p.requires_grad and p.data.numel() > 0:
                dw = (p.data - prev).abs().mean().item()
                w  = p.data.abs().mean().item() + 1e-12
                num += dw
                den += w
    return num / max(den, 1e-12)

def snapshot_params(model):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]

# Loss tracking object
class LossTracking:
    def __init__(self):
        self.tensor_losses = []
        self.l2_norm_losses = []
        self.mse_losses = []
        self.reset()

    def reset(self):
        self.tensor_losses.clear()
        self.l2_norm_losses.clear()
        self.mse_losses.clear()

train_loss_tracking = LossTracking()

global_step = 0
for epoch in range(epochs):
    with epoch_timer():
        net.train()
        trainloss = 0.0
        # Reset loss tracking for this epoch
        train_loss_tracking.reset()
        train_mae  = 0.0
        train_rmse = 0.0
        grad_global_norm_acc = 0.0
        grad_global_norm_cnt = 0
        param_norm_acc = 0.0
        param_norm_cnt = 0
        update_ratio_acc = 0.0
        update_ratio_cnt = 0
        std_reg_acc = 0.0
        std_lambda_acc = 0.0
        beta_l1_acc = 0.0

        prev_params = snapshot_params(net)  # For update ratio

        for batch in tqdm(train_dataloader):
            batch = batch.to(device, non_blocking=True)
            optim.zero_grad()
            with autocast(enabled=False):
                output = net(batch)

                # Simple reshape: ensure output and target are both [batch_size, 27]
                output_flat = output.view(output.size(0), -1)
                target_flat = batch.energy.view(batch.energy.size(0), -1)

                # Use robust loss for tensors instead of MSE
                tensor_loss = robust_loss_fn(output, batch.energy)

                # L2 norm loss: MSE(pred, target) + alpha * MSE(norm(pred), norm(target))
                output_norms = torch.norm(output_flat, p=2, dim=1, keepdim=True)
                target_norms = torch.norm(target_flat, p=2, dim=1, keepdim=True)
                l2_norm_loss = loss(output_norms, target_norms)
                combined_loss = tensor_loss + L2_NORM_ALPHA * l2_norm_loss
                

                std_reg = output.new_zeros(())
                std_lambda = 0.0
                if STD_ALIGNMENT_LAMBDA > 0.0:
                    final_std_lambda = STD_ALIGNMENT_LAMBDA
                    if total_steps > 1:
                        if global_step < warmup_steps:
                            std_lambda = final_std_lambda * (global_step + 1) / max(1, warmup_steps)
                        else:
                            steps_after_warm = global_step - warmup_steps
                            ramp_extra = int(2 * steps_per_epoch)
                            if steps_after_warm < ramp_extra:
                                std_lambda = final_std_lambda * (steps_after_warm + 1) / max(1, ramp_extra)
                            else:
                                std_lambda = final_std_lambda
                    else:
                        std_lambda = final_std_lambda
                if STD_ALIGNMENT_LAMBDA > 0.0:
                    
                    pred_flat = output.reshape(output.size(0), -1)
                    tgt_flat = batch.energy.reshape(batch.energy.size(0), -1)
                    tgt_std_g = tgt_flat.std(dim=1, unbiased=False).clamp_min(1e-6)
                    pred_std_g = pred_flat.std(dim=1, unbiased=False).clamp_min(1e-6)
                    ratio_g = torch.clamp(pred_std_g / tgt_std_g, 1e-2, 1e2)
                    std_reg = (torch.log(ratio_g) ** 2).mean()
            
            beta_l1_loss = 0.0
            try:
                # Collect FiLM beta parameters
                if hasattr(net.GAT, 'last_film_beta_list') and net.GAT.last_film_beta_list:
                    for beta_tensor in net.GAT.last_film_beta_list:
                        beta_l1_loss += torch.abs(beta_tensor).mean()

                # GraphFiLM beta
                if hasattr(net, 'last_graph_film_beta') and net.last_graph_film_beta is not None:
                    beta_l1_loss += torch.abs(net.last_graph_film_beta).mean()

            except Exception as e:
                beta_l1_loss = 0.0

            l = combined_loss + std_lambda * std_reg + 1e-5 * beta_l1_loss

            output_det = output.detach()
            target_det = batch.energy.detach()
            # Target values are already in original scale, no need to denormalize
            std_reg_acc += float(std_reg.detach())
            std_lambda_acc += float(std_lambda)
            beta_l1_acc += float(beta_l1_loss.detach())

            scaler.scale(l).backward()
            scaler.unscale_(optim)

            # Gradient global norm (before clipping)
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            grad_global_norm_acc += float(total_norm)
            grad_global_norm_cnt += 1

            # Parameter norm (sample statistics)
            with torch.no_grad():
                pn = 0.0
                cnt = 0
                for p in net.parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        pn += p.data.norm(2).item()
                        cnt += 1
                if cnt > 0:
                    param_norm_acc += pn / cnt
                    param_norm_cnt += 1

            scaler.step(optim)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            trainloss += l.item()
            with torch.no_grad():
                diff = output_det - target_det
                train_mae  += diff.abs().mean().item()
                train_rmse += torch.sqrt((diff ** 2).mean()).item()
                # Log individual loss components for monitoring
                tensor_loss_val = tensor_loss.item()
                l2_norm_loss_val = l2_norm_loss.item()
                mse_loss_val = loss(output_flat, target_flat).item()
                # Store for printing at epoch end
                train_loss_tracking.tensor_losses.append(tensor_loss_val)
                train_loss_tracking.l2_norm_losses.append(l2_norm_loss_val)
                train_loss_tracking.mse_losses.append(mse_loss_val)

        # Update ratio (based on full round parameter comparison)
        update_ratio = param_update_ratio(net, prev_params)
        update_ratio_acc += update_ratio
        update_ratio_cnt += 1

        # ========== Validation ==========
        net.eval()
        validloss = 0.0
        valid_mae  = 0.0
        valid_rmse = 0.0

        # For collecting validation set statistics
        all_valid_targets = []
        all_valid_outputs = []

        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                batch = batch.to(device, non_blocking=True)
                with autocast(enabled=False):
                    output = net(batch)

                    # Use robust loss for validation instead of MSE
                    l = robust_loss_fn(output, batch.energy).item()
                validloss += l
                output_det = output.detach()
                target_det = batch.energy.detach()
                # Target values are already in original scale, no need to denormalize
                diff = output_det - target_det
                valid_mae  += diff.abs().mean().item()
                valid_rmse += torch.sqrt((diff ** 2).mean()).item()

                # Collect targets and outputs for statistics
                all_valid_targets.append(target_det.cpu().flatten())
                all_valid_outputs.append(output_det.cpu().flatten())

        # ======= Summary and Print =======
        n_train_batches = max(len(train_dataloader), 1)
        n_valid_batches = max(len(valid_dataloader), 1)
        train_std_pen = std_reg_acc / n_train_batches
        avg_std_lambda = std_lambda_acc / n_train_batches
        avg_beta_l1 = beta_l1_acc / n_train_batches

        val_out_stats = None
        val_tgt_stats = None
        std_ratio = None

        # Calculate validation set statistics
        if all_valid_targets:
            all_targets_tensor = torch.cat(all_valid_targets)
            all_outputs_tensor = torch.cat(all_valid_outputs)

            epoch_target_stats = tensor_stats(all_targets_tensor)
            epoch_output_stats = tensor_stats(all_outputs_tensor)

            val_out_stats = epoch_output_stats
            val_tgt_stats = epoch_target_stats
            tgt_std = val_tgt_stats.get("std", 0.0)
            if tgt_std != 0.0:
                std_ratio = val_out_stats.get("std", 0.0) / max(abs(tgt_std), 1e-12)

        grad_gn = grad_global_norm_acc / max(grad_global_norm_cnt, 1)
        param_n = param_norm_acc / max(param_norm_cnt, 1)
        upd_rat = update_ratio_acc / max(update_ratio_cnt, 1)
        lr_now  = optim.param_groups[0]["lr"]
        scalerv = safe_item(scaler.get_scale())

        # Calculate average loss components for this epoch
        avg_tensor_loss = sum(train_loss_tracking.tensor_losses) / len(train_loss_tracking.tensor_losses) if train_loss_tracking.tensor_losses else 0.0
        avg_l2_norm_loss = sum(train_loss_tracking.l2_norm_losses) / len(train_loss_tracking.l2_norm_losses) if train_loss_tracking.l2_norm_losses else 0.0
        avg_mse_loss = sum(train_loss_tracking.mse_losses) / len(train_loss_tracking.mse_losses) if train_loss_tracking.mse_losses else 0.0

        # Detailed training monitoring output
        print(
            f"[Epoch {epoch}] "
            f"lr={lr_now:.6f} "
            f"train_loss={trainloss/n_train_batches:.6f} "
            f"tensor_loss={avg_tensor_loss:.6f} "
            f"l2_norm_loss={avg_l2_norm_loss:.6f} "
            f"mse_loss={avg_mse_loss:.6f} "
            f"train_MAE={train_mae/n_train_batches:.6f} "
            f"train_RMSE={train_rmse/n_train_batches:.6f} "
            f"train_std_pen={train_std_pen:.6f} "
            f"std_lambda={avg_std_lambda:.6f} "
            f"beta_l1={avg_beta_l1:.6f} "
            f"valid_loss={validloss/n_valid_batches:.6f} "
            f"valid_MAE={valid_mae/n_valid_batches:.6f} "
            f"valid_RMSE={valid_rmse/n_valid_batches:.6f}"
        )
        # Keep validation statistics output
        if val_out_stats is not None and val_tgt_stats is not None:
            print(f"[Valid Output] {val_out_stats}")
            print(f"[Valid Target] {val_tgt_stats}")
            if std_ratio is not None:
                print(f"[Valid Ratio] std_ratio={std_ratio:.4f}")
        print(f"[Grad] global_norm(avg)={grad_gn:.4f}  [Param] norm(avg)={param_n:.4f}  [UpdateRatio]={upd_rat:.6e}  [Scaler]={scalerv}")

        # Read FiLM / topo_bias statistics
        try:
            film_g = getattr(net.GAT, "last_film_gamma_stats", None)
            film_b = getattr(net.GAT, "last_film_beta_stats", None)
            topo_b = getattr(net.GAT, "last_topo_bias_stats", None)
            graph_g = getattr(net, "last_graph_film_gamma_stats", None)
            graph_b = getattr(net, "last_graph_film_beta_stats", None)
            if film_g and film_b and topo_b:
                print(f"[FiLM] gamma={film_g}  beta={film_b}")
                print(f"[TopoBias] {topo_b}")
            if graph_g and graph_b:
                print(f"[GraphFiLM] gamma={graph_g}  beta={graph_b}")
        except Exception as e:
            print(f"[Warn] cannot fetch FiLM/Topo stats: {e}")

        # Save best model
        cur_val = valid_rmse / n_valid_batches
        if cur_val < best_val:
            best_val = cur_val
            save_checkpoint(best_path, net, optim, scheduler, epoch, extra={"best_val_rmse": best_val})
            print(f"[Checkpoint] best model updated: {best_path} (val_RMSE={best_val:.6f})")

        # Use ReduceLROnPlateau scheduler
        plateau_scheduler.step(cur_val)

        # Periodically save checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(result_dir, f"ckpt_epoch_{epoch+1}.pt")
            save_checkpoint(checkpoint_path, net, optim, scheduler, epoch)
            print(f"[Checkpoint] periodic save at epoch {epoch+1}: {checkpoint_path}")
