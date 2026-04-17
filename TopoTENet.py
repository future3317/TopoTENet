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

from network_classes import *
from utils import *






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









#from equiformer


#from matten


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








for crystal, energy, slices_string, sg_number in zip(struct, dummy_energies, full_slices_strings, space_groups):
    data = datatransform_from_slices(crystal, energy, slices_string, sg_number, fea, device)
    dataset.append(data)



# Apply outlier filtering
print("Starting outlier filtering...")
dataset = filter_outliers_by_quantile(dataset, quantile=0.95)



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










net = Network(
    irreps_in="{}x0e".format(dim),
    embedding_dim=64,
    irreps_query="32x0e+32x0o+16x1e+16x1o+12x2e+12x2o+8x3e+8x3o+4x4e+4x4o",
    irreps_key="32x0e+32x0o+16x1e+16x1o+12x2e+12x2o+8x3e+8x3o+4x4e+4x4o",
    irreps_out="16x1o+8x2o+4x3o+4x4o",
    formula="ijk=ikj",
    max_radius=max_radius,
    num_nodes=num_nodes,
    pool_nodes=True,
    heads=heads,
    lmax=lmax,
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
