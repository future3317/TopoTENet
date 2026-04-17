import torch
import math
from e3nn import o3, nn
from torch.nn import Linear
from e3nn.nn import FullyConnectedNet
from torch.nn import ModuleList
from torch_scatter import scatter, scatter_max
from e3nn.util.jit import compile_mode
from e3nn.nn import Gate
from e3nn.o3 import Irreps
from e3nn.io import CartesianTensor
from torch.nn import functional as F
import torch.nn as nn
from typing import Union, Dict
from torch_geometric.data import Data
import torch_geometric
import torch_scatter

from symmetry import apply_pointgroup_projection

# 常量定义
LABEL_CHAR2INT = {'o': 0, '+': 1, '-': -1}
CHARS = ['-', 'o', '+']
EDGE_LABEL_VOCAB = [''.join(t) for t in (a+b+c for a in CHARS for b in CHARS for c in CHARS)]
EDGE_LABEL_TO_ID = {label: idx for idx, label in enumerate(EDGE_LABEL_VOCAB)}
EDGE_LABEL_EMBED_DIM = 16
SPACE_GROUP_EMBED_DIM = 8
TOPO_FEATURE_DIM = EDGE_LABEL_EMBED_DIM + SPACE_GROUP_EMBED_DIM


def tensor_stats(t):
    t = t.detach()
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
    return t


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
    def __init__(self, node_attr, irreps_node_input, irreps_query, irreps_key, irreps_output, number_of_basis, heads=2, lmax=4):
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
        heads=2,
        lmax=4,
    ) -> None:
        super().__init__()

        self.heads = heads
        self.lmax = lmax

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

            conv = Attention(
                self.attr,
                self.irreps_node_input, self.irreps_query,self.irreps_key, gate.irreps_in, fc_neurons, heads=self.heads, lmax=self.lmax
            )
            self.irreps_node_input = gate.irreps_out
            self.norm=EquivariantLayerNormFast(self.irreps_node_input)
            self.layers.append(ComposeWithTopo(conv, gate, self.norm))
        self.layers.append(
            Attention(
                self.attr,
                self.irreps_node_input, self.irreps_query, self.irreps_key, self.irreps_node_output, fc_neurons, heads=self.heads, lmax=self.lmax
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
        lmax=4,
        pool_nodes=True,
        heads=2,
    ) -> None:
        super().__init__()

        self.heads = heads
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
            heads=self.heads,
            lmax=self.lmax,
        )

        self.irreps_in = self.GAT.irreps_node_input
        self.irreps_out = self.GAT.irreps_node_output

        self.TI0=o3.Linear(self.irreps_out,self.irreps_out)
        self.TI1 = o3.Linear(self.irreps_out, self.irreps_out)
        self.TI = TensorIrreps(self.formula, self.irreps_out)

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
        node_outputs = graph_gamma * node_outputs + graph_beta
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


