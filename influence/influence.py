import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from tqdm.auto import tqdm
from typing import List, Tuple

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

# -----------------------------------------------------------------------------
# Exact *k*-hop neighbourhood shell
# -----------------------------------------------------------------------------

def k_hop_subsets_rough(
    node_idx: int,
    num_hops: int,
    edge_index: Tensor,
    num_nodes: int,
) -> List[Tensor]:
    """Return *rough* (possibly overlapping) *k*-hop node subsets.

    This is a thin wrapper around
    :pyfunc:`torch_geometric.utils.k_hop_subgraph` that *additionally* returns
    **all** intermediate hop subsets rather than the full union only.

    Parameters
    ----------
    node_idx: int | Sequence[int] | Tensor
        Index or indices of the central node(s).
    num_hops: int
        Number of hops *k*.
    edge_index: Tensor
        Edge index in COO format with shape :math:`[2, \text{num_edges}]`.
    num_nodes: int
        Total number of nodes in the graph. Required to allocate the masks.

    Returns
    -------
    List[Tensor]
        A list ``[H₀, H₁, …, H_k]`` where ``H₀`` contains the seed node(s) and
        ``H_i`` (for *i*>0) contains **all** nodes that are exactly *i* hops
        away in the *expanded* neighbourhood (i.e. overlaps are *not*
        removed).
    """
    col, row = edge_index  # (2, E)

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = torch.tensor([node_idx], device=row.device)

    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.zero_()
        node_mask[subsets[-1]] = True  # mark current frontier
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    return subsets


def k_hop_subsets_exact(
    node_idx: int,
    num_hops: int,
    edge_index: Tensor,
    num_nodes: int,
    device: torch.device | str,
) -> List[Tensor]:
    """Return **disjoint** *k*-hop subsets.

    This function refines :pyfunc:`k_hop_subsets_rough` by removing nodes that
    have already appeared in previous hops, ensuring that each subset contains
    nodes *exactly* *i* hops away from the seed.
    """
    rough_subsets = k_hop_subsets_rough(node_idx, num_hops, edge_index, num_nodes)

    exact_subsets: List[List[int]] = [rough_subsets[0].tolist()]
    visited: set[int] = set(exact_subsets[0])

    for hop_subset in rough_subsets[1:]:
        fresh = set(hop_subset.tolist()) - visited
        visited |= fresh
        exact_subsets.append(list(fresh))

    return [torch.tensor(s, device=device, dtype=edge_index.dtype) for s in exact_subsets]


# -----------------------------------------------------------------------------
# Jacobian‑based influence metrics
# -----------------------------------------------------------------------------

def jacobian_l1(
    model: torch.nn.Module,
    data: Data,
    max_hops: int,
    node_idx: int,
    device: torch.device | str,
    *,
    vectorize: bool = True,
) -> Tensor:
    """Compute the **L1 norm** of the Jacobian for a given node.

    The Jacobian is evaluated w.r.t. the node features of the *k*-hop induced
    sub‑graph centred at ``node_idx``. The result is *folded back* onto the
    **original** node index space so that the returned tensor has length
    ``data.num_nodes``, where the influence score will be zero for nodes outside
    the *k*-hop subgraph.

    Notes
    -----
    *   The function assumes that the model *and* ``data.x`` share the same
        floating‑point precision (e.g. both ``float32`` or both ``float16``).

    """
    # Build the induced *k*-hop sub‑graph (with node re‑labelling).
    k_hop_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, max_hops, data.edge_index, relabel_nodes=True
    )
    # get the location of the *center* node inside the sub‑graph
    root_pos = int(mapping[0])

    # Move tensors & model to the correct device
    device = torch.device(device)
    sub_x = data.x[k_hop_nodes].to(device)
    sub_edge_index = sub_edge_index.to(device)
    model = model.to(device)

    # Jacobian evaluation
    def _forward(x: Tensor) -> Tensor:  # noqa: D401 (simple function)
        return model(x, sub_edge_index)[root_pos]
    jac = jacobian(_forward, sub_x, vectorize=vectorize)
    influence_sub = jac.abs().sum(dim=(0, 2))  # Sum of L1 norm
    # Scatter the influence scores back to the *global* node space
    influence_full = torch.zeros(
        data.num_nodes, dtype=influence_sub.dtype, device=device
    )
    influence_full[k_hop_nodes] = influence_sub

    return influence_full


def jacobian_l1_agg_per_hop(
    model: torch.nn.Module,
    data: Data,
    max_hops: int,
    node_idx: int, 
    device: torch.device | str,
    vectorize: bool = True,
) -> Tensor:
    """Aggregate Jacobian L1 norms **per hop** for node_idx.

    Returns a vector ``[I_0, I_1, …, I_k]`` where ``I_i`` is the *total* influence
    exerted by nodes that are exactly *i* hops away from ``node_idx``.
    """

    influence = jacobian_l1(
        model, data, max_hops, node_idx, device, vectorize=vectorize
    )
    hop_subsets = k_hop_subsets_exact(
        node_idx, max_hops, data.edge_index, data.num_nodes, influence.device
    )
    sigle_node_influence_per_hop = [influence[s].sum() for s in hop_subsets]
    return torch.tensor(sigle_node_influence_per_hop, device=influence.device)


def avg_total_influence(influence_all_nodes, normalize=True):
    """Compute the *influence‑weighted receptive field* ``R``.
    """
    avg_total_influences = torch.mean(influence_all_nodes, axis=0)
    if normalize: # nomalize by hop_0 (jacobian of the center node feature)
        avg_total_influences = avg_total_influences / avg_total_influences[0]
    return avg_total_influences


def influence_weighted_receptive_field(T: torch.Tensor) -> float:
    """Compute the *influence‑weighted receptive field* ``R``.

    Given an influence matrix ``T`` of shape ``[N, k+1]`` (i‑th row contains the
    per‑hop influences of node *i*), the receptive field breadth *R* is defined
    as the expected hop distance when weighting by influence.

    A larger *R* indicates that, on average, influence comes from **farther**
    hops.
    """
    normalised = T / T.sum(axis=1, keepdims=True)
    hops = torch.arange(T.shape[1]).float()  # 0 … k
    breadth = normalised @ hops  # shape (N,)
    return breadth.mean().item()


def total_influence(
    model: torch.nn.Module,
    data: Data,
    max_hops: int,
    num_samples: int | None = None,
    normalize: bool = True,
    average: bool = True,
    device: torch.device | str = 'cpu',
    vectorize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Compute influence aggregates for *multiple* seed nodes.

    Args:
    ----------
    model : torch.nn.Module
        A **PyTorch Geometric** compatible model with signature
        ``model(x: Tensor, edge_index: Tensor) -> Tensor``.
    data : torch_geometric.data.Data
        The underlying data object
    max_hops : int
        Maximum neighbourhood radius *k*.
    num_samples : int | None, default=None
        If given, draw a random subset of nodes. Otherwise use all
        the nodes in the underying graph.
    normalize: bool
        If to normalize the average total influence of each hop by
        the influence of hop 0.
    average: bool
        If to return the raw total influence for all nodes.
    device : torch.device | str
        Target device for the computation.
    vectorize : bool, default=True
        vectorize in jacobian from torch.autograd.functional


    Returns
    -------
    Tensor
        A tensor of shape ``[N, max_hops+1]`` where ``N`` is
        min(``data.num_nodes``, ``num_samples``).
    """
    num_samples = data.num_nodes if num_samples is None else num_samples
    nodes = torch.randperm(data.num_nodes)[:num_samples].tolist()
    influence_all_nodes = [
        jacobian_l1_agg_per_hop(
            model, data, max_hops, n, device, vectorize=vectorize
        )
        for n in tqdm(nodes, desc="Influence")
    ]
    influence_all_nodes = torch.vstack(influence_all_nodes).detach().cpu()
    
    # Average total influence at each hop
    if average:
        avg_influence = avg_total_influence(influence_all_nodes, normalize=normalize)
    else:
        avg_influence = influence_all_nodes
    # influence weighted receptive field
    R = influence_weighted_receptive_field(influence_all_nodes)
    
    return avg_influence, R