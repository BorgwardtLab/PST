import math
from typing import Any, Callable, Iterable, List, Optional

import esm
import requests
import torch
from joblib import Parallel, delayed
from sklearn.neighbors import radius_neighbors_graph
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter
from tqdm import tqdm

AA_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


def to_dense_batch(x, ptr, return_mask=False, fill_value=0.0):
    batch_size = len(ptr) - 1
    num_nodes = torch.diff(ptr)
    max_num_nodes = num_nodes.max().item()
    all_num_nodes = ptr[-1].item()

    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1

    device = x[0].device if isinstance(x, (list, tuple)) else x.device

    batch = torch.arange(batch_size, dtype=torch.long, device=device)
    batch = batch.repeat_interleave(num_nodes)
    idx = torch.arange(all_num_nodes, dtype=torch.long, device=device)
    idx = (idx - ptr[batch]) + batch * max_num_nodes

    size = [batch_size * max_num_nodes]
    if isinstance(x, (list, tuple)):
        new_x = []
        for j in range(len(x)):
            xj = x[j].new_full(size + list(x[j].size())[1:], fill_value)
            xj[idx] = x[j][:all_num_nodes]
            xj = xj.view([batch_size, max_num_nodes] + list(x[j].size())[1:])
            if cls_tokens:
                xj[:, -1, ...] = x[j][all_num_nodes:]
            new_x.append(xj)
    else:
        new_x = x.new_full(size + list(x.size())[1:], fill_value)
        new_x[idx] = x[:all_num_nodes]
        new_x = new_x.view([batch_size, max_num_nodes] + list(x.size())[1:])
        if cls_tokens:
            new_x[:, -1, ...] = x[all_num_nodes:]

    if return_mask:
        mask = torch.ones(batch_size * max_num_nodes, dtype=torch.bool, device=device)
        mask[idx] = 0
        mask = mask.view(batch_size, max_num_nodes)
        if cls_tokens:
            mask[:, -1] = 0
        return new_x, mask
    return new_x


def unpad_dense_batch(x, mask, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
    if cls_tokens:
        new_x = x[:, :-1][~mask[:, :-1]]
        new_x = torch.cat((new_x, x[:, -1]))
    else:
        new_x = x[~mask]
    return new_x


def to_nested_tensor(
    x: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Copied from PyG implementation"""
    if ptr is not None:
        offsets = ptr[1:] - ptr[:-1]
        sizes: List[int] = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    elif batch is not None:
        offsets = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        sizes: List[int] = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    else:
        xs = [x]

    # This currently copies the data, although `x` is already contiguous.
    # Sadly, there does not exist any (public) API to preven this :(
    return torch.nested.as_nested_tensor(xs)


def from_nested_tensor(
    x: Tensor,
    return_batch: bool = False,
):
    """Copied from PyG implementation"""
    if not x.is_nested:
        raise ValueError("Input tensor in 'from_nested_tensor' is not nested")

    sizes = x._nested_tensor_size()

    for dim, (a, b) in enumerate(zip(sizes[0, 1:], sizes.t()[1:])):
        if not torch.equal(a.expand_as(b), b):
            raise ValueError(
                f"Not all nested tensors have the same size "
                f"in dimension {dim + 1} "
                f"(expected size {a.item()} for all tensors)"
            )

    out = x.contiguous().values()
    out = out.view(-1, *sizes[0, 1:].tolist())

    if not return_batch:
        return out

    batch = torch.arange(x.size(0), device=x.device)
    batch = batch.repeat_interleave(sizes[:, 0].to(batch.device))

    return out, batch


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_inverse_sqrt_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        return math.sqrt(warmup_epochs / epoch)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def flatten_lists(lists: List) -> List:
    """Removes nested lists.

    Args:
        lists (List): List of lists to flatten

    Returns:
        List: Flattened list
    """
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:
            continue
    return result


def distribute_function(
    func: Callable,
    X: Iterable,
    n_jobs: int,
    description: str = "",
    total: int = 1,
    **kwargs,
) -> Any:
    """Distributes function `func` over iterable `X` using `n_jobs` cores.

    Args:
        func (Callable): function to be distributed
        X (Iterable): iterable over which the function is distributed
        n_jobs (int): number of cores to use
        description (str, optional): Description of the progress. Defaults to "".
        total (int, optional): Total number of elements in `X`. Defaults to 1.

    Returns:
        Any: result of the `func` applied to `X`.
    """

    if total == 1:
        total = len(X)  # type: ignore

    with tqdm_joblib(tqdm(desc=description, total=total)):
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt


def make_batches(l, n):
    """Make chunks of size n from list l"""
    return [l[i : i + n] for i in range(0, len(l), n)]


def rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def get_rbf(X):
    D = torch.sqrt(torch.sum((X[:, None, :] - X[None, :, :]) ** 2, -1) + 1e-6)
    return rbf(D, 16)


def get_graph_from_ps_protein_worker(
    protein, eps=8.0, use_rbfs=True, mask_cls_idx=False
):
    """Converts an avro dictionary to a pytorch geometric graph."""
    sequence = torch.LongTensor(
        [
            esm_alphabet.get_idx(res)
            for res in esm_alphabet.tokenize(protein["protein"]["sequence"])
        ]
    )

    coords = torch.tensor(
        [
            protein["residue"]["x"],
            protein["residue"]["y"],
            protein["residue"]["z"],
        ]
    ).T

    graph_adj = radius_neighbors_graph(coords, radius=eps, mode="connectivity")
    row, col = torch.nonzero(torch.tensor(graph_adj.todense()), as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)

    sequence = torch.cat(
        [
            torch.LongTensor([esm_alphabet.cls_idx]),
            sequence,
            torch.LongTensor([esm_alphabet.eos_idx]),
        ]
    )

    edge_index = edge_index + 1

    if use_rbfs:
        rbf_dist = get_rbf(coords).squeeze()
        edge_attr = rbf_dist[row, col]
    else:
        edge_attr = None

    return Data(edge_index=edge_index, x=sequence, edge_attr=edge_attr)


def get_graph_from_ps_protein(protein, eps=8.0, use_rbfs=True):
    if isinstance(protein, list):
        res = list()
        for p in protein:
            res.append(get_graph_from_ps_protein_worker(p, eps, use_rbfs))
        return res
    else:
        return get_graph_from_ps_protein_worker(protein, eps, use_rbfs)


def download_url_content(url: str, file_path: str) -> Optional[str]:
    """
    Downloads the content from a given URL and saves it to a specified file.

    Args:
    url (str): The URL of the resource to be downloaded.
    file_path (str): The file path where the content should be saved.

    Returns:
    Optional[str]: Error message in case of failure, None otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, "wb") as file:
            file.write(response.content)
    except requests.exceptions.HTTPError as e:
        raise SystemExit(e)
    except requests.exceptions.ConnectionError as e:
        raise SystemExit(e)
    except requests.exceptions.Timeout as e:
        raise SystemExit(e)
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
