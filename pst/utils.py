import contextlib
import json
import math
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional

import esm
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from fastavro import reader as avro_reader
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


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.

    Code stolen from https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    return device


def make_batches(l, n):
    """Make chunks of size n from list l"""
    return [l[i : i + n] for i in range(0, len(l), n)]


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def prepare(batch, shuffle_fraction=0.0):
    """Pack and pad batch into torch tensors"""
    alphabet = "ACDEFGHIKLMNPQRSTVWXY"
    B = len(batch)
    np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max = max([len(b["seq"]) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ["N", "CA", "C", "O"]], 1)  # [#atom, 4, 3]

        l = len(b["seq"])
        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )  # [#atom, 4, 3]
        X[i, :, :, :] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b["seq"]], dtype=np.int32)
        S[i, :l] = indices

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)  # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X) + np.nan
    for i, n in enumerate(numbers):
        X_new[i, :n, ::] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0
    # Conversion
    X = torch.from_numpy(X).to(dtype=torch.float32)
    S = torch.from_numpy(S).to(dtype=torch.int64)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask


def featurize_backbone(
    X,
    S,
    mask,
    top_k=30,
    num_rbf=16,
    node_dist=True,
    node_angle=True,
    node_direct=True,
    edge_dist=True,
    edge_angle=True,
    edge_direct=True,
):
    device = X.device
    mask_bool = mask == 1
    B, N, _, _ = X.shape
    X_ca = X[:, :, 1, :]
    D_neighbors, E_idx = _full_dist(X_ca, mask, top_k)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1

    def edge_mask_select(x):
        return torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(
            -1, x.shape[-1]
        )

    def node_mask_select(x):
        return torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

    randn = torch.rand(mask.shape, device=X.device) + 5
    decoding_order = torch.argsort(-mask * (torch.abs(randn)))
    mask_size = mask.shape[1]
    permutation_matrix_reverse = torch.nn.functional.one_hot(
        decoding_order, num_classes=mask_size
    ).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend2 = torch.gather(order_mask_backward, 2, E_idx)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1])
    mask_bw = (mask_1D * mask_attend2).unsqueeze(-1)
    mask_fw = (mask_1D * (1 - mask_attend2)).unsqueeze(-1)
    mask_bw = edge_mask_select(mask_bw).squeeze()
    mask_fw = edge_mask_select(mask_fw).squeeze()

    # sequence
    S = torch.masked_select(S, mask_bool)

    # angle & direction
    V_angles = _dihedrals(X, 0)
    V_angles = node_mask_select(V_angles)

    V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
    V_direct = node_mask_select(V_direct)
    E_direct = edge_mask_select(E_direct)
    E_angles = edge_mask_select(E_angles)

    # distance
    atom_N = X[:, :, 0, :]
    atom_Ca = X[:, :, 1, :]
    atom_C = X[:, :, 2, :]
    atom_O = X[:, :, 3, :]  # noqa
    b = atom_Ca - atom_N
    c = atom_C - atom_Ca
    torch.cross(b, c, dim=-1)

    node_list = ["Ca-N", "Ca-C", "Ca-O", "N-C", "N-O", "O-C"]
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split("-")
        node_dist.append(
            node_mask_select(
                _get_rbf(
                    vars()["atom_" + atom1], vars()["atom_" + atom2], None, num_rbf
                ).squeeze()
            )
        )

    V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

    pair_lst = [
        "Ca-Ca",
        "Ca-C",
        "C-Ca",
        "Ca-N",
        "N-Ca",
        "Ca-O",
        "O-Ca",
        "C-C",
        "C-N",
        "N-C",
        "C-O",
        "O-C",
        "N-N",
        "N-O",
        "O-N",
        "O-O",
    ]

    edge_dist = []  # Ca-Ca
    for pair in pair_lst:
        atom1, atom2 = pair.split("-")
        rbf = _get_rbf(vars()["atom_" + atom1], vars()["atom_" + atom2], E_idx, num_rbf)
        edge_dist.append(edge_mask_select(rbf))

    E_dist = torch.cat(tuple(edge_dist), dim=-1)

    h_V = []
    if node_dist:
        h_V.append(V_dist)
    if node_angle:
        h_V.append(V_angles)
    if node_direct:
        h_V.append(V_direct)

    h_E = []
    if edge_dist:
        h_E.append(E_dist)
    if edge_angle:
        h_E.append(E_angles)
    if edge_direct:
        h_E.append(E_direct)

    _V = torch.cat(h_V, dim=-1)
    _E = torch.cat(h_E, dim=-1)

    # edge index
    shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
    src = shift.view(B, 1, 1) + E_idx
    src = torch.masked_select(src, mask_attend).view(1, -1)
    dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(
        1, -1, 1
    ).expand_as(mask_attend)
    dst = torch.masked_select(dst, mask_attend).view(1, -1)
    E_idx = torch.cat((dst, src), dim=0).long()

    decoding_order = (
        node_mask_select((decoding_order + shift.view(-1, 1)).unsqueeze(-1))
        .squeeze()
        .long()
    )

    # 3D point
    sparse_idx = mask.nonzero()  # index of non-zero values
    X = X[sparse_idx[:, 0], sparse_idx[:, 1], :, :]
    batch_id = sparse_idx[:, 0]

    return X, S, _V, _E, E_idx, batch_id


def _full_dist(X, mask, top_k=30, eps=1e-6):
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = (1.0 - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1.0 - mask_2D) * (D_max + 1)
    D_neighbors, E_idx = torch.topk(
        D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False
    )
    return D_neighbors, E_idx


def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor


def _normalize(tensor, dim=-1):
    return nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def cal_dihedral(X, eps=1e-7):
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[
        :, 1:-1, :
    ]  # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:, 2:, :]  # N-C, CA-N, C-CA, ...

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)

    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    v = _normalize(torch.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v * u_1).sum(-1)) * torch.acos(cosD)  # TODO: sign

    return D


def _dihedrals(X, dihedral_type=0, eps=1e-7):
    B, N, _, _ = X.shape
    # psi, omega, phi
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)  # ['N', 'CA', 'C', 'O']
    D = cal_dihedral(X)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    # alpha, beta, gamma
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[:, 1:-1, :]  # C-CA, N-C, CA-N, ...
    cosD = (u_0 * u_1).sum(-1)  # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    D_features = torch.cat((Dihedral_Angle_features, Angle_features), 2)
    return D_features


def _hbonds(X, E_idx, mask_neighbors, eps=1e-3):
    X_atoms = dict(zip(["N", "CA", "C", "O"], torch.unbind(X, 2)))

    X_atoms["C_prev"] = F.pad(X_atoms["C"][:, 1:, :], (0, 0, 0, 1), "constant", 0)
    X_atoms["H"] = X_atoms["N"] + _normalize(
        _normalize(X_atoms["N"] - X_atoms["C_prev"], -1)
        + _normalize(X_atoms["N"] - X_atoms["CA"], -1),
        -1,
    )

    def _distance(X_a, X_b):
        return torch.norm(X_a[:, None, :, :] - X_b[:, :, None, :], dim=-1)

    def _inv_distance(X_a, X_b):
        return 1.0 / (_distance(X_a, X_b) + eps)

    U = (0.084 * 332) * (
        _inv_distance(X_atoms["O"], X_atoms["N"])
        + _inv_distance(X_atoms["C"], X_atoms["H"])
        - _inv_distance(X_atoms["O"], X_atoms["H"])
        - _inv_distance(X_atoms["C"], X_atoms["N"])
    )

    HB = (U < -0.5).type(torch.float32)
    neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1), E_idx)
    return neighbor_HB


def rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, :, None, :]) ** 2, -1) + 1e-6
        )
        RBF_A_B = rbf(D_A_B, num_rbf)
    return RBF_A_B


def _orientations_coarse_gl(X, E_idx, eps=1e-6):
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)

    n_0 = n_0[:, ::3, :]
    b_1 = b_1[:, ::3, :]
    X = X[:, ::3, :]

    O = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0, 0, 0, 1), "constant", 0)

    O_neighbors = gather_nodes(O, E_idx)
    X_neighbors = gather_nodes(X, E_idx)

    O = O.view(list(O.shape[:2]) + [3, 3]).unsqueeze(2)
    O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

    dX = X_neighbors - X.unsqueeze(-2)
    dU = torch.matmul(O, dX.unsqueeze(-1)).squeeze(-1)
    R = torch.matmul(O.transpose(-1, -2), O_neighbors)
    feat = torch.cat((_normalize(dU, dim=-1), _quaternions(R)), dim=-1)
    return feat


def _orientations_coarse_gl_tuple(X, E_idx, eps=1e-6):
    V = X.clone()
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)

    n_0 = n_0[:, ::3, :]
    b_1 = b_1[:, ::3, :]
    X = X[:, ::3, :]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0, 0, 0, 1), "constant", 0)  # [16, 464, 9]

    Q_neighbors = gather_nodes(Q, E_idx)  # [16, 464, 30, 9]
    X_neighbors = gather_nodes(V[:, :, 1, :], E_idx)  # [16, 464, 30, 3]
    N_neighbors = gather_nodes(V[:, :, 0, :], E_idx)
    C_neighbors = gather_nodes(V[:, :, 2, :], E_idx)
    O_neighbors = gather_nodes(V[:, :, 3, :], E_idx)

    Q = Q.view(list(Q.shape[:2]) + [3, 3]).unsqueeze(2)  # [16, 464, 1, 3, 3]
    Q_neighbors = Q_neighbors.view(
        list(Q_neighbors.shape[:3]) + [3, 3]
    )  # [16, 464, 30, 3, 3]

    dX = (
        torch.stack([X_neighbors, N_neighbors, C_neighbors, O_neighbors], dim=3)
        - X[:, :, None, None, :]
    )  # [16, 464, 30, 3]
    dU = torch.matmul(Q[:, :, :, None, :, :], dX[..., None]).squeeze(
        -1
    )  # [16, 464, 30, 3] 邻居的相对坐标
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K, -1)
    R = torch.matmul(Q.transpose(-1, -2), Q_neighbors)
    q = _quaternions(R)
    # edge_feat = torch.cat((dU, q), dim=-1) # 相对方向向量+旋转四元数

    dX_inner = V[:, :, [0, 2, 3], :] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B, N, -1)
    return V_direct, E_direct, q


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view(
        (neighbor_idx.shape[0], -1)
    )  # [4, 317, 30]-->[4, 9510]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(
        -1, -1, nodes.size(2)
    )  # [4, 9510, dim]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)  # [4, 9510, dim]
    return neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1]
    )  # [4, 317, 30, 128]


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(
        torch.abs(
            1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
        )
    )

    def _R(i, j):
        return R[:, :, :, i, j]

    signs = torch.sign(
        torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1)
    )
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    Q = torch.cat((xyz, w), -1)
    return _normalize(Q, dim=-1)


def write_dict_to_json(file_path: str, data: Dict) -> None:
    """Writes a dictionary to a JSON file.

    Args:
        file_path: The path where the JSON file will be stored.
        data: The dictionary to be written to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def read_json_to_dict(file_path: str) -> Dict:
    """Reads a JSON file into a dictionary.

    Args:
        file_path: The path of the JSON file to be read.

    Returns:
        A dictionary representation of the JSON file content.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_pickle(file_path: str, data: Any) -> None:
    """Writes data to a pickle file.

    Args:
        file_path (str): The path to the file to write to.
        data (Any): The data to pickle.

    Returns:
        None
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(file_path: str) -> Any:
    """Reads data from a pickle file.

    Args:
        file_path (str): The path to the file to read from.

    Returns:
        Any: The data read from the file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def find_keys_by_value(d: Dict[Any, Any], target_value: Any) -> List[Any]:
    """Find all keys in a dictionary that map to a given value.

    Args:
        d (Dict[Any, Any]): The dictionary to search.
        target_value (Any): The value to search for.

    Returns:
        List[Any]: A list of keys that map to the target value.
    """
    return [key for key, value in d.items() if value == target_value]


def compute_angles(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute the angles between each consecutive atom represented by a 3D coordinate.

    Args:
        coords (torch.Tensor): A tensor of shape (n, 3) representing the 3D coordinates of atoms.

    Returns:
        torch.Tensor: A tensor of angles between each consecutive atom.
    """
    # Calculate vectors between consecutive atoms
    vectors = coords[1:] - coords[:-1]

    # Normalize vectors
    norm_vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

    # Calculate dot products between consecutive vectors
    dot_products = torch.sum(norm_vectors[:-1] * norm_vectors[1:], dim=1)

    return torch.acos(torch.clamp(dot_products, -1.0, 1.0))


def compute_angle_features(coords: torch.Tensor) -> torch.Tensor:
    angles = compute_angles(coords)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    mean_cos = torch.tensor([torch.mean(cos_angles)])
    mean_sin = torch.tensor([torch.mean(sin_angles)])
    cos_angles = torch.cat([mean_cos, cos_angles, mean_cos])
    sin_angles = torch.cat([mean_sin, sin_angles, mean_sin])
    return torch.stack(
        [cos_angles, sin_angles],
    ).T


def get_rbf(X):
    D = torch.sqrt(torch.sum((X[:, None, :] - X[None, :, :]) ** 2, -1) + 1e-6)
    return rbf(D, 16)


def get_residue(residue, angle_features, alphabet, esm_alphabet):
    sequence_raw = "".join(
        [find_keys_by_value(alphabet, res)[0] for res in residue.tolist()]
    )

    # Remove trailing spaces
    sequence_str = [res for res in sequence_raw.strip().replace(" ", "X")]
    seq_tmp = list()
    for res in sequence_str:
        if "X" in res:
            seq_tmp.append(esm_alphabet.get_idx("<unk>"))
        else:
            seq_tmp.append(esm_alphabet.get_idx(res))
    sequence = torch.tensor(seq_tmp)
    # Add padding to match angle_features length
    padding_value = esm_alphabet.get_idx("<unk>")
    padding_length = angle_features.shape[0] - sequence.shape[0]
    if padding_length < 0:
        # Remove trailing residues
        sequence = sequence[:padding_length]
    else:
        padding = torch.full((padding_length,), padding_value, dtype=torch.long)
        sequence = torch.cat([sequence, padding])
    return sequence


def featurize_ca(sample, esm_alphabet=None, eps=8, use_rbfs=True, mask_cls_idx=True):
    if sample["position"].shape[0] == 0:
        return None
    else:
        graph_adj = radius_neighbors_graph(
            sample["position"], radius=eps, mode="connectivity"
        )
        if use_rbfs:
            rbf_dist = get_rbf(sample["position"]).squeeze()
            edge_attr = rbf_dist[row, col]
        else:
            edge_attr = None
        # mask out rbfs that are not connected
        row, col = torch.nonzero(torch.tensor(graph_adj.todense()), as_tuple=True)
        edge_index = torch.stack([row, col], dim=0)

        # Map residue integeters to sequences using alphabet where keys are letters and integers are values.
        sequence = torch.cat(
            [
                torch.LongTensor([esm_alphabet.cls_idx]),
                sample["sequence"],
                torch.LongTensor([esm_alphabet.eos_idx]),
            ]
        )
        edge_index = edge_index + 1

        if mask_cls_idx:
            idx_mask = torch.ones((len(sequence),), dtype=torch.bool)
            idx_mask[0] = idx_mask[-1] = False
        else:
            idx_mask = None

        return Data(
            x=sequence,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=sample["targets"].unsqueeze(1).T,
            idx_mask=idx_mask,
        )


def read_avro_file(path):
    content = list()
    with open(
        path,
        "rb",
    ) as file:
        total = int(avro_reader(file).metadata["number_of_proteins"])

    with open(path, "rb") as file:
        for x in tqdm(avro_reader(file), desc="Reading avro file", total=total):
            content.append(x)
    return content


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
