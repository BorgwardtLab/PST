import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fastaparser
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from biopandas.mmcif import PandasMmcif
from fastavro import reader as avro_reader

from .utils import AA_THREE_TO_ONE, flatten_lists


def read_avro_file(
    file_path: str,
    struct_to_load: Union[int, None] = None,
) -> List[Any]:
    """Reads protein structures from an AlphaFold2 Avro file for Homo Sapiens.

    This function reads protein structures from an AlphaFold2 dataset. If the
    `struct_to_load` parameter is specified, it reads up to that number of
    protein structures. If `struct_to_load` is None, all available structures
    are read from the file.

    Args:
        file_path (str): The file path to the AlphaFold2 Avro dataset. Default is
            './data/AlphaFoldDataset_homo_sapiens.residue.avro'.
        struct_to_load (Union[int, None]): The number of protein structures to read
            from the dataset. If None, all available structures are read. Default is None.

    Returns:
        List[Any]: A list of protein structures.

    Examples:
        >>> read_af2_homo_sapiens(struct_to_load=10)
        [...]
    """
    proteins = []
    cnt = 1

    with open(file_path, "rb") as file:
        for p in avro_reader(file):
            cnt += 1
            proteins.append(p)
            if struct_to_load is not None:
                if struct_to_load < cnt:
                    break
                else:
                    continue

    return proteins


class Featurizer(nn.Module):
    """Featurizer"""

    def __init__(self, knn_graph, args):
        super(Featurizer, self).__init__()
        self.args = args

    def forward(self, batch, shuffle_fraction=0.0):
        """Featurize a batch of proteins"""
        X, S, mask, lengths = featurize_GTrans(batch, shuffle_fraction)
        return _get_features(self, S, X, mask)


def featurize_GTrans(batch, shuffle_fraction=0.0):
    """Pack and pad batch into torch tensors"""
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    B = len(batch)
    lengths = np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max = max([len(b["seq"]) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    np.ones([B, L_max]) * 100.0

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
        if shuffle_fraction > 0.0:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
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
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask, lengths


def _get_features(self, S, X, mask):
    device = X.device
    mask_bool = mask == 1
    B, N, _, _ = X.shape
    X_ca = X[:, :, 1, :]
    D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

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
    X[:, :, 3, :]
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
                    vars()["atom_" + atom1], vars()["atom_" + atom2], None, self.num_rbf
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
        rbf = _get_rbf(
            vars()["atom_" + atom1], vars()["atom_" + atom2], E_idx, self.num_rbf
        )
        edge_dist.append(edge_mask_select(rbf))

    E_dist = torch.cat(tuple(edge_dist), dim=-1)

    h_V = []
    if self.args.node_dist:
        h_V.append(V_dist)
    if self.args.node_angle:
        h_V.append(V_angles)
    if self.args.node_direct:
        h_V.append(V_direct)

    h_E = []
    if self.args.edge_dist:
        h_E.append(E_dist)
    if self.args.edge_angle:
        h_E.append(E_angles)
    if self.args.edge_direct:
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


def _rbf(D, num_rbf):
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
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, :, None, :]) ** 2, -1) + 1e-6
        )
        RBF_A_B = _rbf(D_A_B, num_rbf)
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


def create_file(filename: str) -> None:
    """
    Create an empty file with the given filename.

    Args:
        filename: Name of the file to be created.
    """
    with open(filename, "w"):
        pass


def add_line_to_file(filename: str, line: str) -> None:
    """
    Append a line to an existing file.

    Args:
        filename: Name of the file to append the line to.
        line: The line to append to the file.
    """
    with open(filename, "a") as f:
        f.write(f"{line}\n")


def read_fasta(filename: str) -> Dict[str, str]:
    """
    Reads a FASTA file and returns a dictionary where keys are sequence identifiers and
        values are sequences.

    Args:
        filename (str): The path to the FASTA file.

    Returns:
        Dict[str, str]: A dictionary where the keys are sequence identifiers and the
            values are sequences.
    """
    sequences = []
    with open(filename) as fasta_file:
        parser = fastaparser.Reader(fasta_file)
        for seq in parser:
            sequences.append(
                {
                    "id": seq.id.split("|")[0],
                    "chain": seq.description.split("|")[0],
                    "seq": seq.sequence_as_string(),
                }
            )
    return sequences


def build_df(proteins):
    protein_dfs = []
    for protein in proteins:
        protein_df = pd.DataFrame(protein["name"])
        protein_df["ID"] = protein["protein"]["ID"]
        protein_dfs.append(protein_df)
    return pd.concat(protein_dfs)


def filter_backbone(df):
    return df[df["label_atom_id"].isin(["CA", "C", "N", "O"])]


def read_ec_split(split="train"):
    split_pdb = pd.read_csv(f"data/nrPDB-EC_2020.04_{split}.txt", header=None)
    split_pdb["split"] = split
    split_pdb = split_pdb.rename(columns={0: "pdb_id_chain"})
    split_pdb = replace_obsolete_pdb_ids(split_pdb)
    split_pdb["pdb_id"] = split_pdb["pdb_id_chain"].apply(lambda x: x[:4])
    split_pdb = remove_ca_only_pdbs(split_pdb)
    split_pdb = remove_unmappable(split_pdb)
    split_pdb["chain_id"] = split_pdb["pdb_id_chain"].apply(lambda x: x[5:]).str.upper()
    return split_pdb


def download_cif(pdb_id: str, dest_dir: str) -> Optional[str]:
    """Download a CIF file from RCSB.

    Args:
        pdb_id (str): The PDB ID of the protein structure.
        dest_dir (str): The directory where the CIF file should be saved.

    Returns:
        Optional[str]: The path to the downloaded file, or None if download failed.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if os.path.join(dest_dir, f"{pdb_id}.cif") in os.listdir(dest_dir):
        return os.path.join(dest_dir, f"{pdb_id}.cif")
    if response.status_code == 200:
        file_path = os.path.join(dest_dir, f"{pdb_id}.cif")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print(f"Failed to download {pdb_id}. HTTP Status Code: {response.status_code}")
        return None


def download_fasta(pdb_id: str, dest_dir: str) -> Optional[str]:
    """Download a FASTA file from RCSB.

    Args:
        pdb_id (str): The PDB ID of the protein structure.
        dest_dir (str): The directory where the FASTA file should be saved.

    Returns:
        Optional[str]: The path to the downloaded file, or None if download failed.
    """
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if os.path.join(dest_dir, f"{pdb_id}.fasta") in os.listdir(dest_dir):
        return os.path.join(dest_dir, f"{pdb_id}.cif")

    if response.status_code == 200:
        file_path = os.path.join(dest_dir, f"{pdb_id}.fasta")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print(f"Failed to download {pdb_id}. HTTP Status Code: {response.status_code}")
        add_line_to_file("data/missing_fasta.txt", pdb_id)
        return None


def download_id(pdb_id: str, dest_dir_cif: str, dest_dir_fasta: str) -> Optional[str]:
    """Download a FASTA file and a CIF file from RCSB.

    Args:
        pdb_id (str): The PDB ID of the protein structure.
        dest_dir (str): The directory where the files should be saved.

    Returns:
        Optional[str]: The path to the downloaded file, or None if download failed.
    """
    download_cif(pdb_id, dest_dir_cif)
    download_fasta(pdb_id, dest_dir_fasta)


def fill_missing_and_remove_duplicates(
    df: pd.DataFrame,
    sequence,
    seq_id_col="label_seq_id",
    atom_id_col="label_atom_id",
    min_len_longest_stretch=4,
) -> List[List[float]]:
    """
    Remove duplicates and fill missing values in label_seq_id.

    Args:
        df (pd.DataFrame): Input DataFrame with label_seq_id.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed and missing label_seq_id filled.
    """
    old_df = df
    old_df
    df = df.sort_values([seq_id_col, atom_id_col])
    df = df.drop_duplicates(subset=[seq_id_col, atom_id_col], keep="first")

    df = df.pivot_table(
        index=seq_id_col, columns=atom_id_col, values=None, aggfunc="first"
    ).reset_index()
    df.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in df.columns.values
    ]

    # drop rows with missing atoms
    df = df.dropna(
        subset=flatten_lists(
            [[f"x_{atom}", f"y_{atom}", f"z_{atom}"] for atom in ["N", "CA", "C", "O"]]
        )
    )

    differences = df["resid"].diff()
    starts = np.where(differences > 1)[0]

    starts = np.r_[0, starts]
    ends = np.r_[starts[1:] - 1, len(df) - 1]
    stretches = ends - starts

    longest_start, longest_end = (
        starts[np.argmax(stretches)],
        ends[np.argmax(stretches)],
    )
    df_longest_stretch = df.iloc[longest_start:longest_end]
    longest_resolved_stretch = "".join(df_longest_stretch["resname_CA"].tolist())
    longest_resolved_stretch = longest_resolved_stretch[:10]
    if len(longest_resolved_stretch) < min_len_longest_stretch:
        print("Longest stretch too short")
    if longest_resolved_stretch not in sequence:
        print("Longest stretch not in sequence")

    start_longest_stretch_in_sequence = sequence.find(longest_resolved_stretch)
    # add padding until start of longest stretch
    missing_residues = sequence[:start_longest_stretch_in_sequence]
    df_missing = pd.DataFrame(
        {
            seq_id_col: np.arange(
                df[seq_id_col].min() - len(missing_residues),
                df[seq_id_col].min(),
            ),
            "resname_CA": list(missing_residues),
        }
    )
    df = pd.concat([df_missing, df], ignore_index=True)
    # For every row that is not consecutive, add missing residues using the sequence
    df["resname_CA"] = df["resname_CA"].fillna(pd.Series(list(sequence)))
    # Add padding until end of the sequence
    missing_residues = sequence[len(df["resname_CA"]) :]
    df_missing = pd.DataFrame(
        {
            seq_id_col: np.arange(
                df[seq_id_col].max() + 1,
                df[seq_id_col].max() + 1 + len(missing_residues),
            ),
            "resname_CA": list(missing_residues),
        }
    )
    df = pd.concat([df, df_missing], ignore_index=True)

    return [
        df[[f"x_{atom_type}", f"y_{atom_type}", f"z_{atom_type}"]].values.tolist()
        for atom_type in ["N", "CA", "C", "O"]
    ]


def cif2backbone(split_data):
    try:
        pmmcif = PandasMmcif()
        pmmcif_file = pmmcif.read_mmcif(split_data["cif_path"])
        atom_df = pmmcif_file.df["ATOM"]
        atom_df_old = pmmcif_file.df["ATOM"]
        atom_df_old
        atom_df = atom_df[atom_df["label_atom_id"].isin(["CA", "C", "N", "O"])]
        atom_df = atom_df[atom_df["auth_asym_id"] == split_data["chain_id"]]
        atom_df = atom_df[
            [
                "label_seq_id",
                "auth_seq_id",
                "auth_comp_id",
                "label_comp_id",
                "label_atom_id",
                "Cartn_x",
                "Cartn_y",
                "Cartn_z",
            ]
        ]
        # Change dtype of label_seq_id to int
        atom_df["label_seq_id"] = atom_df["label_seq_id"].astype(int)

        atom_df["label_comp_id"] = atom_df["label_comp_id"].apply(
            lambda x: AA_THREE_TO_ONE[x]
        )
        atom_df["auth_comp_id"] = atom_df["auth_comp_id"].apply(
            lambda x: AA_THREE_TO_ONE[x]
        )

        sequences = read_fasta(split_data["fasta_path"])

        # Find chain in sequences if key contains chain ID

        sequence = [
            seq["seq"]
            for seq in sequences
            if split_data["chain_id"] in seq["chain"].upper()
        ]
        auth_in_sequence = [seq for seq in sequences if "auth" in seq["chain"]]
        if len(auth_in_sequence) > 0:
            add_line_to_file("data/auth_in_sequence.txt", split_data["pdb_id"])
            return None
        if len(sequence) == 1:
            sequence = sequence[0]

        N_atoms, CA_atoms, C_atoms, O_atoms = fill_missing_and_remove_duplicates(
            atom_df, sequence
        )

        protein_dict = {
            "seq": sequence,
            "N": N_atoms,
            "CA": CA_atoms,
            "C": C_atoms,
            "O": O_atoms,
            "name": split_data["pdb_id"],
            "chain": split_data["chain_id"],
        }

        return protein_dict

    except Exception:
        print(f"Failed {split_data['pdb_id']}")
        add_line_to_file("data/failed_cif2backbone.txt", split_data["pdb_id"])
        return None


def read_txt_file_line_by_line(filename: str) -> List[str]:
    """
    Read a text file line by line and return its contents as a list of strings.

    Args:
        filename (str): Path to the text file to be read.

    Returns:
        List[str]: List containing lines in the text file.
    """
    lines = []
    with open(filename, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def replace_obsolete_pdb_ids(split_data):
    split = split_data["split"].unique()[0]
    obs_path = f"./.cache/{split}_obsolete.csv"
    if Path(f"./.cache/{split}_obsolete.csv").exists():
        obsolete = pd.read_csv(obs_path)
        mapping = dict(zip(obsolete["obsolete"], obsolete["superseding"]))
        withdrawn_id = ["5JM5"]
        for wid in withdrawn_id:
            split_data = split_data[~split_data["pdb_id_chain"].str.contains(wid)]
        # Replace obsolete PDB IDs
        split_data["pdb_id_chain"] = split_data["pdb_id_chain"].apply(
            lambda x: mapping.get(x.split("-")[0], x.split("-")[0])
            + "-"
            + x.split("-")[1]
            if "-" in x
            else x
        )
        # Remove withdrawn PDB IDs
        split_data = split_data[~split_data["pdb_id_chain"].isin(withdrawn_id)]
        return split_data
    else:
        return split_data


def remove_ca_only_pdbs(split_data):
    split = split_data["split"].unique()[0]
    ca_only_path = f".cache/{split}_ca_only_pdbs.txt"
    if Path(ca_only_path).exists():
        ca_only_pdbs = read_txt_file_line_by_line(ca_only_path)
        split_data = split_data[~split_data["pdb_id"].isin(ca_only_pdbs)]
        return split_data
    else:
        return split_data


def remove_unmappable(split_data):
    split = split_data["split"].unique()[0]
    path = f".cache/{split}_unmappable.txt"
    if Path(path).exists():
        unmappable = read_txt_file_line_by_line(path)
        split_data = split_data[~split_data["pdb_id"].isin(unmappable)]
        return split_data
    else:
        return split_data
