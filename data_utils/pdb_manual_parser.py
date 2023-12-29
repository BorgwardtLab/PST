# -*- coding: utf-8 -*-
"""pdb_manual_parser.py

Try to parse the PDBs 
"""
import os
from pathlib import Path
from typing import Dict

import fastaparser
import hydra
import pdbreader
from protein_rep_learning.feat import fill_missing_and_remove_duplicates
from protein_rep_learning.utils import AA_THREE_TO_ONE, distribute_function
from pyprojroot import here


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
                    "id": seq.id.split("-")[0],
                    "chain": seq.id.split("-")[1],
                    "seq": seq.sequence_as_string(),
                }
            )
    return sequences


def pdb2backbone(fname, sequences):
    # try:
    pmmcif_file = pdbreader.read_pdb(fname)
    atom_df = pmmcif_file["ATOM"]
    atom_df_old = atom_df.copy()
    atom_df_old
    chain_id = Path(fname).stem.split("-")[1].split("_")[0]
    pdb_id = Path(fname).stem.split("-")[0]
    atom_df = atom_df[atom_df["name"].isin(["CA", "C", "N", "O"])]
    if len(atom_df["name"].unique()) != 4:
        print(f"Could not find all backbone atoms in {fname}")
        return None

    # atom_df = atom_df[atom_df["chain_id"] == chain_id]
    atom_df = atom_df[
        [
            "x",
            "y",
            "z",
            "resname",
            "resid",
            "name",
        ]
    ]
    # Change dtype of label_seq_id to int
    atom_df["resid"] = atom_df["resid"].astype(float)

    atom_df["resname"] = atom_df["resname"].apply(lambda x: AA_THREE_TO_ONE[x])
    # Find chain in sequences if key contains chain ID

    sequence = [seq["seq"] for seq in sequences if pdb_id == seq["id"]]
    if len(sequence) > 1:
        sequence = [
            seq["seq"]
            for seq in sequences
            if pdb_id == seq["id"] and chain_id == seq["chain"]
        ]
        if len(sequence) == 1:
            sequence = sequence[0]
        else:
            raise ValueError(
                f"Could not find sequence for {pdb_id} and chain {chain_id}"
            )
    else:
        sequence = sequence[0]

    N_atoms, CA_atoms, C_atoms, O_atoms = fill_missing_and_remove_duplicates(
        atom_df, sequence, seq_id_col="resid", atom_id_col="name"
    )

    protein_dict = {
        "seq": sequence,
        "N": N_atoms,
        "CA": CA_atoms,
        "C": C_atoms,
        "O": O_atoms,
        "name": pdb_id,
        "chain": chain_id,
    }

    return protein_dict
    # except Exception:
    #     print("Could not parse file")
    #     return None


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="config"
)
def main(cfg):
    path = "/fs/pool/pool-hartout/Documents/Git/protein_rep_learning/data/td/EnzymeCommission/test/"
    sequences = read_fasta(
        "/fs/pool/pool-hartout/Documents/Git/protein_rep_learning/data/td/EnzymeCommission/nrPDB-EC_sequences.fasta"
    )
    paths = [path + "/" + fname for fname in os.listdir(path)]
    # for fname in tqdm(paths):
    #     _ = pdb2backbone(fname, sequences)

    res = distribute_function(
        func=pdb2backbone,
        X=paths,
        n_jobs=cfg.compute.n_jobs,
        description="Parsing PDBs",
        sequences=sequences,
    )
    res
    # for fname in os.listdir(path):
    #     fname = path + "/" + fname
    #     pdb2backbone(fname, sequences)
    # sequences


if __name__ == "__main__":
    main()
