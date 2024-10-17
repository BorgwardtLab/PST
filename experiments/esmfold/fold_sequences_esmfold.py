# -*- coding: utf-8 -*-
"""fold_sequences_esmfold.py
This must be executed separately to ensure that all structures are folded

"""

import pickle
from pathlib import Path

import esm
import hydra
import pandas as pd
import submitit
import torch
import torchdrug
from easydict import EasyDict as edict
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pyprojroot import here
from torchdrug import core, datasets, models, tasks  # noqa

from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding, EsmModel
# from transformers.models.esm.openfold_utils.protein import to_pdb, from_prediction, Protein
# from tqdm.rich import tqdm


# def fold_sequence(idx: int, sequence: str, model: torch.nn.Module, path_prefix: Path) -> int:
#     """
#     Folds a protein sequence and writes the resulting structure to a PDB file.

#     Args:
#         idx (int): Index of the sequence.
#         sequence (str): Protein sequence to be folded.
#         model (EsmForProteinFolding): Protein folding model.
#         path_prefix (Path): Directory prefix where the PDB file will be saved.

#     Returns:
#         int: The index of the sequence.
#     """
#     # logger.info(f"Folding sequence {idx} - {sequence}")
#     out_path = path_prefix / (str(idx) + ".pdb")
    
#     if not out_path.exists():
#         with torch.no_grad():
#             outputs = model.infer_pdb(sequence)
#         # logger.info(f"PDB file produced")
#         # Write to pdb file
        
#         with open(out_path, "w") as f:
#             f.write(outputs)
#     # logger.info(f"Saved to {path_prefix / (str(idx) + '.pdb')}")
#     return out_path

def fold_sequence_batch(sequences: list, path_prefix: Path) -> list:
    """
    Folds a batch of protein sequences and writes the resulting structures to PDB files.

    Args:
        idx (int): Index of the sequence.
        sequences (list): List of protein sequences to be folded.
        model (EsmForProteinFolding): Protein folding model.
        path_prefix (Path): Directory prefix where the PDB file will be saved.

    Returns:
        list: List of indices of the sequences.
    """
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    # for idx, sequence in tqdm(sequences):
    for idx, sequence in sequences:
        out_path = path_prefix / (str(idx) + ".pdb")
        if not out_path.exists():
            with torch.no_grad():
                outputs = model.infer_pdb(sequence) 
            # Write to pdb file
            with open(out_path, "w") as f:
                f.write(outputs)

# def contact_prediction_batch(sequences: list, path_prefix: Path) -> list:
    
#     # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
#     # model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")
#     sequences = sequences[:1]
#     model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#     batch_converter = alphabet.get_batch_converter()
#     model.eval()
#     batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
#     batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#     with torch.no_grad():
#         results = model(batch_tokens, return_contacts=True)

#     return out_paths
 
def submitit_executor_wrapper(cfg, sequences_batches, folding_function, path_prefix):
    executor = submitit.AutoExecutor(folder=str(here() / "logs"))
    executor.update_parameters(
        slurm_array_parallelism=cfg.compute.array_parallelism,
        cpus_per_task=cfg.compute.cpus_per_task,
        slurm_mem_per_cpu=cfg.compute.mem_per_cpu,
        timeout_min=cfg.compute.timeout_min,
        slurm_job_name=cfg.compute.job_name,
        slurm_partition=cfg.compute.partition,
        slurm_gpus_per_node=cfg.compute.gpus_per_node,
        slurm_gpus_per_task=cfg.compute.gpus_per_task,
        slurm_tasks_per_node=cfg.compute.tasks_per_node,
    )

    total_jobs = len(sequences_batches)
    logger.info(f"Starting {total_jobs} jobs")
    
    # path_prefix = here() / "datasets/esmfold/structures/"

    jobs = []
    with executor.batch():
        for batch in sequences_batches:
            job = executor.submit(
                folding_function, sequences=batch, path_prefix=path_prefix
            )
            jobs.append(job)

# def loader_wrapper(loader, indices, cfg, partition="train", parallel=False):
    


@hydra.main(
    version_base="1.3",
    config_path=str(here() / "config"),
    config_name="esmfold_config",
)
def main(cfg):
    # task = core.Configurable.load_config_dict(
    #     edict(OmegaConf.to_container(cfg.task, resolve=True))
    # )
    cached_seq = here() / "datasets/.cached_esmfold_data.pkl"
    if not cached_seq.exists():
        dataset = core.Configurable.load_config_dict(
            OmegaConf.to_container(cfg.dataset, resolve=True)
        )
        # train, val, test = dataset.split()
        sequences = [
            (idx, protein["graph"].to_sequence().replace(".G", "").replace(".", ""))
            for idx, protein in tqdm(
                enumerate(dataset),
                desc="Extracting sequences",
                total=len(dataset),
            )
        ]
        sequences_batches = [
            sequences[i : i + cfg.data.fold_batch_size]
            for i in range(0, len(sequences), cfg.data.fold_batch_size)
        ]
        # Save list as pkl file
        with open(cached_seq, "wb") as file:
            pickle.dump(sequences_batches, file)
        
    else:
        with open(cached_seq, 'rb') as file:
            sequences_batches = pickle.load(file) 
    # if not cfg.parallel:
    #     for batch in tqdm(sequences_batches, desc="Folding sequences without submitit"):
    #         fold_sequence_batch(sequences=batch, path_prefix=here() / "datasets/esmfold/structures/")
    # else:
    submitit_executor_wrapper(cfg, sequences_batches, fold_sequence_batch, here() / "datasets/esmfold/structures/")
    # train_loader = torchdrug.data.DataLoader(
    #     train, batch_size=1, shuffle=False
    # )
    # val_loader = torchdrug.data.DataLoader(
    #     val, batch_size=1, shuffle=False
    # )
    # test_loader = torchdrug.data.DataLoader(
    #     test, batch_size=1, shuffle=False
    # )
    # loader_wrapper(train_loader, cfg, partition="train", parallel=cfg.parallel)
    # loader_wrapper(val_loader, cfg, partition="val", parallel=cfg.parallel)
    # loader_wrapper(test_loader, cfg, partition="test", parallel=cfg.parallel)

if __name__ == "__main__":
    main()