# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
# from Bio import SeqIO
import itertools
import json
import pathlib
import string
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from tqdm import tqdm

from data_utils.mutation import DeepSequenceDataset


def remove_insertions(sequence: str) -> str:
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file, automatically removes insertions.

    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa


def get_sequences():
    with open("./dms_seq.json", "r") as f:
        sequences = json.load(f)
    return pd.DataFrame(sequences["offset_list"])


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    # fmt: off
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--protein-id",
        type=int,
        default=-1,
        nargs='+',
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        default='../datasets/dms/raw/measurements.csv',
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        default="./outputs/esm.csv",
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="masked-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    # fmt: on
    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    return parser


# def label_row(row, sequence, token_probs, alphabet, offset_idx=1):
#     wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
#     assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

#     wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

#     # add 1 for BOS
#     score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
#     return score.item()


def label_row(row, sequence, token_probs, alphabet, offset_idx=1):
    scores = []
    for r in row.split(":"):
        wt, idx, mt = r[0], int(r[1:-1]) - offset_idx, r[-1]
        assert (
            sequence[idx] == wt
        ), "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score = (
            token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        )
        scores.append(score.item())
    return sum(scores)


def compute_masked_marginals(
    row, batch_tokens, sequence, model, alphabet, offset_idx=1
):
    batch_tokens_masked = batch_tokens.clone()
    indices = []
    for r in row.split(":"):
        wt, idx, mt = r[0], int(r[1:-1]) - offset_idx, r[-1]
        assert (
            sequence[idx] == wt
        ), "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        batch_tokens_masked[0, idx + 1] = alphabet.mask_idx
        indices.append([wt_encoded, idx, mt_encoded])

    with torch.no_grad():
        token_probs = torch.log_softmax(
            model(batch_tokens_masked.cuda())["logits"], dim=-1
        )

    scores = []
    for wt_encoded, idx, mt_encoded in indices:
        score = (
            token_probs[0, idx + 1, mt_encoded] - token_probs[0, idx + 1, wt_encoded]
        )
        scores.append(score.item())
    return sum(scores)


# def compute_pppl(row, sequence, model, alphabet, offset_idx=1):
#     wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
#     assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

#     # modify the sequence
#     sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

#     # encode the sequence
#     data = [
#         ("protein1", sequence),
#     ]

#     batch_converter = alphabet.get_batch_converter()

#     batch_labels, batch_strs, batch_tokens = batch_converter(data)

#     wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

#     # compute probabilities at each position
#     log_probs = []
#     for i in range(1, len(sequence) - 1):
#         batch_tokens_masked = batch_tokens.clone()
#         batch_tokens_masked[0, i] = alphabet.mask_idx
#         with torch.no_grad():
#             token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
#         log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
#     return sum(log_probs)


def main(args):
    # Load the deep mutational scan
    # df = pd.read_csv(args.dms_input)
    # df = get_dms(args.protein_name, args.dms_input)
    df = pd.read_csv(args.dms_input)
    dms_seq = get_sequences()
    print(dms_seq)

    protein_ids = DeepSequenceDataset.available_ids()
    if isinstance(args.protein_id, list) and args.protein_id[0] != -1:
        protein_ids = [protein_ids[i] for i in args.protein_id if i < len(protein_ids)]
    print(f"# of Datasets: {len(protein_ids)}")

    final_df = []

    # inference for each model
    for model_location in args.model_location:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
            print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

        for protein_id in protein_ids:
            print(protein_id)
            current_df = df[df["protein_name"] == protein_id].copy()
            seq_df = dms_seq[dms_seq["protein_name"] == protein_id]
            sequence = seq_df["wt_sequence"].iloc[0]
            offset_idx = seq_df["offset"].iloc[0]
            print(sequence)
            print(offset_idx)

            data = [
                ("protein1", sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if args.scoring_strategy == "wt-marginals":
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens.cuda())["logits"], dim=-1
                    )
                current_df[model_location] = current_df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "masked-marginals":
                tqdm.pandas()
                current_df[model_location] = current_df.progress_apply(
                    lambda row: compute_masked_marginals(
                        row[args.mutation_col],
                        batch_tokens,
                        sequence,
                        model,
                        alphabet,
                        offset_idx=offset_idx,
                    ),
                    axis=1,
                )
            #     all_token_probs = []
            #     for i in tqdm(range(batch_tokens.size(1))):
            #         batch_tokens_masked = batch_tokens.clone()
            #         batch_tokens_masked[0, i] = alphabet.mask_idx
            #         with torch.no_grad():
            #             token_probs = torch.log_softmax(
            #                 model(batch_tokens_masked.cuda())["logits"], dim=-1
            #             )
            #         all_token_probs.append(token_probs[:, i])  # vocab size
            #     token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            #     df[model_location] = df.apply(
            #         lambda row: label_row(
            #             row[args.mutation_col],
            #             sequence,
            #             token_probs,
            #             alphabet,
            #             offset_idx,
            #         ),
            #         axis=1,
            #     )
            # elif args.scoring_strategy == "pseudo-ppl":
            #     tqdm.pandas()
            #     df[model_location] = df.progress_apply(
            #         lambda row: compute_pppl(
            #             row[args.mutation_col], sequence, model, alphabet, offset_idx
            #         ),
            #         axis=1,
            #     )
            final_df.append(current_df)
        df = pd.concat(final_df)

    df.to_csv(args.dms_output)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
