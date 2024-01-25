import argparse
import os
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
import scipy
import torch
from proteinshake.utils import residue_alphabet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pst.data.mutation import DeepSequenceDataset
from pst.esm2 import PST
from pst.transforms import MutationDataset


def load_args():
    parser = argparse.ArgumentParser(
        description="Use ESM2SAT for mutation prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=str, default='.cache/pst', help="directory for downloading models"
    )
    parser.add_argument(
        "--model", type=str, default='pst_t6', help="pretrained model names (see README for models)"
    )
    parser.add_argument(
        "--datapath", type=str, default="./datasets", help="dataset prefix"
    )
    parser.add_argument("--dataset", type=str, default="dms", help="which dataset?")
    parser.add_argument(
        "--protein_id", type=int, default=-1, nargs="+", help="protein id list"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["masked", "wt", "mt", "mt-all"],
        default="masked",
        help="scoring strategy: masked marginals or wildtype marginals",
    )
    args = parser.parse_args()

    args.datapath = Path(args.datapath) / args.dataset
    args.log_path = Path(args.pretrained_prefix) / args.dataset

    args.device = (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    return args


@torch.no_grad()
def predict_masked(model, data_loader):
    logits = []
    y_true = []
    all_scores = []
    sample_lengths = []
    mt_indices = []
    wt_indices = []
    tic = timer()
    for data in tqdm(data_loader, desc="Predicting"):
        data = data.to(cfg.device)
        out = model.mask_predict(data)
        probs = torch.log_softmax(out, dim=-1)
        if cfg.strategy == "mt-all":
            score = probs.gather(-1, data.x.view(-1, 1))
        else:
            score = probs.gather(-1, data.mt_indices) - probs.gather(
                -1, data.wt_indices
            )
        logits.append(out.cpu())
        y_true.append(data.y.cpu())
        all_scores.append(score.sum(dim=0).cpu())
        mt_indices.append(data.mt_indices.cpu())
        wt_indices.append(data.wt_indices.cpu())
        sample_lengths.append(len(out))
    toc = timer()

    logits = torch.cat(logits)
    y_true = torch.cat(y_true)
    all_scores = torch.cat(all_scores)
    mt_indices = torch.cat(mt_indices)
    wt_indices = torch.cat(wt_indices)
    return {
        "probabilities": logits,
        "y_true": y_true,
        "y_score": all_scores,
        "mt_indices": mt_indices,
        "wt_indices": wt_indices,
        "sample_lengths": sample_lengths,
        "total_time": toc - tic,
    }


def label_row_wt(row, probs):
    row = row.split()
    wt_indices = torch.tensor(
        list(map(lambda x: residue_alphabet.index(x[0]), row))
    ).view(-1, 1)
    mt_indices = torch.tensor(
        list(map(lambda x: residue_alphabet.index(x[-1]), row))
    ).view(-1, 1)
    score = probs.gather(-1, mt_indices) - probs.gather(-1, wt_indices)
    return score.sum(dim=0).item()


def main():
    global cfg
    cfg = load_args()
    print(cfg)

    if cfg.dataset == "dms":
        dataset_cls = DeepSequenceDataset
    else:
        raise ValueError("Not supported!")

    protein_ids = dataset_cls.available_ids()
    if isinstance(cfg.protein_id, list) and cfg.protein_id[0] != -1:
        protein_ids = [protein_ids[i] for i in cfg.protein_id if i < len(protein_ids)]
    else:
        cfg.protein_id = list(range(len(protein_ids)))
    print(f"# of Datasets: {len(protein_ids)}")

    dataset = dataset_cls(root=cfg.datapath)
    mutations_list = dataset.mutations

    pretrained_path = Path(f"{cfg.model_dir}/{cfg.model}.pt")
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    model, model_cfg = PST.from_pretrained_url(
        cfg.model, pretrained_path
    )

    model.eval()
    model.to(cfg.device)
    dataset = dataset.to_graph(eps=model_cfg.data.graph_eps).pyg()

    all_results = defaultdict(list)
    all_scores = []

    for i, protein_id in zip(cfg.protein_id, protein_ids):
        print("-" * 40)
        print(f"Protein id: {protein_id}")

        mutations = mutations_list[i]
        graph, protein_dict = dataset[i]

        df = mutations.copy()
        df.rename(columns={"y": "effect"}, inplace=True)
        df["protein_id"] = protein_id
        df = df[["protein_id", "mutations", "effect"]]
        if graph.num_nodes > 3000:
            all_scores.append(df)
            continue
        if cfg.strategy == "masked" or cfg.strategy == "mt" or cfg.strategy == "mt-all":
            ds = MutationDataset(
                graph,
                protein_dict,
                mutations,
                strategy=cfg.strategy,
            )
            data_loader = DataLoader(ds, batch_size=1, shuffle=False)
            results = predict_masked(model, data_loader)

            if cfg.strategy == "mt-all":
                data_loader = DataLoader([graph], batch_size=1, shuffle=False)
                graph = next(iter(data_loader)).to(cfg.device)
                with torch.no_grad():
                    out = model.mask_predict(graph)
                    probs = torch.log_softmax(out, dim=-1).cpu()
                bias = probs.gather(-1, graph.x.cpu().view(-1, 1)).sum(dim=0)
                results["y_score"] = results["y_score"] - bias

            current_dir = cfg.log_path / f"{protein_id}"
            print(current_dir)
            os.makedirs(current_dir, exist_ok=True)
            torch.save(results, current_dir / "results.pt")

            df["ESM2SAT"] = results["y_score"]
        elif cfg.strategy == "wt":
            data_loader = DataLoader([graph], batch_size=1, shuffle=False)
            graph = next(iter(data_loader)).to(cfg.device)
            with torch.no_grad():
                out = model.mask_predict(graph)
                probs = torch.log_softmax(out, dim=-1).cpu()

            df["ESM2SAT"] = df.apply(
                lambda row: label_row_wt(
                    row["mutations"],
                    probs,
                ),
                axis=1,
            )

        rho = scipy.stats.spearmanr(df["effect"], df["ESM2SAT"])
        print(f"Spearman: {rho}")

        all_scores.append(df)
        all_results["protein_id"].append(protein_id)
        all_results["spearmanr"].append(rho.correlation)

    all_results = pd.DataFrame.from_dict(all_results)
    all_results.to_csv(cfg.log_path / "results.csv")
    all_scores = pd.concat(all_scores, ignore_index=True)
    all_scores.to_csv(cfg.log_path / "scores.csv")


if __name__ == "__main__":
    main()
