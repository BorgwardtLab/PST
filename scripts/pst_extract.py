import argparse
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from pst.esm2 import PST
from example_dataset import ExampleDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use PST to extract per-token representations \
        for pdb files stored in datadir/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./scripts/examples",
        help="Path to the dataset, pdb files should be stored in datadir/raw/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pst_t6",
        help="Name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "--include-seq",
        action='store_true',
        help="Add sequence representation to the final representation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the data loader"
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default=None,
        help="How to aggregate protein representations across layers. \
        `None`: last layer; `mean`: mean pooling, `concat`: concatenation",
    )
    cfg = parser.parse_args()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    embeddings = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        if cfg.include_seq:
            if "so" not in cfg.model:
                raise ValueError("Use models pretrained using struct only updates strategy!")
            data.edge_index = None
            out_seq = model(data, return_repr=True, aggr=cfg.aggr)
            out_seq = out_seq[data.idx_mask]
            out = (out + out_seq) * 0.5
        embeddings = embeddings + list(unbatch(out, batch))
    return embeddings


def main():
    cfg = parse_args()

    pretrained_path = Path(f".cache/pretrained_models/{cfg.model}.pt")
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model, pretrained_path
        )
    except:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model,
            pretrained_path,
            map_location=torch.device("cpu"),
        )
    model.eval()
    model.to(cfg.device)

    dataset = ExampleDataset(
        root=cfg.datadir,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    protein_repr_all = compute_repr(data_loader, model, cfg)
    for protein_repr in protein_repr_all:
        print("Shape of representation (length, d_model):")
        print(protein_repr.shape)
        print(protein_repr)


if __name__ == "__main__":
    main()
