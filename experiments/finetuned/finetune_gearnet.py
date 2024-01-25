import copy
import logging
from pathlib import Path

import esm
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric.nn as gnn
import torchdrug
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from pyprojroot import here
from sklearn.neighbors import radius_neighbors_graph
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torchdrug import core, datasets, models, tasks
from tqdm import tqdm

from pst.esm2 import PST, ProteinNet
from pst.utils import get_linear_schedule_with_warmup

log = logging.getLogger(__name__)

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    embeddings = []
    for batch_idx, data in enumerate(tqdm(data_loader)):
        data = data.to(cfg.device)
        out = model(data, return_repr=True)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        out = gnn.global_mean_pool(out, batch)

        if cfg.include_seq:
            data.edge_index = None
            out_seq = model(data, return_repr=True)
            out_seq = out_seq[data.idx_mask]
            out_seq = gnn.global_mean_pool(out_seq, batch)
            out = (out + out_seq) * 0.5

        out = out.cpu()

        embeddings = embeddings + list(torch.chunk(out, len(data.ptr) - 1))

    return torch.cat(embeddings)


def get_structures(dataset, task, eps=8.0):
    data_loader = torchdrug.data.DataLoader(dataset, batch_size=1, shuffle=False)
    structures = []
    labels = []
    for protein in tqdm(data_loader):
        out = task.graph_construction_model(protein["graph"])
        sequence = out.to_sequence()[0]
        if len(sequence) == 0:
            continue
        coords = out.node_position
        labels.append(protein["targets"])

        torch_sequence = torch.LongTensor(
            [esm_alphabet.get_idx(res) for res in esm_alphabet.tokenize(sequence)]
        )
        graph_adj = radius_neighbors_graph(coords, radius=eps, mode="connectivity")
        edge_index = from_scipy_sparse_matrix(graph_adj)[0].long()
        torch_sequence = torch.cat(
            [
                torch.LongTensor([esm_alphabet.cls_idx]),
                torch_sequence,
                torch.LongTensor([esm_alphabet.eos_idx]),
            ]
        )
        edge_index = edge_index + 1  # shift for cls_idx
        edge_attr = None

        structures.append(
            Data(edge_index=edge_index, x=torch_sequence, edge_attr=edge_attr)
        )

    return structures, torch.cat(labels)


def add_label(data, y):
    data.y = y.view(1, -1).float()
    return data


def mask_cls_idx(data):
    data.idx_mask = torch.ones((len(data.x),), dtype=torch.bool)
    data.idx_mask[0] = data.idx_mask[-1] = False
    return data


class ProteinTaskTrainer(pl.LightningModule):
    def __init__(self, model, cfg, task):
        super().__init__()
        self.model = model

        self.cfg = cfg
        self.task = task
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.main_metric = cfg.metric
        self.best_val_score = -float("inf")
        self.main_val_metric = "val_" + self.main_metric

        self.best_weights = None

        self.output_dir = Path(cfg.logs.path)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch, include_seq=self.cfg.include_seq)
        y = batch.y
        loss = self.criterion(y_hat, y).mean(0).sum()

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(y))

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch, include_seq=self.cfg.include_seq)
        y = batch.y

        outputs = {"y_pred": y_hat, "y_true": y}
        self.validation_step_outputs.append(outputs)
        return outputs

    def evaluate_epoch_end(self, outputs, stage="val"):
        all_preds = torch.vstack([out["y_pred"] for out in outputs])
        all_true = torch.cat([out["y_true"] for out in outputs])
        all_true, all_preds = all_true.cpu(), all_preds.cpu()
        scores = self.task.evaluate(all_preds, all_true)
        scores = {
            "{}_".format(stage) + str(key): val.item() for key, val in scores.items()
        }
        return scores

    def on_validation_epoch_end(self):
        scores = self.evaluate_epoch_end(self.validation_step_outputs, "val")
        self.log_dict(scores)
        if scores[self.main_val_metric] >= self.best_val_score:
            self.best_val_score = scores[self.main_val_metric]
            self.best_weights = copy.deepcopy(self.model.state_dict())
        self.validation_step_outputs.clear()
        return scores

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch, include_seq=self.cfg.include_seq)
        y = batch.y
        outputs = {"y_pred": y_hat, "y_true": y}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        scores = self.evaluate_epoch_end(self.test_step_outputs, "test")
        scores["best_val_score"] = self.best_val_score
        # df = pd.DataFrame.from_dict(scores)
        # df.to_csv(self.output_dir / "results.csv")
        df = pd.DataFrame.from_dict(scores, orient="index")
        df.to_csv(self.output_dir / "results.csv", header=["value"], index_label="name")
        log.info(f"Test scores:\n{df}")
        self.test_step_outputs.clear()
        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.base_model.parameters(),
                    "lr": self.cfg.training.lr * 0.01,
                },
                {
                    "params": self.model.head_parameters(),
                    "weight_decay": self.cfg.training.weight_decay,
                },
            ],
            lr=self.cfg.training.lr,
            # weight_decay=self.cfg.training.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_f1_max"},
        }


@hydra.main(
    version_base="1.3",
    config_path=str(here() / "config"),
    config_name="pst_gearnet_finetune",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.include_seq and "so" not in cfg.model:
        cfg.model = f"{cfg.model}_so"

    pretrained_path = Path(cfg.pretrained) / f"{cfg.model}.pt"
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    model, model_cfg = PST.from_pretrained_url(
        cfg.model, pretrained_path,
    )

    task = core.Configurable.load_config_dict(
        edict(OmegaConf.to_container(cfg.task, resolve=True))
    )

    structure_path = (
        Path(cfg.dataset.path) / f"structures_{model_cfg.data.graph_eps}.pt"
    )
    if structure_path.exists():
        tmp = torch.load(structure_path)
        train_str, y_tr = tmp["train_str"], tmp["y_tr"]
        val_str, y_val = tmp["val_str"], tmp["y_val"]
        test_str, y_te = tmp["test_str"], tmp["y_te"]
        del tmp
    else:
        dataset = core.Configurable.load_config_dict(
            OmegaConf.to_container(cfg.dataset, resolve=True)
        )
        train_dset, val_dset, test_dset = dataset.split()

        train_str, y_tr = get_structures(train_dset, task, eps=model_cfg.data.graph_eps)
        val_str, y_val = get_structures(val_dset, task, eps=model_cfg.data.graph_eps)
        test_str, y_te = get_structures(test_dset, task, eps=model_cfg.data.graph_eps)
        torch.save(
            {
                "train_str": train_str,
                "val_str": val_str,
                "test_str": test_str,
                "y_tr": y_tr,
                "y_val": y_val,
                "y_te": y_te,
            },
            structure_path,
        )

    train_str = [
        mask_cls_idx(add_label(data, y_tr[i])) for i, data in enumerate(train_str)
    ]
    val_str = [
        mask_cls_idx(add_label(data, y_val[i])) for i, data in enumerate(val_str)
    ]
    test_str = [
        mask_cls_idx(add_label(data, y_te[i])) for i, data in enumerate(test_str)
    ]

    train_loader = DataLoader(
        train_str,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
    )
    val_loader = DataLoader(
        val_str,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.compute.num_workers,
    )
    test_loader = DataLoader(
        test_str,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.compute.num_workers,
    )

    cfg.num_iterations = len(train_loader) * cfg.training.epochs

    net = ProteinNet(model, y_tr.shape[1], out_head=cfg.out_head, aggr=cfg.aggr)
    model = ProteinTaskTrainer(net, cfg, task)

    trainer = pl.Trainer(
        limit_train_batches=5 if cfg.debug else None,
        limit_val_batches=5 if cfg.debug else None,
        max_epochs=cfg.training.epochs,
        precision=cfg.compute.precision,
        accelerator=cfg.compute.accelerator,
        devices="auto",
        strategy=cfg.compute.strategy,
        enable_checkpointing=False,
        default_root_dir=cfg.logs.path,
        logger=[pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")],
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    model.model.load_state_dict(model.best_weights)
    model.best_weights = None
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
