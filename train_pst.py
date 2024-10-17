import logging
from functools import partial

import hydra
import proteinshake.datasets as ps_dataset
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from proteinshake import datasets
from proteinshake.transforms import Compose
from pyprojroot import here
from torch_geometric.loader import DataLoader

from pst.dataset import CustomGraphDataset
from pst.esm2 import PST
from pst.trainer import BertTrainer
from pst.transforms import (
    MaskNode,
    PretrainingAttr,
    Proteinshake2ESM,
    RandomCrop,
    RandomizeEdges,
    SequenceEdges,
    CompleteEdges,
)
from pst.utils import get_graph_from_ps_protein

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')

def get_loggers(cfg):
    loggers = [
    pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"),
    pl.loggers.TensorBoardLogger(cfg.logs.path, name="tb_logs"),
    ]
    if cfg.logs.wandb.enable:
        loggers.append(
            pl.loggers.WandbLogger(
            name=cfg.logs.wandb.name,
            tags=cfg.logs.wandb.tags,
            entity=cfg.logs.wandb.entity,
            project=cfg.logs.wandb.project,
            save_dir=cfg.logs.wandb.save_dir,
            )
        )
    return loggers

@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="pst_pretrain"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)
    if cfg.model.use_edge_attr:
        featurizer_fn = partial(
            get_graph_from_ps_protein, use_rbfs=True, eps=cfg.data.graph_eps
        )
        transforms = [
            RandomCrop(cfg.data.crop_len),
            MaskNode(mask_rate=cfg.data.mask_rate),
        ]
     
        dataset = CustomGraphDataset(
            root=cfg.data.datapath,
            dataset=ps_dataset.AlphaFoldDataset(
                root=cfg.data.datapath, organism=cfg.data.organism
            ),
            pre_transform=featurizer_fn,
            transform=Compose(
                transforms=transforms,
            ),
            n_jobs=cfg.compute.n_jobs,
        )
        cfg.model.edge_dim = 16
    else:
        dataset = datasets.AlphaFoldDataset(
            root=cfg.data.datapath, organism=cfg.data.organism
        )
        transforms = [
            PretrainingAttr(),
            Proteinshake2ESM(),
            RandomCrop(cfg.data.crop_len),
            MaskNode(mask_rate=cfg.data.mask_rate),
        ]
        
        if cfg.data.edge_perturb == "random":
            transforms.append(RandomizeEdges())
        elif cfg.data.edge_perturb == "sequence":
            transforms.append(SequenceEdges())
        elif cfg.data.edge_perturb == "complete":
            transforms.append(CompleteEdges())
        elif cfg.data.edge_perturb is None:
            pass
        else:
            raise ValueError("Invalid value for cfg.data.edge_perturb")

        dataset = dataset.to_graph(eps=cfg.data.graph_eps).pyg(
            transform=Compose(
                transforms=transforms,
            )
        )

    log.info(f"Total number of samples: {len(dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    net = PST.from_model_name(
        cfg.model.name,
        k_hop=cfg.model.k_hop,
        gnn_type=cfg.model.gnn_type,
        edge_dim=cfg.model.edge_dim,
    )

    net.train_struct_only(cfg.model.train_struct_only)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    iterations = len(train_loader) // num_devices
    log.info(f"Number of devices: {num_devices}")

    model = BertTrainer(
        net,
        cfg,
        iterations,
    )

    trainer = pl.Trainer(
        limit_train_batches=5 if cfg.debug else None,
        max_epochs=cfg.training.epochs,
        precision=cfg.compute.precision,
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.devices,
        strategy=cfg.compute.strategy,
        enable_checkpointing=True,
        default_root_dir=cfg.logs.path,
        logger=get_loggers(cfg),
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.RichProgressBar(),
        ],
    )

    trainer.fit(model, train_loader)

    net.save(f"{cfg.logs.path}/model.pt", cfg)


if __name__ == "__main__":
    main()
