import pytorch_lightning as pl
import torch
from torch import nn

from .utils import (
    get_cosine_schedule_with_warmup,
    get_inverse_sqrt_schedule_with_warmup,
)


class BertTrainer(pl.LightningModule):
    def __init__(self, model, cfg, iterations):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.iterations = iterations
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        y_hat, y = self.model.mask_forward(batch)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.detach().argmax(dim=-1) == y).float().mean().item()
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model.mask_predict(batch)
        y_hat = torch.log_softmax(y_hat, dim=-1)
        y_pred = y_hat.gather(-1, batch.mt_indices) - y_hat.gather(-1, batch.wt_indices)
        y_true = batch.y
        outputs = {"y_pred": y_pred, "y_true": y_true}
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        import scipy

        y_pred = torch.vstack(
            [out["y_pred"] for out in self.validation_step_outputs]
        ).view(-1)
        y_true = torch.cat(
            [out["y_true"] for out in self.validation_step_outputs]
        ).view(-1)
        rho = scipy.stats.spearmanr(y_true.cpu().numpy(), y_pred.cpu().double().numpy())
        self.log_dict({"val_rho": rho.correlation}, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        if self.cfg.training.schedule == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                self.cfg.training.warmup * self.iterations,
                self.cfg.training.epochs * self.iterations,
            )
        elif self.cfg.training.schedule == "inv_sqrt":
            lr_scheduler = get_inverse_sqrt_schedule_with_warmup(
                optimizer,
                self.cfg.training.warmup * self.iterations,
                self.cfg.training.epochs * self.iterations,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
