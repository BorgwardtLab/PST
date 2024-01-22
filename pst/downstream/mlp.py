import copy
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from timeit import default_timer as timer


class MLPTrainer(pl.LightningModule):
    def __init__(self, embed_dim, num_class, cfg, task, head='mlp'):
        super().__init__()
        if head == "linear":
            self.model = nn.Linear(embed_dim, num_class)
        else:
            self.model = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(embed_dim // 4, num_class)
            )

        self.cfg = cfg
        self.task = task
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.main_metric = cfg.metric
        self.best_val_score = -float('inf')
        self.main_val_metric = 'val_' + self.main_metric

        self.best_weights = None

        self.output_dir = Path(cfg.logs.path)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y).mean(0).sum()

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=len(y)
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        outputs = {'y_pred': y_hat, 'y_true': y}
        self.validation_step_outputs.append(outputs)
        return outputs

    def evaluate_epoch_end(self, outputs, stage='val'):
        all_preds = torch.vstack([out['y_pred'] for out in outputs])
        all_true = torch.cat([out['y_true'] for out in outputs])
        all_true, all_preds = all_true.cpu(), all_preds.cpu()
        if self.task is not None:
            scores = self.task.evaluate(all_preds, all_true)
        else:
            all_preds = all_preds.argmax(-1)
            scores = torch.mean((all_preds == all_true).float())
            scores = {'accuracy': scores}
        scores = {'{}_'.format(stage) + str(key): val.item() for key, val in scores.items()}
        return scores

    def on_validation_epoch_end(self):
        scores = self.evaluate_epoch_end(self.validation_step_outputs, 'val')
        self.log_dict(scores)
        if scores[self.main_val_metric] >= self.best_val_score:
            self.best_val_score = scores[self.main_val_metric]
            self.best_weights = copy.deepcopy(self.model.state_dict())
        self.validation_step_outputs.clear()
        return scores

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        outputs = {'y_pred': y_hat, 'y_true': y}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        scores = self.evaluate_epoch_end(self.test_step_outputs, 'test')
        scores['best_val_score'] = self.best_val_score
        df = pd.DataFrame.from_dict(scores, orient='index')
        df.to_csv(self.output_dir / "results.csv",
                  header=['value'], index_label='name')
        print(f"Test scores:\n{df}")
        self.test_step_outputs.clear()
        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-07,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_f1_max"},
        }

def train_and_eval_mlp(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        cfg, task, batch_size=32, epochs=100,
    ):
    pl.seed_everything(cfg.seed, workers=True)
    train_dset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True
    )
    val_dset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=False
    )
    test_dset = TensorDataset(X_te, y_te)
    test_loader = DataLoader(
        test_dset, batch_size=batch_size, shuffle=False
    )

    model = MLPTrainer(X_tr.shape[1], y_tr.shape[1], cfg, task)

    trainer = pl.Trainer(
        max_epochs=epochs,
        precision='16-mixed',
        accelerator='auto',
        devices="auto",
        strategy='auto',
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


class Linear(nn.Linear):
    def forward(self, input):
        bias = self.bias
        if bias is not None and hasattr(self, 'scale_bias') and self.scale_bias is not None:
            bias = self.scale_bias * bias

        if bias is not None and hasattr(self, 'scale_bias') and self.scale_bias is not None:
            bias = self.scale_bias * bias
        out = torch.nn.functional.linear(torch.nn.functional.dropout(input, 0.5, training=self.training), self.weight, bias)
        return out

    def fit(self, Xtr, ytr, criterion, reg=0.0, epochs=100, optimizer=None, use_cuda=False):
        if optimizer is None:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, history_size=10)
        if self.bias is not None:
            scale_bias = (Xtr ** 2).mean(-1).sqrt().mean().item()
            self.scale_bias = scale_bias
        self.train()
        if use_cuda:
            self.cuda()
            Xtr = Xtr.cuda()
            ytr = ytr.cuda()
        def closure():
            optimizer.zero_grad()
            output = self(Xtr)
            loss = criterion(output, ytr)
            loss = loss + 0.5 * reg * self.weight.pow(2).sum()
            loss.backward()
            return loss

        for epoch in range(epochs):
            optimizer.step(closure)
        if self.bias is not None:
            self.bias.data.mul_(self.scale_bias)
        self.scale_bias = None

    @torch.no_grad()
    def score(self, X, y):
        self.eval()
        scores = self(X)
        scores = scores.argmax(-1)
        scores = scores.cpu()
        return torch.mean((scores == y).float()).item()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def train_and_eval_linear(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        num_class, stratified_indices, use_cuda=False
    ):
    embed_dim = X_tr.shape[1]
    search_grid = 2. ** np.arange(3, 18)
    search_grid = 1. / search_grid
    best_score = -np.inf
    clf = Linear(embed_dim, num_class)#, bias=True)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    if X_tr.shape[1] > 20000:
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
        epochs = 800
    else:
        optimizer = torch.optim.LBFGS(
                clf.parameters(), lr=1.0, max_eval=10, history_size=10, tolerance_grad=1e-05, tolerance_change=1e-05)
        epochs = 100
    torch.cuda.empty_cache()
    print("Start cross validation")
    for alpha in search_grid:
        tic = timer()
        clf.fit(X_tr, y_tr, criterion, reg=alpha, epochs=epochs, optimizer=optimizer, use_cuda=use_cuda)
        toc = timer()
        scores = []
        for X, y in zip(X_val, y_val):
            if use_cuda:
                X = X.cuda()
            score = clf.score(X, y)
            scores.append(score)

            score = clf.score(X, y)
            scores.append(score)
        score = np.mean(scores)
        print("CV alpha={}, acc={:.2f}, ts={:.2f}s".format(alpha, score * 100., toc - tic))
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_weight = copy.deepcopy(clf.state_dict())

    clf.load_state_dict(best_weight)

    print("Finished, elapsed time: {:.2f}s".format(toc - tic))

    if use_cuda:
        X_te = X_te.cuda()
    with torch.no_grad():
        y_pred = clf(X_te).cpu()

    scores = accuracy(y_pred, y_te, (1, 5, 10))
    print(scores)

    stratified_scores = {}
    for key, idx in stratified_indices.items():
        stratified_scores[key] = accuracy(y_pred[idx], y_te[idx], (1, 5, 10))
    print(stratified_scores)
    return best_score, scores, stratified_scores
