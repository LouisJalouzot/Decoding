from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.live import Live
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

import wandb
from src.metrics import retrieval_metrics, scores
from src.utils import _get_progress, console, device


def mixco_sample_augmentation(samples, beta=0.15, s_thresh=0.5):
    """Augment samples with MixCo augmentation.

    Parameters
    ----------
    samples : torch.Tensor
        Samples to augment.
    beta : float, optional
        Beta parameter for the Beta distribution, by default 0.15
    s_thresh : float, optional
        Proportion of samples which should be affected by MixCo, by default 0.5

    Returns
    -------
    samples : torch.Tensor
        Augmented samples.
    perm : torch.Tensor
        Permutation of the samples.
    betas : torch.Tensor
        Betas for the MixCo augmentation.
    select : torch.Tensor
        Samples affected by MixCo augmentation
    """
    # Randomly select samples to augment
    select = (torch.rand(samples.shape[0]) <= s_thresh).to(samples.device)

    # Randomly select samples used for augmentation
    perm = torch.randperm(samples.shape[0])
    samples_shuffle = samples[perm].to(samples.device, dtype=samples.dtype)

    # Sample MixCo coefficients from a Beta distribution
    betas = (
        torch.distributions.Beta(beta, beta)
        .sample([samples.shape[0]])
        .to(samples.device, dtype=samples.dtype)
    )
    betas[~select] = 1
    betas_shape = [-1] + [1] * (len(samples.shape) - 1)

    # Augment samples
    samples[select] = samples[select] * betas[select].reshape(
        *betas_shape
    ) + samples_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)

    return samples, perm, betas, select


def mixco_symmetrical_nce_loss(
    preds,
    targs,
    temperature=0.1,
    perm=None,
    betas=None,
    select=None,
    bidirectional=True,
):
    """Compute symmetical NCE loss with MixCo augmentation.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted latent features.
    targs : torch.Tensor
        Target latent features.
    temperature : float, optional
        Temperature for the softmax, by default 0.1
    perm : torch.Tensor, optional
        Permutation of the samples, by default None
    betas : torch.Tensor, optional
        Betas for the MixCo augmentation, by default None
    select : torch.Tensor, optional
        Selection of the samples, by default None
    bidirectional : bool, optional
        Whether to compute the loss in both directions, by default True

    Returns
    -------
    torch.Tensor
        Symmetrical NCE loss.
    """
    brain_clip = (preds @ targs.T) / temperature

    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2) / 2
        return loss
    else:
        loss = F.cross_entropy(
            brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device)
        )
        if bidirectional:
            loss2 = F.cross_entropy(
                brain_clip.T,
                torch.arange(brain_clip.shape[0]).to(brain_clip.device),
            )
            loss = (loss + loss2) / 2

        return loss


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=1024, dropout=0.7):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_size_backbone=512,
        hidden_size_projector=512,
        dropout=0.7,
        n_res_blocks=2,
        n_proj_blocks=1,
        norm_type="ln",
        activation_layer_first=False,
    ):
        super().__init__()

        self.n_res_blocks = n_res_blocks

        norm_backbone = (
            partial(nn.BatchNorm1d, num_features=hidden_size_backbone)
            if norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=hidden_size_backbone)
        )
        activation_backbone = (
            partial(nn.ReLU, inplace=True) if norm_type == "bn" else nn.GELU
        )
        activation_and_norm = (
            (activation_backbone, norm_backbone)
            if activation_layer_first
            else (norm_backbone, activation_backbone)
        )

        # First linear
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, hidden_size_backbone),
            *[item() for item in activation_and_norm],
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size_backbone, hidden_size_backbone),
                    *[item() for item in activation_and_norm],
                    nn.Dropout(dropout),
                )
                for _ in range(n_res_blocks)
            ]
        )

        # Second linear
        self.lin1 = nn.Linear(hidden_size_backbone, hidden_size_projector, bias=True)

        # Projector
        assert n_proj_blocks >= 0
        projector_layers = []
        for _ in range(n_proj_blocks):
            projector_layers.extend(
                [
                    nn.LayerNorm(hidden_size_projector),
                    nn.GELU(),
                    nn.Linear(hidden_size_projector, hidden_size_projector),
                ]
            )
        projector_layers.extend(
            [
                nn.LayerNorm(hidden_size_projector),
                nn.GELU(),
                nn.Linear(hidden_size_projector, out_dim),
            ]
        )
        self.projector = nn.Sequential(*projector_layers)

    def forward(self, x):
        x = self.lin0(x)

        residual = x
        for res_block in range(self.n_res_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)

        x = self.lin1(x)

        return self.projector(x)


def evaluate(dl, decoder, negatives, top_k_accuracies, temperature):
    decoder.eval()
    metrics = defaultdict(list)
    negatives = negatives.to(device)
    for X, Y in dl:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                X = X.to(device)
                Y = Y.to(device)
                Y_preds = decoder(X)
                # Evaluate retrieval metrics
                for key, value in retrieval_metrics(
                    Y,
                    Y_preds,
                    negatives,
                    return_ranks=True,
                    top_k_accuracies=top_k_accuracies,
                ).items():
                    metrics[key].append(value)
                # Evaluate symmetrical NCE loss
                Y_preds_norm = nn.functional.normalize(Y_preds, dim=-1)
                Y_true_norm = nn.functional.normalize(Y, dim=-1)
                symm_nce_loss = mixco_symmetrical_nce_loss(
                    Y_preds_norm,
                    Y_true_norm,
                    temperature=temperature,
                )
                metrics["symm_nce_loss"].append(symm_nce_loss.item())
                # Evaluate mixco symmetrical NCE loss
                (
                    X_mixco,
                    perm,
                    betas,
                    select,
                ) = mixco_sample_augmentation(X)
                Y_preds = decoder(X_mixco)
                Y_preds_norm = nn.functional.normalize(Y_preds, dim=-1)
                Y_true_norm = nn.functional.normalize(Y, dim=-1)
                mixco_loss = mixco_symmetrical_nce_loss(
                    Y_preds_norm,
                    Y_true_norm,
                    temperature=temperature,
                    perm=perm,
                    betas=betas,
                    select=select,
                )
                metrics["mixco_loss"].append(mixco_loss.item())
    for key, value in metrics.items():
        if len(value) > 0 and isinstance(value[0], torch.Tensor):
            metrics[key] = wandb.Histogram(torch.cat(value).cpu(), num_bins=100)
        else:
            metrics[key] = np.mean(value)
    return metrics


def train_brain_decoder(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    X_test,
    Y_test,
    patience=5,
    decoder="brain_decoder",
    monitor="val/relative_median_rank",
    seed=0,
    setup_config={},
    verbose=True,
    weight_decay=1e-6,
    lr=1e-4,
    max_epochs=100,
    batch_size=128,
    temperature=0.01,
    checkpoints_path=None,
    **decoder_params,
):
    # Monitor: name of the metric to monitor for early stopping, lower is better
    torch.manual_seed(seed)
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_valid = torch.from_numpy(X_valid)
    Y_valid = torch.from_numpy(Y_valid).to(device)
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)
    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    config = {
        "decoder": decoder,
        "patience": patience,
        "seed": seed,
        "weight_decay": weight_decay,
        "lr": lr,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "temperature": temperature,
    }
    config.update(decoder_params)
    name = "_".join([f"{key}={value}" for key, value in config.items()])
    config.update(setup_config)
    wandb_run = wandb.init(
        name=name,
        config=config,
        id=f"{hash(frozenset(config.items())):02x}",
        save_code=True,
    )

    if decoder.lower() == "brain_decoder":
        decoder_class = BrainDecoder
    elif decoder.lower() == "simple_mlp":
        decoder_class = SimpleMLP
    decoder = decoder_class(
        in_dim=in_dim,
        out_dim=out_dim,
        **decoder_params,
    ).to(device)
    if verbose:
        console.log(
            f"Decoder has {sum([p.numel() for p in decoder.parameters()]):.3g} parameters."
        )
    no_decay = ["bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in decoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in decoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        opt_grouped_parameters,
        lr=lr,
    )

    train_dl = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_dl = DataLoader(TensorDataset(X_valid, Y_valid), batch_size=batch_size)
    test_dl = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

    # top_k_accuracies = [1, 5, 10, int(len(Y_valid) / 10), int(len(Y_valid) / 5)]
    top_k_accuracies = []
    best_monitor_metric, patience_counter = np.inf, 0
    torch.autograd.set_detect_anomaly(True)

    table = Table(
        "Epoch",
        "Monitor",
        "Patience",
        "Duration",
        title="Training loop",
    )
    with Live(table, console=console):
        for epoch in range(1, max_epochs + 1):
            t = time()
            decoder.train()
            train_metrics = defaultdict(list)
            for X, Y in train_dl:
                with torch.cuda.amp.autocast():
                    X = X.to(device)
                    Y = Y.to(device)

                    # Evaluate mixco loss and back-propagate on it
                    (
                        X_mixco,
                        perm,
                        betas,
                        select,
                    ) = mixco_sample_augmentation(X)
                    optimizer.zero_grad()
                    Y_preds = decoder(X_mixco)
                    Y_preds_norm = nn.functional.normalize(Y_preds, dim=-1)
                    Y_true_norm = nn.functional.normalize(Y, dim=-1)
                    mixco_loss = mixco_symmetrical_nce_loss(
                        Y_preds_norm,
                        Y_true_norm,
                        temperature=temperature,
                        perm=perm,
                        betas=betas,
                        select=select,
                    )
                    mixco_loss.backward()
                    optimizer.step()

                    # Evaluate symmetrical NCE loss
                    with torch.no_grad():
                        Y_preds = decoder(X)
                        Y_preds_norm = nn.functional.normalize(Y_preds, dim=-1)
                        Y_true_norm = nn.functional.normalize(Y, dim=-1)
                        symm_nce_loss = mixco_symmetrical_nce_loss(
                            Y_preds_norm,
                            Y_true_norm,
                            temperature=temperature,
                        )

                train_metrics["symm_nce_loss"].append(symm_nce_loss.item())
                train_metrics["mixco_loss"].append(mixco_loss.item())

            # Validation step
            val_metrics = evaluate(
                val_dl, decoder, Y_valid, top_k_accuracies, temperature
            )

            # Log metrics
            output = {
                **{
                    "train/" + key: np.mean(values)
                    for key, values in train_metrics.items()
                },
                **{"val/" + key: value for key, value in val_metrics.items()},
            }
            if epoch == 1:
                for key in output:
                    if not key.endswith("ranks"):
                        table.add_column(key)
                for col in table.columns:
                    col.overflow = "fold"
            wandb.log(output)

            # Early stopping
            monitor_metric = np.mean(output[monitor])
            if monitor_metric < best_monitor_metric:
                best_monitor_metric = monitor_metric
                patience_counter = 0
                monitor_metric = f"[green]{monitor_metric:.5g}"
                best_decoder_state_dict = deepcopy(decoder.state_dict())
                # For saving
                if checkpoints_path is not None:
                    best_epoch = epoch
                    best_optimizer_state_dict = optimizer.state_dict().copy()
            else:
                patience_counter += 1
                monitor_metric = f"[red]{monitor_metric:.5g}"
                if patience_counter >= patience:
                    console.log(
                        f"Early stopping at epoch {epoch} as {monitor} did not improve for {patience} epochs."
                    )
                    break
            table.add_row(
                f"{epoch} / {max_epochs}",
                monitor_metric,
                str(patience - patience_counter),
                f"{time() - t:.3g}s",
                *[f"{v:.3g}" for k, v in output.items() if not k.endswith("ranks")],
            )

    # Restore best model
    decoder.load_state_dict(best_decoder_state_dict)
    # Saving best model
    if checkpoints_path is not None:
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_decoder_state_dict,
                "optimizer_state_dict": best_optimizer_state_dict,
            },
            Path(checkpoints_path) / f"checkpoint_{epoch:03d}.pt",
        )

    for split, dl, negatives in [
        ("train/", train_dl, Y_train),
        ("test/", test_dl, Y_test),
    ]:
        metrics = evaluate(
            dl,
            decoder,
            negatives,
            top_k_accuracies,
            temperature,
        )
        for key, value in metrics.items():
            output[split + key] = value
    wandb.log(output)
    wandb.finish()
    wandb_run.finish()

    for key, value in output.items():
        if isinstance(value, wandb.Histogram):
            output[key] = {"bins": value.bins, "histogram": value.histogram}

    return output
