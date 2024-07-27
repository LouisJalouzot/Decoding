from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from rich.live import Live
from rich.table import Table

import wandb
from src.decoders import GRU, LSTM, RNN, BrainDecoder, DecoderWrapper, SimpleMLP
from src.losses import (
    compute_mixco_symm_nce_loss,
    compute_mse_loss,
    compute_symm_nce_loss,
)
from src.metrics import retrieval_metrics
from src.utils import console, device


def combine_xarray_to_torch(Y):
    out = []
    for run in Y:
        data = (
            run.sel(
                tr=slice(run.n_trs.item()),
            )
            .transpose("tr", "hidden_dim")
            .data
        )
        out.append(torch.from_numpy(data))
    return torch.concat(out).to(device)


def evaluate(dl, decoder, negatives, top_k_accuracies, temperature):
    decoder.eval()
    dl.per_subject = True
    metrics = defaultdict(list)
    negatives = negatives.to(device)
    with torch.no_grad():
        for subject, X, Y in dl:
            X = decoder.project_subject(X)
            Y_preds = decoder(X)
            Y = combine_xarray_to_torch(Y)
            # Evaluate retrieval metrics
            for key, value in retrieval_metrics(
                Y,
                Y_preds,
                negatives,
                return_ranks=True,
                top_k_accuracies=top_k_accuracies,
            ).items():
                metrics[key].append(value)
                metrics[f"{subject}/{key}"].append(value)
            # Evaluate losses
            mse_loss = compute_mse_loss(X, Y, decoder)
            symm_nce_loss = compute_symm_nce_loss(X, Y, decoder, temperature)
            mixco_loss = compute_mixco_symm_nce_loss(X, Y, decoder, temperature)
            for name in ["", f"{subject}/"]:
                metrics[name + "mse"].append(mse_loss.item())
                metrics[name + "symm_nce"].append(symm_nce_loss.item())
                metrics[name + "mixco"].append(mixco_loss.item())
                metrics[name + "aug"].append(mixco_loss.item() - symm_nce_loss.item())
    other_metrics = {}
    for key, value in metrics.items():
        if key.endswith("relative_ranks"):
            relative_ranks = torch.cat(value).cpu()
            relative_median_rank = torch.quantile(relative_ranks, q=0.5).item()
            metrics[key] = wandb.Histogram(relative_ranks, num_bins=100)
            other_metrics[key.replace("relative_ranks", "relative_median_rank")] = (
                relative_median_rank
            )
            other_metrics[key.replace("relative_ranks", "size")] = relative_ranks.shape[
                0
            ]
        else:
            metrics[key] = np.mean(value)
    dl.per_subject = False
    metrics.update(other_metrics)
    return metrics


def train_brain_decoder(
    X_ds,
    Y_ds,
    train_runs,
    valid_runs,
    test_runs,
    decoder="brain_decoder",
    patience=5,
    monitor="valid/relative_median_rank",
    loss="mixco",
    weight_decay=1e-6,
    lr=1e-4,
    max_epochs=200,
    batch_size=4,
    temperature=0.01,
    checkpoints_path=None,
    **decoder_params,
):
    if decoder.lower() in ["rnn", "gru", "lstm"] and loss == "mixco":
        console.log("[red]MixCo augmentation should not be used with time series.")
    train_dl = CustomDataloader(
        X_ds[X_ds.run.isin(train_runs)],
        Y_ds.sel(run=train_runs),
        batch_size,
        shuffle=True,
    )
    Y_valid = combine_xarray_to_torch(Y_ds.sel(run=valid_runs))
    valid_dl = CustomDataloader(
        X_ds[X_ds.run.isin(valid_runs)],
        Y_ds.sel(run=valid_runs),
        batch_size,
        per_subject=True,
    )
    test_dl = CustomDataloader(
        X_ds[X_ds.run.isin(test_runs)],
        Y_ds.sel(run=test_runs),
        batch_size,
        per_subject=True,
    )

    out_dim = Y_ds.hidden_dim.size
    if decoder.lower() == "brain_decoder":
        decoder = BrainDecoder(out_dim=out_dim, **decoder_params)
    elif decoder.lower() == "rnn":
        decoder = RNN(out_dim=out_dim, **decoder_params)
    elif decoder.lower() == "gru":
        decoder = GRU(out_dim=out_dim, **decoder_params)
    elif decoder.lower() == "lstm":
        decoder = LSTM(out_dim=out_dim, **decoder_params)
    elif decoder.lower() == "simple_mlp":
        decoder = SimpleMLP(out_dim=out_dim, **decoder_params)
    else:
        raise ValueError(f"Unsupported decoder {decoder}.")
    decoder = DecoderWrapper(
        decoder=decoder,
        in_dims={
            e.subject.item(): e.item() for e in X_ds.n_voxels.groupby("subject").last()
        },
        **decoder_params,
    ).to(device)

    n_params = sum([p.numel() for p in decoder.parameters()])
    console.log(f"Decoder has {n_params:.3g} parameters.")
    wandb.config["n_params"] = n_params

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
    losses = ["mixco", "symm_nce", "mse"]
    if loss not in losses:
        raise ValueError(f"Unsupported loss {loss}. Choose one of {losses}.")

    top_k_accuracies = [1, 5, 10, 50]
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
            train_losses = []
            for X, Y in train_dl:
                optimizer.zero_grad()
                X = decoder.project_subject(X)
                Y = combine_xarray_to_torch(Y)

                if loss == "mse":
                    # Evaluate MSE loss
                    train_loss = compute_mse_loss(X, Y, decoder)

                if loss == "symm_nce":
                    # Evaluate symmetrical NCE loss
                    train_loss = compute_symm_nce_loss(X, Y, decoder, temperature)

                if loss == "mixco":
                    # Evaluate mixco loss and back-propagate on it
                    train_loss = compute_mixco_symm_nce_loss(X, Y, decoder, temperature)

                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.item())

            # Validation step
            val_metrics = evaluate(
                valid_dl, decoder, Y_valid, top_k_accuracies, temperature
            )

            # Log metrics
            output = {
                "train/" + loss: np.mean(train_losses),
                **{"valid/" + key: value for key, value in val_metrics.items()},
            }
            if epoch == 1:
                for key in output:
                    if not key.endswith("ranks") and key.count("/") < 2:
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

            table.add_row(
                f"{epoch} / {max_epochs}",
                monitor_metric,
                str(patience - patience_counter),
                f"{time() - t:.3g}s",
                *[
                    f"{v:.3g}"
                    for k, v in output.items()
                    if not k.endswith("ranks") and k.count("/") < 2
                ],
            )

            if patience_counter >= patience:
                console.log(
                    f"Early stopping at epoch {epoch} as [bold green]{monitor}[/] did not improve for {patience} epochs."
                )
                break

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

    train_dl.per_subject = True
    for split, dl, Y_split in [
        ("train/", train_dl, Y_ds.sel(run=train_runs)),
        ("test/", test_dl, Y_ds.sel(run=test_runs)),
    ]:
        Y_split_tensor = []
        for run in Y_split:
            data = run.sel(
                tr=slice(run.n_trs.item()),
            ).transpose("tr", "hidden_dim")
            Y_split_tensor.append(torch.from_numpy(data))
        Y_split = torch.concat(Y_split_tensor)
        metrics = evaluate(
            dl,
            decoder,
            Y_split,
            top_k_accuracies,
            temperature,
        )
        for key, value in metrics.items():
            output[split + key] = value
    wandb.log(output)

    for key, value in output.items():
        if isinstance(value, wandb.Histogram):
            output[key] = {"bins": value.bins, "histogram": value.histogram}

    for split in ["Train", "Valid", "Test"]:
        console.log(
            f"{split} relative median rank {output[f'{split.lower()}/relative_median_rank']:.3g} (size {output[f'{split.lower()}/size']})"
        )

    return output
