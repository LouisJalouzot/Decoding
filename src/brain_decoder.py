from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from rich.live import Live
from rich.table import Table
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pack_sequence

import wandb
from src.decoders import GRU, LSTM, RNN, BrainDecoder, DecoderWrapper, SimpleMLP
from src.losses import (
    compute_mixco_symm_nce_loss,
    compute_mse_loss,
    compute_symm_nce_loss,
)
from src.metrics import retrieval_metrics
from src.utils import console, device


def evaluate(df, decoder, negatives, top_percent_accuracies, temperature):
    decoder.eval()
    metrics = defaultdict(list)
    negatives = negatives.to(device)
    with torch.no_grad():
        for _, row in df.iterrows():
            subject = row.subject
            X = decoder.projector[subject](row.X.to(device))
            X = pack_sequence([X], enforce_sorted=False)
            Y_preds = decoder(X)
            Y = row.Y.to(device)
            # Evaluate retrieval metrics
            for key, value in retrieval_metrics(
                Y,
                Y_preds,
                negatives,
                return_ranks=True,
                top_percent_accuracies=top_percent_accuracies,
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
    metrics.update(other_metrics)
    return metrics


def train_brain_decoder(
    df_train,
    df_valid,
    df_test,
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
    plot_gradient=False,
    **decoder_params,
):
    if decoder.lower() in ["rnn", "gru", "lstm"] and loss == "mixco":
        console.log("[red]MixCo augmentation should not be used with time series.")

    Y_valid = df_valid.drop_duplicates(["dataset", "run"]).Y
    Y_valid = torch.cat(tuple(Y_valid)).to(device)

    out_dim = df_train.hidden_dim.iloc[0]
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
        in_dims=df_train[["subject", "n_voxels"]].drop_duplicates(),
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

    top_percent_accuracies = [1, 5, 10]
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
            df_train = shuffle(df_train)
            t = time()
            decoder.train()
            train_losses = []
            for i in range(0, len(df_train), batch_size):
                optimizer.zero_grad()
                X = []
                Y = []
                for _, row in df_train.iloc[i : i + batch_size].iterrows():
                    X.append(decoder.projector[row.subject](row.X.to(device)))
                    Y.append(row.Y)
                X = pack_sequence(X, enforce_sorted=False)
                Y = torch.cat(Y).to(device)

                if loss == "mse":
                    # Evaluate MSE loss
                    train_loss = compute_mse_loss(X, Y, decoder)

                if loss == "symm_nce":
                    # Evaluate symmetrical NCE loss
                    train_loss = compute_symm_nce_loss(X, Y, decoder, temperature)

                if loss == "mixco":
                    # Evaluate mixco loss and back-propagate on it
                    train_loss = compute_mixco_symm_nce_loss(X, Y, decoder, temperature)

                if plot_gradient:
                    from torchviz import make_dot

                    graph = make_dot(
                        train_loss,
                        params=dict(decoder.named_parameters()),
                    )
                    graph.render(filename="gradient_graph", format="png")

                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.item())

            # Validation step
            val_metrics = evaluate(
                df_valid, decoder, Y_valid, top_percent_accuracies, temperature
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

    for split, df_split in [
        ("train/", df_train),
        ("test/", df_test),
    ]:
        Y_split = df_split.drop_duplicates(["dataset", "run"]).Y
        Y_split = torch.cat(tuple(Y_split))
        metrics = evaluate(
            df_split,
            decoder,
            Y_split,
            top_percent_accuracies,
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
