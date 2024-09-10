from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from rich.live import Live
from rich.table import Table
from sklearn.utils import shuffle

import wandb
from src.base_decoders import GRU, LSTM, RNN, BrainDecoder, SimpleMLP
from src.decoder_wrapper import DecoderWrapper
from src.metrics import retrieval_metrics, scores
from src.utils import console, device


def evaluate(
    df,
    decoder,
    negatives,
    top_k_accuracies,
    extra_metrics,
    log_run_metrics,
):
    decoder.eval()

    metrics = defaultdict(list)
    negatives = negatives.to(device)
    with torch.no_grad():
        for _, row in df.iterrows():
            X = row.X.to(device)
            Y = row.Y.to(device)
            run_metrics = {}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                X_proj = decoder.projector[row.dataset][row.subject](X)
                # Evaluate losses
                _, mixco = decoder.mixco_loss(X_proj, Y)
                run_metrics["mixco"] = mixco.item()
                Y_preds, symm_nce = decoder.symm_nce_loss(X_proj, Y)
                run_metrics["symm_nce"] = symm_nce.item()
                run_metrics["mse"] = decoder.mse_loss(X_proj, Y, Y_preds)[
                    1
                ].item()
            # Evaluate retrieval metrics
            run_metrics = retrieval_metrics(
                Y,
                Y_preds,
                negatives,
                return_ranks=True,
                top_k_accuracies=top_k_accuracies,
            )
            # if extra_metrics:
            #     run_metrics.update(scores(Y, Y_preds))
            #     relative_ranks = run_metrics["relative_ranks"]
            #     negatives_sorted = relative_ranks.argsort(dim=1)[:10]
            subject_id = f"{row.dataset}/{row.subject}"
            if log_run_metrics:
                run_id = f"{row.dataset}/{row.run}"
                names = ["", f"{subject_id}/", f"{run_id}/"]
            else:
                names = ["", f"{subject_id}/"]
            for key, value in run_metrics.items():
                for name in names:
                    metrics[name + key].append(value)
    metrics_agg = {}
    for key, value in metrics.items():
        if key.endswith("relative_ranks"):
            relative_ranks = torch.cat(value).cpu()
            relative_median_rank = torch.quantile(relative_ranks, q=0.5).item()
            metrics[key] = wandb.Histogram(relative_ranks, num_bins=100)
            metrics_agg[
                key.replace("relative_ranks", "relative_median_rank")
            ] = relative_median_rank
            metrics_agg[key.replace("relative_ranks", "size")] = (
                relative_ranks.shape[0]
            )
        else:
            metrics[key] = np.mean(value)
    metrics.update(metrics_agg)
    return metrics


def train(
    df_train,
    df_valid,
    df_test,
    decoder="brain_decoder",
    patience=20,
    monitor="valid/relative_median_rank",
    loss="mixco",
    weight_decay=1e-6,
    lr=1e-4,
    max_epochs=200,
    clip_grad_norm=1.0,
    batch_size=1,
    log_run_metrics=False,
    extra_metrics=True,
    **decoder_params,
):
    if decoder.lower() in ["rnn", "gru", "lstm"] and loss == "mixco":
        console.log(
            "[red]MixCo augmentation should not be used with time series."
        )

    Y_valid = df_valid.drop_duplicates(["dataset", "run"]).Y
    Y_valid = torch.cat(tuple(Y_valid)).to(device)

    out_dim = df_train.hidden_dim.iloc[0]
    wandb.config["out_dim"] = out_dim
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
    in_dims = df_train[["dataset", "subject", "n_voxels"]].drop_duplicates()
    in_dims = in_dims.set_index(["dataset", "subject"]).n_voxels
    in_dims = {
        level: in_dims.xs(level).to_dict() for level in in_dims.index.levels[0]
    }
    wandb.config["in_dims"] = in_dims
    decoder = DecoderWrapper(
        decoder=decoder,
        in_dims=in_dims,
        loss=loss,
        **decoder_params,
    ).to(device)
    decoder = torch.compile(decoder)

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

    top_k_accuracies = [1, 5, 10]
    best_monitor_metric, patience_counter = np.inf, 0
    torch.autograd.set_detect_anomaly(True)

    table = Table(
        "Epoch",
        "Monitor",
        "Patience",
        "Duration",
        title="Training loop",
    )
    with Live(table, console=console, vertical_overflow="visible"):
        for epoch in range(1, max_epochs + 1):
            df_train = shuffle(df_train)
            t = time()
            decoder.train()
            train_losses = []
            for i in range(0, len(df_train), batch_size):
                optimizer.zero_grad()
                for _, row in df_train.iloc[i : i + batch_size].iterrows():
                    X = row.X.to(device)
                    Y = row.Y.to(device)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16
                    ):
                        X_proj = decoder.projector[row.dataset][row.subject](X)
                        _, train_loss = decoder.loss(X_proj, Y)
                        train_loss /= batch_size

                    train_loss.backward()
                    train_losses.append(train_loss.item())

                if clip_grad_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(), clip_grad_norm
                    )
                optimizer.step()

            # Validation step
            val_metrics = evaluate(
                df=df_valid,
                decoder=decoder,
                negatives=Y_valid,
                top_k_accuracies=top_k_accuracies,
                extra_metrics=extra_metrics,
                log_run_metrics=log_run_metrics,
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
                best_epoch = epoch
                best_monitor_metric = monitor_metric
                patience_counter = 0
                monitor_metric = f"[green]{monitor_metric:.5g}"
                best_decoder_state_dict = deepcopy(decoder.state_dict())
                best_optimizer_state_dict = deepcopy(optimizer.state_dict())
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

    # Restore best model and optimizer
    output["best_epoch"] = best_epoch
    decoder.load_state_dict(best_decoder_state_dict)
    optimizer.load_state_dict(best_optimizer_state_dict)

    for split, df_split in [
        ("train/", df_train),
        ("test/", df_test),
    ]:
        Y_split = df_split.drop_duplicates(["dataset", "run"]).Y
        Y_split = torch.cat(tuple(Y_split))
        metrics = evaluate(
            df=df_split,
            decoder=decoder,
            negatives=Y_split,
            top_k_accuracies=top_k_accuracies,
            extra_metrics=extra_metrics,
            log_run_metrics=log_run_metrics,
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

    return output, decoder.cpu()
