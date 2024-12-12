import logging
from copy import deepcopy
from numbers import Number
from time import time

import numpy as np
import pandas as pd
import torch
from rich.live import Live
from rich.table import Table
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from src import base_decoders
from src.decoder_wrapper import DecoderWrapper
from src.evaluate import evaluate
from src.utils import compute_gradient_norm, console, device

logger = logging.getLogger(__name__)


def init_optimizer(
    decoder,
    prefix="Training",
):
    opt_grouped_parameters = []

    n_params = 0

    for module in [decoder.projector, decoder.decoder]:
        wd_params, no_wd_params = [], []
        for p in module.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    wd_params.append(p)
                if len(p.shape) == 1:
                    no_wd_params.append(p)
                if module.lr > 0:
                    n_params += p.numel()
        for params, wd in [wd_params, module.weight_decay], [no_wd_params, 0]:
            params = {
                "params": params,
                "weight_decay": wd,
                "lr": module.lr,
            }
            opt_grouped_parameters.append(params)

    if wandb.run is not None:
        if prefix == "Training":
            wandb.config["n_params"] = n_params
        else:
            wandb.config["n_params_ft"] = n_params
    logger.info(f"{prefix} {n_params:.3g} parameters")

    return torch.optim.AdamW(opt_grouped_parameters)


def switch_to_finetune_mode(module):
    # Updates parameters with their "_ft" counterpart
    for attr_name in dir(module):
        if attr_name.endswith("_ft"):
            base_attr = attr_name.replace("_ft", "")
            if hasattr(module, base_attr):
                setattr(module, attr_name, getattr(module, base_attr))


def train_loop(
    decoder,
    optimizer,
    df_train,
    df_valid,
    max_epochs,
    batch_size,
    monitor,
    patience,
    scheduler_patience,
    top_k_accuracies,
    nlp_distances,
    n_candidates,
    metrics_prefix,
    **kwargs,
):
    init_train_index = df_train.index

    negatives = df_valid.drop_duplicates(["dataset", "run"]).Y
    negatives = torch.cat(tuple(negatives)).to(device)

    best_monitor_metric = np.inf
    patience_counter = 0
    best_epoch = -1
    output = {}

    table = Table(
        "Epoch",
        "Monitor",
        "Patience",
        "Duration",
        title="Training loop",
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=0.5
    )
    with Live(table, console=console, vertical_overflow="visible"):
        for epoch in range(1, max_epochs + 1):
            df_train = shuffle(df_train)
            t = time()
            decoder.train()
            train_losses, grad_norms = [], []
            for i in range(0, len(df_train), batch_size):
                optimizer.zero_grad()
                for _, row in df_train.iloc[i : i + batch_size].iterrows():
                    X = row.X.to(device)
                    Y = row.Y.to(device)
                    X_proj = decoder.projector[row.dataset][row.subject](X)
                    train_loss = decoder(X_proj, Y)
                    train_loss /= batch_size

                    train_loss.backward()
                    train_losses.append(train_loss.item())

                grad_norms.append(compute_gradient_norm(decoder))
                optimizer.step()

            # Validation step
            val_metrics = evaluate(
                df=df_valid,
                decoder=decoder,
                negatives=negatives,
                top_k_accuracies=top_k_accuracies,
                nlp_distances=nlp_distances.get("valid", None),
                n_candidates=n_candidates,
            )

            # Log metrics
            output = {
                "train/" + decoder.loss: np.mean(train_losses),
                "grad_norm": np.mean(grad_norms),
                "lr": scheduler.get_last_lr()[0],
                **{"valid/" + key: value for key, value in val_metrics.items()},
            }
            if epoch == 1:
                for k, v in output.items():
                    if isinstance(v, Number) and len(k.split("/")) < 3:
                        table.add_column(k)
                for col in table.columns:
                    col.overflow = "fold"
            if wandb.run is not None:
                wandb.log({metrics_prefix + k: v for k, v in output.items()})

            # Early stopping
            monitor_metric = np.mean(output[monitor])
            scheduler.step(monitor_metric)
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
                    if isinstance(v, Number) and len(k.split("/")) < 3
                ],
            )

            if patience_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} as {monitor} did not improve for {patience} epochs."
                )
                break

    # Restore best model and optimizer
    if best_epoch != -1:
        decoder.load_state_dict(best_decoder_state_dict)
        optimizer.load_state_dict(best_optimizer_state_dict)

    # Revert to original train set ordering
    df_train = df_train.loc[init_train_index]

    return decoder, optimizer, best_epoch


def train(
    df_train,
    df_valid,
    df_test,
    df_ft_train,
    df_ft_valid,
    in_dims,
    decoder_cfg,
    wrapper_cfg,
    train_cfg,
    return_tables,
    nlp_distances,
    n_candidates,
    metrics_prefix,
):
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("high")

    out_dim = df_train.Y.iloc[0].shape[1]
    decoder = decoder_cfg["class"]
    if hasattr(base_decoders, decoder):
        decoder = getattr(base_decoders, decoder)(
            out_dim=out_dim, **decoder_cfg
        )
    else:
        raise ValueError(f"Unsupported decoder {decoder}.")
    decoder = DecoderWrapper(
        decoder=decoder,
        in_dims=in_dims,
        **wrapper_cfg,
    ).to(device)
    decoder = torch.compile(decoder)

    if wandb.run is not None:
        wandb.config["in_dims"] = in_dims
        wandb.config["out_dim"] = out_dim

    # Initialize optimizer
    optimizer = init_optimizer(decoder)

    top_k_accuracies = [1, 5, 10]

    # Train the decoder
    decoder, optimizer, best_epoch = train_loop(
        decoder=decoder,
        optimizer=optimizer,
        df_train=df_train,
        df_valid=df_valid,
        top_k_accuracies=top_k_accuracies,
        nlp_distances=nlp_distances,
        n_candidates=n_candidates,
        metrics_prefix=metrics_prefix,
        **train_cfg,
    )
    output = {"best_epoch": best_epoch}

    # Fine-tune if df_ft_train and df_ft_valid are not empty
    if not df_ft_train.empty and not df_ft_valid.empty:
        # Switch to fine-tuning parameters
        switch_to_finetune_mode(decoder.projector)
        switch_to_finetune_mode(decoder.decoder)
        for k, v in train_cfg.items():
            if k.endswith("_ft"):
                train_cfg[k.replace("_ft", "")] = v

        # Reinitialize optimizer
        optimizer = init_optimizer(decoder, prefix="Fine-tuning")

        decoder, optimizer, best_epoch_ft = train_loop(
            decoder=decoder,
            optimizer=optimizer,
            df_train=df_ft_train,
            df_valid=df_ft_valid,
            top_k_accuracies=top_k_accuracies,
            nlp_distances=nlp_distances,
            n_candidates=n_candidates,
            metrics_prefix="ft_" + metrics_prefix,
            **train_cfg,
        )
        output["best_epoch_ft"] = best_epoch_ft

    for split, df_split in [
        ("train", df_train),
        ("valid", df_valid),
        ("ft_train", df_ft_train),
        ("ft_valid", df_ft_valid),
        ("test", df_test),
    ]:
        if df_split.empty:
            continue
        Y_split = df_split.drop_duplicates(["dataset", "run"]).Y
        Y_split = torch.cat(tuple(Y_split))
        for key, value in evaluate(
            df=df_split,
            decoder=decoder,
            negatives=Y_split,
            top_k_accuracies=top_k_accuracies,
            nlp_distances=nlp_distances.get(split, None),
            n_candidates=n_candidates,
            return_tables=return_tables,
        ).items():
            output[split + "/" + key] = value
    if wandb.run is not None:
        wandb.summary.update(
            {
                metrics_prefix
                + k: (
                    wandb.Table(dataframe=v)
                    if isinstance(v, pd.DataFrame)
                    else v
                )
                for k, v in output.items()
            }
        )

    for split in ["Train", "Valid", "Test"]:
        split_key = f"{split.lower()}/relative_rank_median"
        if split_key in output:
            logger.info(f"{split} relative median rank {output[split_key]:.3g}")

    return output, decoder._orig_mod.cpu()
