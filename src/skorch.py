from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import wandb
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping, EpochScoring, WandbLogger
from src.metrics import retrieval_metrics, scores
from src.utils import device


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=0.2),
        )

    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X


def simple_MLP(
    in_dim,
    out_dim,
    hidden_size=1024,
    max_epochs=100,
    batch_size=1024,
):
    return {
        "module": SimpleMLP(in_dim, hidden_size, out_dim),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "optimizer": optim.AdamW,
        "criterion": nn.MSELoss,
    }


class MixcoSymmetricalNCELoss(nn.Module):
    def __init__(self, temperature, bidirectional):
        super().__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional

    def forward(self, preds, targs):
        brain_clip = (preds @ targs.T) / self.temperature
        loss = F.cross_entropy(
            brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device)
        )
        if self.bidirectional:
            loss2 = F.cross_entropy(
                brain_clip.T,
                torch.arange(brain_clip.shape[0]).to(brain_clip.device),
            )
            loss = (loss + loss2) / 2

        return loss


def simple_MLP_contrastive(
    in_dim,
    out_dim,
    hidden_size=1024,
    max_epochs=100,
    batch_size=1024,
    temperature=0.1,
    bidirectional=True,
):
    return {
        "module": SimpleMLP(in_dim, hidden_size, out_dim),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "optimizer": optim.AdamW,
        "criterion": MixcoSymmetricalNCELoss(temperature, bidirectional),
    }


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_size_backbone=512,
        hidden_size_projector=512,
        dropout=0.2,
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

        return x, self.projector(x)


def brain_decoder_contrastive(
    in_dim,
    out_dim,
    max_epochs=100,
    batch_size=2048,
    temperature=0.1,
    bidirectional=True,
    **module_params,
):
    return {
        "module": BrainDecoder(in_dim, out_dim, **module_params),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "optimizer": optim.AdamW,
        "criterion": MixcoSymmetricalNCELoss(temperature, bidirectional),
    }


def skorch(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    X_test,
    Y_test,
    decoder,
    patience=10,
    seed=0,
    setup_config={},
    verbose=False,
    **decoder_params,
):
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    config = {"decoder": decoder}
    config.update(decoder_params)
    name = "_".join([f"{key}={value}" for key, value in config.items()])
    train_size = X_train.shape[0]
    X_train = np.vstack([X_train, X_valid])
    Y_train = np.vstack([Y_train, Y_valid])
    dataset = data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))

    def train_split(dataset):
        dataset_train = data.Subset(dataset, torch.arange(train_size))
        dataset_valid = data.Subset(dataset, torch.arange(train_size, len(dataset)))
        return dataset_train, dataset_valid

    config.update(setup_config)
    wandb_run = wandb.init(
        name=name,
        config=config,
        id=f"{hash(frozenset(config.items())):02x}",
        save_code=True,
    )

    def retrieval_scorer(net, X, y):
        return retrieval_metrics(
            y, net.predict(X), metric="cosine", top_k_accuracies=[], n_jobs=-2
        )["relative_median_rank"]

    if decoder.lower() == "simple_mlp":
        neuralnet_params = simple_MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            **decoder_params,
        )
    elif decoder.lower() == "simple_mlp_contrastive":
        neuralnet_params = simple_MLP_contrastive(
            in_dim=in_dim,
            out_dim=out_dim,
            **decoder_params,
        )
    elif decoder.lower() == "brain_decoder_contrastive":
        neuralnet_params = brain_decoder_contrastive(
            in_dim=in_dim,
            out_dim=out_dim,
            **decoder_params,
        )

    model = NeuralNet(
        iterator_train__shuffle=True,
        train_split=train_split,
        device=device,
        callbacks=[
            EpochScoring(
                retrieval_scorer,
                name="train_relative_median_rank",
                on_train=True,
                target_extractor=lambda y: y,
            ),
            EpochScoring(
                retrieval_scorer,
                name="valid_relative_median_rank",
                target_extractor=lambda y: y,
            ),
            EarlyStopping(
                monitor="valid_relative_median_rank",
                patience=patience,
            ),
            WandbLogger(wandb_run, save_model=False),
        ],
        verbose=verbose,
        **neuralnet_params,
    )
    model.fit(dataset)
    wandb_run.finish()

    output = {}
    for t, (X, Y) in [
        ("train", (X_train, Y_train)),
        ("test", (X_test, Y_test)),
    ]:
        Y_pred = model.predict(X)
        for key, value in scores(Y, Y_pred).items():
            output[f"{t}_{key}"] = value

    return output
