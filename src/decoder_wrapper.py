from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DecoderWrapper(nn.Module):
    def __init__(
        self,
        decoder,
        in_dims,
        loss="mixco",
        multi_subject_mode="individual",
        hidden_size=512,
        dropout=0.7,
        norm_type="ln",
        activation_layer_first=False,
        temperature=0.05,
        beta=0.15,
        s_thresh=0.5,
        **kwargs,
    ):
        super().__init__()
        self.decoder = decoder
        self.in_dims = in_dims
        self.loss = loss
        self.multi_subject_mode = multi_subject_mode
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.norm_type = norm_type
        self.activation_layer_first = activation_layer_first
        self.temperature = temperature
        self.beta = beta
        self.s_thresh = s_thresh
        self.train_decoder = True

        if self.loss == "mixco":
            self.loss = self.mixco_loss
        elif self.loss == "symm_nce":
            self.loss = self.symm_nce_loss
        elif self.loss == "mse":
            self.loss = self.mse_loss
        else:
            raise ValueError(
                f"loss must be one of ['mixco', 'symm_nce', 'mse'] but got {self.loss}"
            )

        norm_backbone = (
            partial(nn.BatchNorm1d, num_features=self.hidden_size)
            if self.norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=self.hidden_size)
        )
        activation_backbone = (
            partial(nn.ReLU, inplace=True)
            if self.norm_type == "bn"
            else nn.GELU
        )
        self.activation_and_norm = (
            (activation_backbone, norm_backbone)
            if self.activation_layer_first
            else (norm_backbone, activation_backbone)
        )
        self.projector = self.init_projector()

    def init_projector(self):
        if self.multi_subject_mode == "shared":
            all_dims = [
                dim for dims in self.in_dims.values() for dim in dims.values()
            ]
            assert (
                len(set(all_dims)) == 1
            ), f"In multi_subject_mode 'shared', all subjects must have the same input dimension but got {self.in_dims}"
            in_dim = all_dims[0]
            projector = nn.Sequential(
                nn.Linear(in_dim, self.hidden_size),
                *[item() for item in self.activation_and_norm],
                nn.Dropout(self.dropout),
            )

            return nn.ModuleDict(
                {
                    dataset: nn.ModuleDict(
                        {subject: projector for subject in dims}
                    )
                    for dataset, dims in self.in_dims.items()
                }
            )
        elif self.multi_subject_mode == "dataset":
            projector = nn.ModuleDict()
            for dataset, dims in self.in_dims.items():
                dims = list(dims.values())
                assert (
                    len(set(dims)) == 1
                ), f"In multi_subject_mode 'dataset', all subjects in a dataset must have the same input dimension but got {in_dims[dataset]} for {dataset}"
                in_dim = dims[0]
                dataset_projector = nn.Sequential(
                    nn.Linear(in_dim, self.hidden_size),
                    *[item() for item in self.activation_and_norm],
                    nn.Dropout(self.dropout),
                )
                projector[dataset] = nn.ModuleDict(
                    {
                        subject: dataset_projector
                        for subject in self.in_dims[dataset]
                    }
                )

            return projector
        elif self.multi_subject_mode == "individual":

            return nn.ModuleDict(
                {
                    dataset: nn.ModuleDict(
                        {
                            subject: nn.Sequential(
                                nn.Linear(in_dim, self.hidden_size),
                                *[item() for item in self.activation_and_norm],
                                nn.Dropout(self.dropout),
                            )
                            for subject, in_dim in dims.items()
                        }
                    )
                    for dataset, dims in self.in_dims.items()
                }
            )
        else:
            raise ValueError(
                f"multi_subject_mode must be one of ['shared', 'dataset', 'individual'] but got {self.multi_subject_mode}"
            )

    def forward(self, X):
        return self.decoder(X)

    def mixco_loss(self, X_proj, Y):
        # Randomly select samples to augment
        select = (torch.rand(X_proj.shape[0]) <= self.s_thresh).to(
            X_proj.device
        )

        X_proj_aug = X_proj.clone()
        # Randomly select samples used for augmentation
        perm = torch.randperm(X_proj.shape[0])
        samples_shuffle = X_proj_aug[perm].to(X_proj.device, dtype=X_proj.dtype)

        # Sample MixCo coefficients from a Beta distribution
        betas = (
            torch.distributions.Beta(self.beta, self.beta)
            .sample([X_proj.shape[0]])
            .to(X_proj.device, dtype=X_proj.dtype)
        )
        betas[~select] = 1
        betas_shape = [-1] + [1] * (len(X_proj.shape) - 1)

        # Augment samples
        X_proj_aug[select] = X_proj_aug[select] * betas[select].reshape(
            *betas_shape
        ) + samples_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)

        Y_preds = self(X_proj_aug)
        Y_preds_norm = F.normalize(Y_preds, dim=-1)
        Y_norm = F.normalize(Y, dim=-1)
        logits = Y_preds_norm @ Y_norm.T / self.temperature
        probs = torch.diag(betas)
        probs[
            torch.arange(Y_preds_norm.shape[0]).to(Y_preds_norm.device), perm
        ] = (1 - betas)
        loss = -(logits.log_softmax(-1) * probs).sum(-1).mean()
        loss2 = -(logits.T.log_softmax(-1) * probs.T).sum(-1).mean()

        return (loss + loss2) / 2

    def symm_nce_loss(self, X_proj, Y, Y_preds=None):
        if Y_preds is None:
            Y_preds = self(X_proj)
        Y_preds_norm = F.normalize(Y_preds, dim=-1)
        Y_norm = F.normalize(Y, dim=-1)
        logits = Y_preds_norm @ Y_norm.T / self.temperature
        target = torch.arange(logits.shape[0]).to(logits.device)
        loss = F.cross_entropy(logits, target)
        loss2 = F.cross_entropy(logits.T, target)

        return (loss + loss2) / 2

    def mse_loss(self, X_proj, Y, Y_preds=None):
        if Y_preds is None:
            Y_preds = self(X_proj)

        return F.mse_loss(Y_preds, Y)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_decoder:
            self.decoder.eval()
