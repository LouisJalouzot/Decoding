from functools import partial

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from src.utils import device


class SimpleMLP(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_size=512, num_layers=3, dropout=0.7, **kwargs
    ):
        super().__init__()
        self.fc = [
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            self.fc.extend(
                [
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                ]
            )
        self.fc.append(nn.Linear(in_features=hidden_size, out_features=out_dim))
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        x = x.data
        for layer in self.fc:
            x = layer(x)
        return x


class RNN(nn.Module):
    def __init__(
        self,
        out_dim,
        hidden_size=512,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        dropout=0.7,
        **kwargs,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=out_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        return self.rnn(x)[0].data


class GRU(nn.Module):
    def __init__(
        self,
        out_dim,
        hidden_size=512,
        num_layers=1,
        bias=True,
        dropout=0.7,
        **kwargs,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=out_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        return self.gru(x)[0].data


class LSTM(nn.Module):
    def __init__(
        self,
        out_dim,
        hidden_size=512,
        num_layers=1,
        bias=True,
        proj_size=0,
        dropout=0.7,
        **kwargs,
    ):
        super().__init__()
        input_size = hidden_size
        if proj_size > 0:
            assert proj_size < hidden_size
            assert proj_size == out_dim
        else:
            assert hidden_size == out_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            proj_size=proj_size,
        )

    def forward(self, x):
        return self.lstm(x)[0].data


class BrainDecoder(nn.Module):
    def __init__(
        self,
        out_dim,
        hidden_size=512,
        hidden_size_projector=512,
        dropout=0.7,
        n_res_blocks=2,
        n_proj_blocks=1,
        norm_type="ln",
        activation_layer_first=False,
        **kwargs,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.hidden_size_projector = hidden_size_projector
        self.dropout = dropout
        self.n_res_blocks = n_res_blocks
        self.n_proj_blocks = n_proj_blocks
        self.norm_type = norm_type
        self.activation_layer_first = activation_layer_first

        norm_backbone = (
            partial(nn.BatchNorm1d, num_features=self.hidden_size)
            if self.norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=self.hidden_size)
        )
        activation_backbone = (
            partial(nn.ReLU, inplace=True) if self.norm_type == "bn" else nn.GELU
        )
        self.activation_and_norm = (
            (activation_backbone, norm_backbone)
            if self.activation_layer_first
            else (norm_backbone, activation_backbone)
        )

        # Residual blocks
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    *[item() for item in self.activation_and_norm],
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.n_res_blocks)
            ]
        )

        # Second linear
        self.lin1 = nn.Linear(self.hidden_size, self.hidden_size_projector, bias=True)

        # Projector
        assert self.n_proj_blocks >= 0
        projector_layers = []
        for _ in range(self.n_proj_blocks):
            projector_layers.extend(
                [
                    nn.LayerNorm(self.hidden_size_projector),
                    nn.GELU(),
                    nn.Linear(self.hidden_size_projector, self.hidden_size_projector),
                ]
            )
        projector_layers.extend(
            [
                nn.LayerNorm(self.hidden_size_projector),
                nn.GELU(),
                nn.Linear(self.hidden_size_projector, self.out_dim),
            ]
        )
        self.projector = nn.Sequential(*projector_layers)

    def forward(self, x):
        x = x.data
        residual = x
        for res_block in range(self.n_res_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)

        x = self.lin1(x)

        return self.projector(x)


class DecoderWrapper(nn.Module):
    def __init__(
        self,
        decoder,
        in_dims,
        multi_subject_mode="individual",
        hidden_size=512,
        dropout=0.7,
        norm_type="ln",
        activation_layer_first=False,
        **kwargs,
    ):
        super().__init__()
        self.decoder = decoder
        self.in_dims = in_dims
        self.multi_subject_mode = multi_subject_mode
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.norm_type = norm_type
        self.activation_layer_first = activation_layer_first

        norm_backbone = (
            partial(nn.BatchNorm1d, num_features=self.hidden_size)
            if self.norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=self.hidden_size)
        )
        activation_backbone = (
            partial(nn.ReLU, inplace=True) if self.norm_type == "bn" else nn.GELU
        )
        self.activation_and_norm = (
            (activation_backbone, norm_backbone)
            if self.activation_layer_first
            else (norm_backbone, activation_backbone)
        )

        if self.multi_subject_mode == "shared":
            assert (
                len(set(in_dim for _, in_dim in self.in_dims.items())) == 1
            ), f"In multi_subject_mode 'shared', all subjects must have the same input dimension but got {self.in_dims}"
            _, in_dim = list(self.in_dims.items())[0]
            projector = nn.Sequential(
                nn.Linear(in_dim, self.hidden_size),
                *[item() for item in self.activation_and_norm],
                nn.Dropout(self.dropout),
            )
            self.projector = nn.ModuleDict(
                {subject: projector for subject in self.in_dims}
            )
        elif self.multi_subject_mode == "individual":
            self.projector = nn.ModuleDict(
                {
                    subject: nn.Sequential(
                        nn.Linear(in_dim, self.hidden_size),
                        *[item() for item in self.activation_and_norm],
                        nn.Dropout(self.dropout),
                    )
                    for subject, in_dim in self.in_dims.items()
                }
            )

    def project_subject(self, x):
        out = []
        for run in x:
            data = run.sel(
                tr=slice(run.n_trs.item()),
                voxel=slice(run.n_voxels.item()),
            ).transpose("tr", "voxel")
            data = torch.from_numpy(data).to(device)
            data = self.projector[run.subject.item()](data)
            out.append(data)
        return pack_sequence(out, enforce_sorted=False)

    def forward(self, x):
        return self.decoder(x)
