from functools import partial

from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=512, num_layers=3, dropout=0.7):
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

    def forward(self, X):
        for layer in self.fc:
            X = layer(X)
        return X


class RNN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers=1,
        dropout=0.7,
        **rnn_params,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            **rnn_params,
        )

    def forward(self, X):
        return self.rnn(X)[0].data


class GRU(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers=1,
        dropout=0.7,
        **gru_params,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            **gru_params,
        )

    def forward(self, X):
        return self.gru(X)[0].data


class LSTM(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_size=None,
        num_layers=1,
        dropout=0.7,
        **lstm_params,
    ):
        if hidden_size is None:
            hidden_size = out_dim
            proj_size = 0
        else:
            proj_size = out_dim
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            proj_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            **lstm_params,
        )

    def forward(self, X):
        return self.lstm(X)[0].data


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_dims,
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
        self.lin0 = nn.ModuleDict(
            {
                subject: nn.Sequential(
                    nn.Linear(in_dim, hidden_size_backbone),
                    *[item() for item in activation_and_norm],
                    nn.Dropout(dropout),
                )
                for subject, in_dim in in_dims.items()
            }
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

    def project_subject(self, x, subject):
        return self.lin0[subject](x)

    def forward(self, x):
        residual = x
        for res_block in range(self.n_res_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)

        x = self.lin1(x)

        return self.projector(x)
