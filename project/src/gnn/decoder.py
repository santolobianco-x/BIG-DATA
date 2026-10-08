import torch
import torch.nn as nn


class MLPDecoder(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        z,
        edge_index
    ):

        edge_index = edge_index.long()

        src = edge_index[0]
        dst = edge_index[1]

        z_src = z[src]
        z_dst = z[dst]

        product = z_src * z_dst

        difference = torch.abs(
            z_src - z_dst
        )

        edge_features = torch.cat(
            [
                product,
                difference
            ],
            dim=1
        )

        return self.mlp(
            edge_features
        ).view(-1)