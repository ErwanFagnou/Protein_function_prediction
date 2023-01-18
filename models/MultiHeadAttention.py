import torch
from torch import nn

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class MultiHeadAttention(BaseProteinModel):
    CREATE_SUBMISSION = False
    experiment_name = 'masked_sequence_prediction'

    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(MultiHeadAttention, self).__init__()

        self.config = ConfigDict(
            name='multihead_attention',
            hidden_dim=256,
            num_layers=3,
            num_heads=16,

            mask_rate=0.2,
            mask_error_rate=0.1,
            dropout=0.2,

            epochs=200,
            batch_size=16,
            num_validation_samples=100,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            # lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=pow(1e-4, 1/200)),
        )
        self.output_dim = self.config.hidden_dim * self.config.num_layers

        d = self.config.hidden_dim

        self.mask_vector = nn.Parameter(torch.randn(d))
        # self.bos_vector = nn.Parameter(torch.randn(d))
        # self.eos_vector = nn.Parameter(torch.randn(d))

        self.node_proj = nn.LazyLinear(d)
        self.fc1 = nn.LazyLinear(d)
        self.fc2 = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=self.config.num_heads, dropout=self.config.dropout)

    def forward(self, sequences, graphs, return_embeddings=True, random_mask=False):

        # sequences: (batch_size, seq_len, d)
        # graphs: (batch_size, seq_len, seq_len)

        #batch_size, seq_len, d = sequences.shape

        #just apply multihead attention to the sequences
        x = graphs.x
        x = self.node_proj(x)
        x, _ = self.attention(x, x, x)

        #MLP to produce output

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out
