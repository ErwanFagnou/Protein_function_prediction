import torch
from torch import nn

from models.BaseProteinModel import BaseProteinModel, ConfigDict


class GNN(BaseProteinModel):

    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()

        self.config = ConfigDict(
            name='GNN+CNN',
            epochs=100,
            batch_size=32,
            num_validation_samples=100,  # there are 4888 training samples, so 100 validation samples is ok
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            hidden_dim=64,
            dropout=0.2,
        )

        d = self.config.hidden_dim

        # Graph encoder
        self.gnn1 = GNNLayer(num_node_features, d)
        self.gnn2 = GNNLayer(d, d)
        self.aggregator = GNNSumAggregator()

        # Sequence encoder
        self.acid_embedding = nn.Embedding(21, d)
        self.conv1 = nn.Conv1d(d, d, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(d, d, kernel_size=5, padding='same')

        self.bn = nn.BatchNorm1d(d)
        self.fc3 = nn.Linear(2*d, d)
        self.fc4 = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, sequences, adj, node_features, edge_features, node_idx, edge_idx):

        # Graph encoder
        x = node_features[:, 3:]  # remove coordinates
        x = self.gnn1(x, adj)
        x = self.dropout(x)
        x = self.gnn2(x, adj)
        x = self.aggregator(x, node_idx)
        graph_emb = self.bn(x)

        # Sequence encoder
        seq_emb = []
        for seq in sequences:
            # print(seq)
            acid_embs = self.acid_embedding(seq.unsqueeze(0)).transpose(-1, -2)
            # print(acid_embs.shape, seq.shape)
            # print(acid_embs)
            x = self.conv1(acid_embs)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = x.mean(dim=-1)
            seq_emb.append(x)
        seq_emb = torch.cat(seq_emb, dim=0)

        x = torch.cat([graph_emb, seq_emb], dim=-1)

        # mlp to produce output
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        out = self.fc4(x)

        return out


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation_cls=nn.ReLU):
        super(GNNLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation_cls()

    def forward(self, x, adj):
        x = self.fc(x)
        x = torch.mm(adj, x)
        x = self.activation(x)
        return x


class GNNSumAggregator(nn.Module):
    def __init__(self):
        super(GNNSumAggregator, self).__init__()

    def forward(self, x, node_idx):
        idx = node_idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x.device)
        out = out.scatter_add_(0, idx, x)
        return out