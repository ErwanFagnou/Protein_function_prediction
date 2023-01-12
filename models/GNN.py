import torch
from torch import nn
from torch_geometric.nn import GCNConv, aggr

from models.BaseProteinModel import BaseProteinModel, ConfigDict


class GNN(BaseProteinModel):

    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()

        self.config = ConfigDict(
            name='GNN+CNN',
            epochs=50,
            batch_size=32,
            num_validation_samples=100,  # there are 4888 training samples, so 100 validation samples is ok
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            hidden_dim=64,
            dropout=0.2,
        )

        d = self.config.hidden_dim

        # Graph encoder
        self.gnn1 = GCNConv(num_node_features, d)  # TODO: cached=True + use edges
        self.gnn2 = GCNConv(d, d)
        self.aggregator = aggr.MeanAggregation()  # TODO: choose

        self.bn = nn.BatchNorm1d(d)
        self.fc3 = nn.Linear(d, d)
        self.fc4 = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, sequences, graphs):
        node_features = graphs.x

        # Graph encoder
        x = self.gnn1(node_features, graphs.edge_index)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.gnn2(x, graphs.edge_index)
        x = self.aggregator(x, graphs.batch)
        graph_emb = self.bn(x)
        x = graph_emb

        # mlp to produce output
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        out = self.fc4(x)

        return out
