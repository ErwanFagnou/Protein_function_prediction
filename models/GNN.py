import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, aggr, GENConv, AttentiveFP
from torch_geometric import transforms as T

from features import *
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class GNN(BaseProteinModel):

    # PCA_DIM = 64  # comment or -1 for no PCA

    # Add node features to the graph
    # transforms = T.Compose([
    #     TorsionFeatures(),
    #     # AnglesFeatures(),
    #     PositionInSequence(),
    #     CenterDistance(),
    #     # MahalanobisCenterDistance(),
    #
    #     T.VirtualNode(),
    #     T.LocalDegreeProfile(),
    #     # T.GDC(),
    #     T.AddLaplacianEigenvectorPE(k=3, attr_name=None, is_undirected=True),
    # ])

    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(GNN, self).__init__()

        self.config = ConfigDict(
            name='ESM2+GNN',
            epochs=400,
            batch_size=64,
            num_validation_samples=500,  # there are 4888 training samples, so 500 validation samples is 10%
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),  # , weight_decay=1e-5),
            hidden_dim=32,
            embedding_dim=32,
            dropout=0.5,
            num_layers=2,
            # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            # lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=pow(1e-2, 1/400)),
        )

        d = self.config.hidden_dim

        # self.node_proj = nn.Linear(num_node_features, self.config.embedding_dim)
        self.node_proj = nn.LazyLinear(self.config.embedding_dim)

        # Graph neural network
        self.gnn1 = GCNConv(self.config.embedding_dim, d)
        self.gnns = nn.ModuleList([GCNConv(d, d) for _ in range(self.config.num_layers-1)])

        # gnn_kwargs = dict(edge_dim=num_edge_features, dropout=0.2, add_self_loops=True, concat=False)
        # self.gnn1 = GATConv(self.config.embedding_dim, d, heads=2, **gnn_kwargs)
        # self.gnns = nn.ModuleList([GATConv(d, d, heads=2, **gnn_kwargs) for _ in range(self.config.num_layers-1)])

        # self.gnn = AttentiveFP(self.config.embedding_dim, d, d, edge_dim=num_edge_features, num_layers=self.config.num_layers, num_timesteps=2, dropout=self.config.dropout)

        # self.gnn = GENConv(self.config.embedding_dim, d, edge_dim=num_edge_features, aggr='softmax', num_layers=self.config.num_layers,
        #                    learn_p=True, learn_t=True, learn_msg_scale=True)

        # Aggregator
        self.aggregator = aggr.MeanAggregation()  # TODO: choose
        # self.aggregator = aggr.SumAggregation()
        # self.aggregator = aggr.SoftmaxAggregation(learn=True)
        # self.aggregator = aggr.GraphMultisetTransformer(in_channels=d, out_channels=d, hidden_channels=d, num_heads=8)
        # self.aggregator = aggr.LSTMAggregation(in_channels=d, out_channels=d)

        self.fc1 = nn.LazyLinear(d)
        self.fc2 = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, sequences, graphs):
        """
        :param sequences: A list of protein sequences of integers from 0 to 20, with different lengths.
        :param graphs: A Batch object from torch_geometric.data.Batch. It contains:
            - x: (num_nodes, num_node_features)
            - edge_index: (2, num_edges)
            - edge_attr: (num_edges, num_edge_features)
            - batch: (num_nodes, ) with the batch index of each node
        """

        # Reduce input dimensionality
        x = graphs.x
        x = self.node_proj(x)

        # x = self.dropout(x)  # Maybe some dropout here?

        # Apply graph neural network
        for gnn in [self.gnn1, *self.gnns]:  # for GCNConv or GATConv
            x = gnn(x, graphs.edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        # x = self.gnn(x, graphs.edge_index, graphs.edge_attr)  # for GENConv

        # x = self.gnn(x, graphs.edge_index, graphs.edge_attr, graphs.batch)  # for AttentiveFP -> no aggregation needed
        # x = x / torch.tensor([s.shape[0] for s in sequences], device=x.device).unsqueeze(1)  # taking mean instead of sum (better)

        # Aggregate node embeddings from graph (not for AttentiveFP)
        x = self.aggregator(x, graphs.batch)
        # x = self.aggregator(x, graphs.batch, edge_index=graphs.edge_index)  # for GraphMultisetTransformer

        # MLP to produce output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out
