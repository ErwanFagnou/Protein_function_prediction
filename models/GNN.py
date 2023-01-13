import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, aggr, GENConv, AttentiveFP

from models.BaseProteinModel import BaseProteinModel, ConfigDict


class GNN(BaseProteinModel):

    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(GNN, self).__init__()

        self.config = ConfigDict(
            name='AttentiveFP+LSTM',
            epochs=200,
            batch_size=32,
            num_validation_samples=100,  # there are 4888 training samples, so 100 validation samples is ok
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            hidden_dim=64,
            embedding_dim=10,
            dropout=0.2,
            num_layers=3,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
        )

        d = self.config.hidden_dim

        self.node_proj = nn.Linear(num_node_features, self.config.embedding_dim)
        self.seq_embed = nn.Embedding(21, self.config.embedding_dim)

        # Graph encoder  # TODO: cached=True + use edges
        # gnn_kwargs = dict(edge_dim=num_edge_features, dropout=0.2, add_self_loops=True, concat=False)
        # self.gnn1 = GATConv(num_node_features, d, heads=5, **gnn_kwargs)
        # self.gnns = nn.ModuleList([GATConv(d, d, heads=5, **gnn_kwargs) for _ in range(self.config.num_layers-1)])

        self.gnn = AttentiveFP(self.config.embedding_dim, d, d, edge_dim=num_edge_features, num_layers=2, num_timesteps=5, dropout=0.2)

        # self.gnn = GENConv(self.config.embedding_dim, d, edge_dim=num_edge_features, aggr='softmax', num_layers=10,
        #                    learn_p=True, learn_t=True, learn_msg_scale=True)

        self.aggregator = aggr.MeanAggregation()  # TODO: choose
        # self.aggregator = aggr.GraphMultisetTransformer(in_channels=d, out_channels=d, hidden_channels=d, num_heads=8)
        # self.aggregator = aggr.LSTMAggregation(in_channels=d, out_channels=d)

        self.bn = nn.BatchNorm1d(d)
        self.fc3 = nn.LazyLinear(d)
        self.fc4 = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

        # self.seq_attn = nn.MultiheadAttention(embed_dim=self.config.embedding_dim, num_heads=5, dropout=0.2, batch_first=True)
        self.seq_rnn = nn.LSTM(self.config.embedding_dim, self.config.embedding_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, sequences, graphs):
        x = graphs.x
        x = self.node_proj(x)

        # idx_n = 0
        # x = []
        # max_len = max([len(s) for s in sequences])
        # for seq_acid_ids in sequences:
        #     seq = graphs.x[idx_n:idx_n+len(seq_acid_ids)]
        #     seq = self.node_proj(seq)
        #     idx_n += len(seq_acid_ids)
        #
        #     x.append(torch.cat([seq, torch.zeros(max_len - len(seq), seq.shape[1], device=seq.device)], dim=0))
        # x = torch.stack(x, dim=0)
        # x, _ = self.seq_rnn(x)
        # x = torch.cat([x[i, :len(s)] for i, s in enumerate(sequences)], dim=0)

        # new_x = []
        # for seq in sequences:
        #     seq = self.seq_embed(seq)
        #     # seq, _ = self.seq_attn(seq, seq, seq)
        #     seq, _ = self.seq_rnn(seq)
        #     new_x.append(seq)
        # x = torch.cat(new_x, dim=0)
        # print(x.shape, graphs.x.shape)

        # Graph encoder
        # for gnn in [self.gnn1, *self.gnns]:
        #     x = gnn(x, graphs.edge_index, graphs.edge_attr)
        #     x = self.relu(x)
        #     x = self.dropout(x)

        # x = self.gnn(x, graphs.edge_index, graphs.edge_attr)
        # x = self.dropout(x)

        x = self.gnn(x, graphs.edge_index, graphs.edge_attr, graphs.batch)  # AttentiveFP

        # x = self.aggregator(x, graphs.batch)
        # x = self.aggregator(x, graphs.batch, edge_index=graphs.edge_index)
        graph_emb = self.bn(x)
        x = graph_emb

        # frequencies = []
        # for seq in sequences:
        #     freq = torch.bincount(seq, minlength=21).float()
        #     freq = freq * torch.log(seq.shape[0] / (freq + 1))
        #     # freq = freq / seq.shape[0]
        #     frequencies.append(freq)
        # frequencies = torch.stack(frequencies)
        # # x = torch.cat([x, frequencies], dim=1)
        # x = frequencies

        # seq_embeds = []
        # for seq in sequences:
        #     seq = self.seq_embed(seq).unsqueeze(0)
        #     seq_embeds.append(self.seq_attn(seq, seq, seq)[0])
        # seq_embeds = torch.cat(seq_embeds, dim=0)
        # # x = torch.cat([x, seq_embeds], dim=1)
        # x = seq_embeds

        # mlp to produce output
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        out = self.fc4(x)

        return out
