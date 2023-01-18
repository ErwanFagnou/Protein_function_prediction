import torch
from torch import nn
from torch_geometric.nn import aggr
from torch_geometric import transforms as T

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class MultiHeadAttention(BaseProteinModel):
    CREATE_SUBMISSION = True
    PCA_DIM = 64  # comment or -1 for no PCA

    # Add node features to the graph
    """transforms = T.Compose([
        T.TorsionFeatures(),
         # AnglesFeatures(),
        T.PositionInSequence(),
        T.CenterDistance(),
         # MahalanobisCenterDistance(),
    
        T.VirtualNode(),
        T.LocalDegreeProfile(),
        # T.GDC(),
        T.AddLaplacianEigenvectorPE(k=3, attr_name=None, is_undirected=True),
     ])"""

    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(MultiHeadAttention, self).__init__()

        self.config = ConfigDict(
            name='ESM2_35M+MHA(d=128,h=10)+query=random+10queries+dropout=0.5+PCA+lastHidden',
            hidden_dim=128,
            num_layers=1,
            num_heads=10,

            dropout=0.5,
            num_queries=10,
            epochs=200,
            batch_size=64,
            num_validation_samples=500,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            # lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=pow(1e-4, 1/200)),
        )

        d = self.config.hidden_dim

        self.node_proj = nn.LazyLinear(d)
        self.fc1 = nn.LazyLinear(d)
        self.fc2 = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=self.config.num_heads, dropout=self.config.dropout, batch_first=True)
        # self.aggregator = aggr.MeanAggregation()

        self.queries = torch.randn(self.config.num_queries, d)
        self.dev = None

    def forward(self, sequences, graphs, return_embeddings=True, random_mask=False):
        node_features = graphs.x
        node_features = self.node_proj(node_features)

        if not self.dev:
            self.dev = node_features.device
            self.queries = self.queries.to(self.dev) 
        # Pad sequences
        idx_n = 0
        x, attn_mask = [], []
        lengths = [len(s)-1 for s in sequences]
        max_len = max(lengths)+1
        for acid_ids in sequences:
            seq = node_features[idx_n:idx_n + len(acid_ids)]
            idx_n += seq.shape[0]

            x.append(torch.cat([seq, torch.zeros(max_len - len(seq), seq.shape[1], device=seq.device)], dim=0))
            attn_mask.append(torch.cat(
                [torch.ones(len(acid_ids), device=seq.device), torch.zeros(max_len - len(acid_ids), device=seq.device)],
                dim=0))
        x = torch.stack(x, dim=0)  # (batch_size, max_len, d)
        attn_mask = torch.stack(attn_mask, dim=0)  # (batch_size, max_len)

        # Just one query (the number of output vectors is equal to the number of queries)
        #query = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)  # constant vector, but can also be a function of the node features

        #query the embedding of the token cls
        #query = x[:, 0, :].unsqueeze(1)


        #query all the embeddings
        #query = x

        #get all embeddings at index len(seq)-1 and the embedding of the token cls
        #query = torch.cat([x[torch.arange(x.shape[0]), lengths, :].unsqueeze(1), x[:, 0, :].unsqueeze(1)], dim=1)

        #get a batch of queries, full of self.queries
        query = self.queries.repeat(x.shape[0], 1, 1)

        # just apply multihead attention to the sequences, to produce a single vector for each sequence
        x, _ = self.attention(query, x, x, key_padding_mask=attn_mask)  # (batch_size, max_len, d)

        #concatenate the final vectors
        x = x.reshape(x.shape[0], -1)

        #x = x[:, 0, :]  # (batch_size, d)

        # x = self.aggregator(x, graphs.batch)

        # MLP to produce output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out
