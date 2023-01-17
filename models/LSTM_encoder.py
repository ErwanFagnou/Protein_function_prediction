import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, aggr, GENConv, AttentiveFP

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class LSTMEncoder(BaseProteinModel):
    CREATE_SUBMISSION = False
    experiment_name = 'sequence_prediction'

    def __init__(self, num_node_features, num_classes):
        super(LSTMEncoder, self).__init__()

        self.config = ConfigDict(
            name='LSTM_256',
            hidden_dim=256,
            dropout=0.2,
            mask_rate=0.2,

            epochs=200,
            batch_size=32,
            num_validation_samples=100,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=2e-3),
            # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            # lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=pow(1e-2, 1/200)),
        )
        self.output_dim = 2 * self.config.hidden_dim

        d = self.config.hidden_dim

        self.dropout = nn.Dropout(self.config.dropout)

        self.rnn1 = nn.LSTM(num_node_features, d, batch_first=True)#, bidirectional=True)
        self.rnn2 = nn.LSTM(d, d, batch_first=True)#, bidirectional=True)
        self.res_projection1 = nn.Linear(num_node_features, d)
        self.res_projection2 = nn.Linear(d, d)

        # self.input_projection = nn.Linear(num_node_features, d)
        # self.attn1 = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        # self.attn2 = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        # nn.init.eye(self.input_projection.weight)
        # nn.init.eye(self.attn1.in_proj_weight)

        self.final_projection = nn.Linear(d, ProteinDataset.NUM_ACIDS)

        self.mask_token = nn.Parameter(torch.randn(1, num_node_features))

    def forward(self, sequences, graphs, return_embeddings=True, return_acc=False):
        node_features = graphs.x

        # Pad sequences
        idx_n = 0
        x, y = [], []
        max_len = max([len(s) for s in sequences])
        for seq_acid_ids in sequences:
            seq = node_features[idx_n:idx_n+len(seq_acid_ids)]  # TODO: + degree, angles, edge lengths, distance to center...
            idx_n += seq_acid_ids.shape[0]

            x.append(torch.cat([seq, torch.zeros(max_len - len(seq), seq.shape[1], device=seq.device)], dim=0))
            if not return_embeddings:
                y.append(torch.cat([seq_acid_ids, -torch.ones(max_len - len(seq_acid_ids), device=seq.device)], dim=0))
        x = torch.stack(x, dim=0)
        if not return_embeddings:
            x = x[:, :-1]
            y = torch.stack(y, dim=0).long()
            y = y[:, 1:]


        # # Mask some sequence vectors with a special token
        # masked = torch.rand(x.shape[0], x.shape[1], device=x.device) < self.config.mask_rate
        # x = torch.where(masked[:, :, None], self.mask_token, x)

        x1, _ = self.rnn1(x)
        # seq = self.dropout(x)

        x, _ = self.rnn2(x1)
        # x = self.dropout(x)

        if return_embeddings:
            return torch.cat([x1, x], dim=-1)

        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        pred = self.final_projection(x)

        # Compute loss and accuracy
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))

        if return_acc:
            acc = (pred.argmax(dim=-1) == y).float().sum() / (y != -1).float().sum()
            return loss, acc

        return loss

    def training_step(self, batch, batch_idx, log=True, log_prefix='train_', prog_bar=False):
        sequences, graphs, labels = batch

        if log:
            loss, acc = self(sequences, graphs, return_embeddings=False, return_acc=True)

            self.log(log_prefix + 'acc', acc, on_epoch=True, batch_size=labels.shape[0], prog_bar=prog_bar)
            self.log(log_prefix + 'loss', loss, on_epoch=True, batch_size=labels.shape[0], prog_bar=prog_bar)
        else:
            loss = self(sequences, graphs)
        return loss
