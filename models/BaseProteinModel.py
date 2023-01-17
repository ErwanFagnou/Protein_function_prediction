import sys
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.data import Batch


class BaseProteinModel(ABC, pl.LightningModule):

    CREATE_SUBMISSION = True
    experiment_name = 'protein_classification'
    PCA_DIM = -1  # no PCA

    def __init__(self):
        super(BaseProteinModel, self).__init__()

        self.config = ConfigDict(
            name='Base',
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, sequences, graphs: Batch):
        """
        :param sequences: A list of protein sequences of integers from 0 to 20, with different lengths.
        :param graphs: A Batch object from torch_geometric.data.Batch. It contains:
            - x: (num_nodes, num_node_features)
            - edge_index: (2, num_edges)
            - edge_attr: (num_edges, num_edge_features)
            - batch: (num_nodes, ) with the batch index of each node
        :return: A tensor of shape (batch_size, num_classes)
        """
        raise NotImplementedError

    def prepare_inputs(self, sequences, adj, node_features, edge_features, node_idx, edge_idx):
        return sequences, adj, node_features, edge_features, node_idx, edge_idx

    def configure_optimizers(self, params=None):
        if params is None:
            params = self.parameters()
        optimizer = self.config.optimizer(params, **self.config.optimizer_kwargs)

        lr_scheduler = self.config.lr_scheduler(optimizer, **self.config.lr_scheduler_kwargs)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx, log=True, log_prefix='train_', prog_bar=False):
        sequences, graphs, labels = batch

        logits = self(sequences, graphs)
        loss = self.loss_fn(logits, labels)

        if log:
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            self.log(log_prefix + 'acc', acc, on_epoch=True, batch_size=labels.shape[0], prog_bar=prog_bar)

            self.log(log_prefix + 'loss', loss, on_epoch=True, batch_size=labels.shape[0], prog_bar=prog_bar)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.training_step(batch, batch_idx, log_prefix='val_', prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log=False)
        return loss

    def predict_probabilities(self, *inputs):
        logits = self(*inputs)
        return torch.softmax(logits, dim=-1)


class ConfigDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
