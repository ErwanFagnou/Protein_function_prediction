import csv
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
import pickle
import bz2

import utils


class ProteinDataset:
    data_dir = './dataset/'

    cache_dataset = True
    cache_file = data_dir + 'dataset.pkl.bz2'

    NUM_CLASSES = 18
    NUM_NODE_FEATURES = 86

    NUM_ACIDS = 21
    amino_acids = 'ACDEFGHIKLMNPQRSTVWXY'
    acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    def __init__(self, batch_size, num_validation_samples=0, normalize_adj=True):
        self.batch_size = batch_size
        self.validation_samples = num_validation_samples

        if self.cache_dataset and os.path.exists(self.cache_file):
            print('Loading dataset from cache...')
            with bz2.BZ2File(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.protein_names = data['protein_names']
                self.protein_labels = data['protein_labels']
                self.has_label = data['has_label']
                self.sequences = data['sequences']
                self.adjacency_matrices = data['adjacency_matrices']
                self.node_features = data['node_features']
                self.edge_features = data['edge_features']
        else:
            print('Loading dataset from raw files. This may take a few minutes.')
            # Read labels
            print('\t[1/6] Reading labels')
            self.protein_names = []
            self.protein_labels = []
            self.has_label = []
            with open(self.data_dir + 'graph_labels.txt', 'r') as f:
                for i, line in enumerate(f):
                    t = line.split(',')
                    self.protein_names.append(t[0])
                    has_label = (len(t[1][:-1]) > 0)
                    self.has_label.append(has_label)
                    if has_label:
                        self.protein_labels.append(int(t[1][:-1]))
                    else:
                        self.protein_labels.append(-1)
            self.protein_names = np.array(self.protein_names)
            self.protein_labels = np.array(self.protein_labels)
            self.has_label = np.array(self.has_label)

            # Read sequences
            print('\t[2/6] Reading sequences')
            self.sequences = []
            with open(self.data_dir + 'sequences.txt', 'r') as f:
                for line in f:
                    acids = np.array([self.acid_to_index[aa] for aa in line[:-1]])
                    self.sequences.append(acids)

            # Read node attributes
            print('\t[3/6] Reading node attributes')
            node_attr = np.loadtxt(self.data_dir + "node_attributes.txt", delimiter=",")
            total_num_nodes = node_attr.shape[0]

            # Read edge attributes
            print('\t[4/6] Reading edge attributes')
            edge_attr = np.loadtxt(self.data_dir + "edge_attributes.txt", delimiter=",")

            # Read adjacency matrix
            print('\t[5/6] Reading adjacency matrix')
            edges = np.loadtxt(self.data_dir + "edgelist.txt", dtype=np.int64, delimiter=",")
            A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(total_num_nodes, total_num_nodes))
            A += A.T
            A = (A > 0)  # fix

            # Separate each protein graph
            print('\t[6/6] Separating graphs')
            self.adjacency_matrices = []
            self.node_features = []
            self.edge_features = []
            idx_n = 0
            idx_m = 0
            graph_indicator = np.loadtxt(self.data_dir + "graph_indicator.txt", dtype=np.int64)
            _, graph_sizes = np.unique(graph_indicator, return_counts=True)
            for i in range(len(graph_sizes)):
                self.adjacency_matrices.append(A[idx_n:idx_n + graph_sizes[i], idx_n:idx_n + graph_sizes[i]])

                self.node_features.append(node_attr[idx_n:idx_n + graph_sizes[i]])
                self.edge_features.append(edge_attr[idx_m:idx_m + self.adjacency_matrices[i].nnz])

                idx_n += graph_sizes[i]
                idx_m += self.adjacency_matrices[i].nnz

            if self.cache_dataset:
                print('(Saving dataset to load faster next time)')
                data = {
                    'protein_names': self.protein_names,
                    'protein_labels': self.protein_labels,
                    'has_label': self.has_label,
                    'sequences': self.sequences,
                    'adjacency_matrices': self.adjacency_matrices,
                    'node_features': self.node_features,
                    'edge_features': self.edge_features
                }
                with bz2.BZ2File(self.cache_file, 'wb') as f:
                    pickle.dump(data, f)

        if normalize_adj:
            print('Normalizing adjacency matrices...')
            self.adjacency_matrices = [utils.normalize_adjacency(A) for A in self.adjacency_matrices]

        inputs = (self.sequences, self.node_features, self.edge_features, self.adjacency_matrices)
        labels = self.protein_labels
        all_samples = list(zip(zip(*inputs), labels))  # list of ((input1, input2, ...), labels)

        # Split into train, test and validation
        train_data = [x for i, x in enumerate(all_samples) if self.has_label[i]]
        test_data = [x for i, x in enumerate(all_samples) if not self.has_label[i]]
        self.test_protein_names = self.protein_names[~self.has_label]
        if num_validation_samples > 0:
            random.Random(0).shuffle(train_data)
            val_data = train_data[:num_validation_samples]
            train_data = train_data[num_validation_samples:]

        # Make dataloaders
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=batch_collate_fn)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=batch_collate_fn)
        if num_validation_samples > 0:
            self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=batch_collate_fn)
        else:
            self.val_loader = None

        print('Dataset loaded:')
        print(f'\tTrain: {len(train_data)} samples')
        print(f'\tTest: {len(test_data)} samples')
        if num_validation_samples > 0:
            print(f'\tValidation: {len(val_data)} samples')


def batch_collate_fn(batch):
    seq_batch = []
    adj_batch = []
    node_features_batch = []
    edge_features_batch = []
    node_idx_batch = []
    edge_idx_batch = []
    labels_batch = []

    # Create tensors
    for i, ((seq, node_features, edge_features, adj), labels) in enumerate(batch):
        n = adj.shape[0]
        seq_batch.append(torch.LongTensor(seq))
        adj_batch.append(adj)
        node_features_batch.append(node_features)
        edge_features_batch.append(edge_features)
        node_idx_batch.append(torch.full((n,), i, dtype=torch.long))
        edge_idx_batch.append(torch.full((adj.nnz,), i, dtype=torch.long))
        labels_batch.append(labels)

    adj_batch = sp.block_diag(adj_batch)
    node_features_batch = np.vstack(node_features_batch)
    edge_features_batch = np.vstack(edge_features_batch)

    adj_batch = utils.sparse_mx_to_torch_sparse_tensor(adj_batch)
    node_features_batch = torch.FloatTensor(node_features_batch)
    edge_features_batch = torch.FloatTensor(edge_features_batch)
    node_idx_batch = torch.cat(node_idx_batch)
    edge_idx_batch = torch.cat(edge_idx_batch)
    labels_batch = torch.LongTensor(labels_batch)

    return (seq_batch, adj_batch, node_features_batch, edge_features_batch, node_idx_batch, edge_idx_batch), labels_batch
