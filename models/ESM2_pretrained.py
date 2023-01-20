import torch
from torch import nn
from transformers import EsmModel

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class ESM2Pretrained(BaseProteinModel):
    CREATE_SUBMISSION = False

    def __init__(self, num_node_features, num_classes):
        super(ESM2Pretrained, self).__init__()

        self.config = ConfigDict(
            name='EMS2_pretrained',

            batch_size=1,
            num_validation_samples=0,
        )

        # self.esm2_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # self.esm2_model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        # self.esm2_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.esm2_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        for param in self.esm2_model.parameters():
            param.requires_grad = False

        self.config.hidden_dim = self.esm2_model.config.hidden_size
        self.config.num_layers = self.esm2_model.config.num_hidden_layers
        self.output_dim = self.config.hidden_dim * self.config.num_layers

        vocab = ("<cls>", "<pad>", "<eos>", "<unk>", "L",  "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q",
                 "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>")
        acid2vocab = {a: i for i, a in enumerate(vocab)}
        self.myId2EsmId = {i1: acid2vocab[a] for a, i1 in ProteinDataset.acid_to_index.items()}
        self.pad_token_id = acid2vocab["<pad>"]
        self.sos_token_id = acid2vocab["<cls>"]
        self.eos_token_id = acid2vocab["<eos>"]

        self.result = None

    def forward(self, sequences, graphs):
        # Pad sequences
        idx_n = 0
        x, attn_mask = [], []
        lengths = torch.tensor([len(s) for s in sequences])
        max_len = lengths.max().item()
        for acid_ids in sequences:
            seq = torch.tensor([self.myId2EsmId[i.item()] for i in acid_ids], device=self.device, dtype=torch.long)
            idx_n += seq.shape[0]

            # add sos, eos, pad tokens
            seq = torch.cat([torch.tensor([self.sos_token_id], device=self.device, dtype=torch.long),
                             seq,
                             torch.tensor([self.eos_token_id], device=self.device, dtype=torch.long),
                             torch.tensor([self.pad_token_id] * (max_len - seq.shape[0]), device=self.device, dtype=torch.long)])
            x.append(seq)
            attn_mask.append((x[-1] != self.pad_token_id).float())
        x = torch.stack(x, dim=0).long()
        attn_mask = torch.stack(attn_mask, dim=0)

        # Get embeddings
        self.result = self.esm2_model(input_ids=x, attention_mask=attn_mask, output_hidden_states=True)

        # all hidden states
        # x = torch.cat(self.result.hidden_states[-self.config.num_layers+1:], dim=-1)

        # trying to do with only the last layer
        # x = self.result.hidden_states[-1]

        # the layer in the middle
        # x = self.result.hidden_states[len(self.result.hidden_states) // 2]

        # one layer every 5
        # x = torch.cat(self.result.hidden_states[3::5], dim=-1)

        # one layer every 10  [2, 12, 22, 32]
        # x = torch.cat(self.result.hidden_states[2::10], dim=-1)

        # custom indices
        ids = [32]  # penultimate
        x = torch.cat([self.result.hidden_states[i] for i in ids], dim=-1)

        # hack: replacing first and last embeddings with the embeddings of <cls> and <eos>
        x[:, 1] = x[:, 0]
        x[torch.arange(x.shape[0]), lengths] = x[torch.arange(x.shape[0]), lengths+1]

        x = x[:, 1:]  # remove <cls> token
        return x
