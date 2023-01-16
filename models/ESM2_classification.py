import torch
from torch import nn
from transformers import EsmForMaskedLM, EsmConfig, EsmForSequenceClassification

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class ESM2Classification(BaseProteinModel):
    CREATE_SUBMISSION = True

    def __init__(self, num_node_features, num_classes):
        super(ESM2Classification, self).__init__()

        self.config = ConfigDict(
            name='EMS2_C',
            hidden_dim=320,

            epochs=200,
            batch_size=2,
            accumulate_grad_batches=32 // 4,
            num_validation_samples=500,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            # lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=pow(1e-4, 1 / 200)),
        )

        self.esm2_model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=num_classes)
        for param in self.esm2_model.esm.parameters():
            param.requires_grad = False

        self.output_dim = self.esm2_model.config.hidden_size * self.esm2_model.config.num_hidden_layers

        vocab = ("<cls>", "<pad>", "<eos>", "<unk>", "L",  "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q",
                 "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>")
        acid2vocab = {a: i for i, a in enumerate(vocab)}
        self.myId2EsmId = {i1: acid2vocab[a] for a, i1 in ProteinDataset.acid_to_index.items()}
        self.pad_token_id = acid2vocab["<pad>"]
        self.sos_token_id = acid2vocab["<cls>"]
        self.eos_token_id = acid2vocab["<eos>"]

        self.result = None

    def configure_optimizers(self, params=None):
        params = self.esm2_model.classifier.parameters()
        return super().configure_optimizers(params)

    def forward(self, sequences, graphs, return_embeddings=True, random_mask=False):
        # Pad sequences
        idx_n = 0
        x, attn_mask = [], []
        max_len = max([len(s) for s in sequences])
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

        self.result = self.esm2_model(input_ids=x, attention_mask=attn_mask)

        return self.result.logits
