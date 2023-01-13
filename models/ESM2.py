import torch
from torch import nn
from transformers import EsmForMaskedLM, EsmConfig

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel, ConfigDict


class ESM2(BaseProteinModel):
    CREATE_SUBMISSION = False
    experiment_name = 'masked_sequence_prediction'

    def __init__(self, num_node_features, num_classes):
        super(ESM2, self).__init__()

        self.config = ConfigDict(
            name='EMS2_small',
            hidden_dim=32,
            mask_rate=0.5,
            mask_error_rate=0.1,
            dropout=0.2,

            epochs=200,
            batch_size=16,
            num_validation_samples=100,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=2e-3, betas=(0.9, 0.98), weight_decay=1e-2),
            # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            # lr_scheduler_kwargs=dict(T_max=200, eta_min=1e-5),
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs=dict(gamma=pow(1e-2, 1/200)),
        )

        d = self.config.hidden_dim

        self.esm2_config = EsmConfig(
            hidden_size=d,
            intermediate_size=2*d,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=ProteinDataset.MAX_SEQ_LEN,
            position_embedding_type="absolute",  # TODO: try relative

            vocab_size=ProteinDataset.NUM_ACIDS + 2,
            mask_token_id=ProteinDataset.NUM_ACIDS,
            pad_token_id=ProteinDataset.NUM_ACIDS + 1,
        )
        self.pad_token_id = self.esm2_config.pad_token_id
        self.result = None

        self.esm2_model = EsmForMaskedLM(self.esm2_config)

        self.feature_projection = nn.Linear(num_node_features, d)

        self.mask_vector = nn.Parameter(torch.randn(d))
        # self.bos_vector = nn.Parameter(torch.randn(d))
        # self.eos_vector = nn.Parameter(torch.randn(d))

    def forward(self, sequences, graphs, return_embeddings=True, random_mask=False):
        acid_features = graphs.x

        # Pad sequences
        idx_n = 0
        x, y, attn_mask = [], [], []
        max_len = max([len(s) for s in sequences])
        for acid_ids in sequences:
            seq = acid_features[idx_n:idx_n+len(acid_ids)]  # TODO: + degree, angles, edge lengths, distance to center...
            idx_n += seq.shape[0]

            x.append(torch.cat([seq, torch.zeros(max_len - len(seq), seq.shape[1], device=seq.device)], dim=0))
            y.append(torch.cat([acid_ids, self.pad_token_id*torch.ones(max_len - len(acid_ids), device=seq.device)], dim=0))
            attn_mask.append((y[-1] != self.pad_token_id).float())
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0).long()
        attn_mask = torch.stack(attn_mask, dim=0)

        x = self.feature_projection(x)

        # Mask random inputs
        if random_mask:
            rand = torch.rand_like(x)
            mask_normal = rand < self.config.mask_rate
            mask_error = rand > 1 - self.config.mask_error_rate
            x = torch.where(mask_normal, self.mask_vector, x)  # fill with mask vector
            x = torch.where(mask_error, x[:, torch.randperm(x.shape[1])], x)  # fill with other sequence elements

        self.result = self.esm2_model(inputs_embeds=x, labels=y, attention_mask=attn_mask, encoder_attention_mask=attn_mask)

        return self.result.loss

    def training_step(self, batch, batch_idx, log=True, log_prefix='train_', prog_bar=False):
        sequences, graphs, labels = batch

        loss = self(sequences, graphs, return_embeddings=False, random_mask=True)

        if log:
            self.log(log_prefix + 'loss', loss, on_epoch=True, batch_size=labels.shape[0], prog_bar=prog_bar)
        else:
            loss = self(sequences, graphs)
        return loss
