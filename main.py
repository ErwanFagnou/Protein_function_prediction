import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ProteinDataset
from models.ESM2 import ESM2
from models.GNN import GNN
from models.SequencePredictor import SequencePredictor
from save_predictions import save_predictions
from train import train
from utils import get_unique_file_path

# CUDA_LAUNCH_BLOCKING=1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained_seq_encoder = None

    pretrained_seq_encoder = SequencePredictor(
        num_node_features=ProteinDataset.NUM_NODE_FEATURES,
        num_classes=ProteinDataset.NUM_CLASSES,
    )
    pretrained_seq_encoder.load_state_dict(torch.load('trained_models/LSTM_256_23-01-13_17-47-26.pt'))
    model = GNN(
        num_node_features=pretrained_seq_encoder.output_dim,
        num_edge_features=ProteinDataset.NUM_EDGE_FEATURES,
        num_classes=ProteinDataset.NUM_CLASSES,
    )


    # model = GNN(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_edge_features=ProteinDataset.NUM_EDGE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )

    # model = SequencePredictor(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )

    # model = ESM2(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )

    model = model.to(device)
    if pretrained_seq_encoder is not None:
        print("Pretrained sequence encoder:", pretrained_seq_encoder)
        pretrained_seq_encoder = pretrained_seq_encoder.to(device)

    # Get dataset and train
    protein_dataset = train(model, device, pretrained_seq_encoder=pretrained_seq_encoder)

    # Save model
    model_path = get_unique_file_path("trained_models", f"{model.config.name}", "pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save predictions
    if model.CREATE_SUBMISSION:
        save_predictions(model, protein_dataset)

