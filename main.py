import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ProteinDataset
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

    # model = GNN(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_edge_features=ProteinDataset.NUM_EDGE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )

    model = SequencePredictor(
        num_classes=ProteinDataset.NUM_CLASSES,
        num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    )

    model = model.to(device)

    # Get dataset and train
    protein_dataset = train(model, device)

    # Save model
    model_path = get_unique_file_path("trained_models", f"{model.config.name}", "pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save predictions
    if model.CREATE_SUBMISSION:
        save_predictions(model, protein_dataset)

