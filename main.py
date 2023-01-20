import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ProteinDataset
from models.ESM2_custom import ESM2Custom
from models.ESM2_classification import ESM2Classification
from models.ESM2_pretrained import ESM2Pretrained
from models.GNN import GNN
from models.LSTM_encoder import LSTMEncoder
from models.MultiHeadAttention import MultiHeadAttention
from save_predictions import save_predictions
from train import train
from utils import get_unique_file_path


# # Better error message with cuda
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


TRAIN_MODEL = True


def get_pretrained_encoder():
    ...
    # return LSTMEncoder(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )
    # return ESM2Custom(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )
    return ESM2Pretrained(ProteinDataset.NUM_NODE_FEATURES, ProteinDataset.NUM_CLASSES)


def get_model(num_node_features):
    ...
    #return GNN(
    #    num_node_features=num_node_features,
    #    num_edge_features=ProteinDataset.NUM_EDGE_FEATURES,
    #    num_classes=ProteinDataset.NUM_CLASSES,
    #)
    # return ESM2Custom(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )
    # return ESM2Classification(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )
    return MultiHeadAttention(
        num_node_features=num_node_features,
        num_edge_features=ProteinDataset.NUM_EDGE_FEATURES,
        num_classes=ProteinDataset.NUM_CLASSES,
    )


if __name__ == '__main__':
    device_id = 1
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Create models (pretrained encoder and classification model)
    pretrained_seq_encoder = get_pretrained_encoder().to(device)  # Pretrained encoder
    if pretrained_seq_encoder is not None:
        print("Pretrained sequence encoder:", pretrained_seq_encoder)

    num_node_features = ProteinDataset.NUM_NODE_FEATURES if pretrained_seq_encoder is None else pretrained_seq_encoder.output_dim
    model = get_model(num_node_features)
    # model.load_state_dict(torch.load('trained_models/ESM2_650M+MHA(d=128,h=4)+query=random+20queries+dropout=0.2+labelSmoothing=0.05+1of10layers_23-01-20_02-32-34.pt'))
    model = model.to(device)  # Classification model

    protein_dataset = ProteinDataset(
        batch_size=model.config.batch_size,
        num_validation_samples=model.config.num_validation_samples,
        pretrained_seq_encoder=pretrained_seq_encoder,
        transforms=model.transforms if hasattr(model, 'transforms') else None,
        pca_dim=model.PCA_DIM,
    )

    # Train
    if TRAIN_MODEL:
        train(model, protein_dataset, device, device_id, pretrained_seq_encoder=pretrained_seq_encoder)

        # Save model
        model_path = get_unique_file_path("trained_models", f"{model.config.name}", "pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Save predictions
    if model.CREATE_SUBMISSION:
        save_predictions(model, protein_dataset)
