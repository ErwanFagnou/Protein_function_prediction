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


TRAIN_MODEL = True


def get_pretrained_encoder():
    ...
    return SequencePredictor(
        num_node_features=ProteinDataset.NUM_NODE_FEATURES,
        num_classes=ProteinDataset.NUM_CLASSES,
    )
    # return ESM2(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )


def get_model(num_node_features):
    return GNN(
        num_node_features=num_node_features,
        num_edge_features=ProteinDataset.NUM_EDGE_FEATURES,
        num_classes=ProteinDataset.NUM_CLASSES,
    )
    # return ESM2(
    #     num_node_features=ProteinDataset.NUM_NODE_FEATURES,
    #     num_classes=ProteinDataset.NUM_CLASSES,
    # )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_seq_encoder = get_pretrained_encoder()
    pretrained_seq_encoder.load_state_dict(torch.load('trained_models/LSTM_256_23-01-13_17-47-26.pt'))
    # pretrained_seq_encoder.load_state_dict(torch.load('trained_models/EMS2_small_23-01-13_21-18-41.pt'))
    # pretrained_seq_encoder.load_state_dict(torch.load('trained_models/EMS2_large_23-01-14_00-45-00.pt'))
    # pretrained_seq_encoder.load_state_dict(torch.load('trained_models/EMS2_large_23-01-14_03-35-00.pt'))

    # pretrained_seq_encoder.load_state_dict(torch.load('checkpoints/ALTeGraD Kaggle challenge-7761zfzr/epoch=179-step=54000-val_loss=0.0000.ckpt', map_location=device)['state_dict'])
    # torch.save(pretrained_seq_encoder.state_dict(), 'trained_models/EMS2_large_23-01-14_03-35-00.pt')

    num_node_features = ProteinDataset.NUM_NODE_FEATURES if pretrained_seq_encoder is None else pretrained_seq_encoder.output_dim
    model = get_model(num_node_features)
    # model.load_state_dict(torch.load('trained_models\AttentiveFP+LSTM_23-01-13_22-53-21.pt'))
    # model.load_state_dict(torch.load('checkpoints/ALTeGraD Kaggle challenge-3l05qzgu/epoch=77-step=11700-val_loss=1.4649.ckpt', map_location=device)['state_dict'])

    # Get dataset and train
    protein_dataset = train(model, device, do_train=TRAIN_MODEL, pretrained_seq_encoder=pretrained_seq_encoder)

    # Save model
    if TRAIN_MODEL:
        model_path = get_unique_file_path("trained_models", f"{model.config.name}", "pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Save predictions
    if model.CREATE_SUBMISSION:
        save_predictions(model, protein_dataset)

