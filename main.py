import torch

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
NUM_MODELS_TRAIN = 1000

# submissions_dir = "submissions"
submissions_dir = "submissions/ensemble_final"


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


def main_loop(model_num: int, model=None):
    print("\n", "#" * 30, f"Model {model_num}", "#" * 30)

    if model is None:
        model = get_model(num_node_features).to(device)
    model.config.name = f'{model_num}_{model.config.name}'

    if TRAIN_MODEL:
        train(model, protein_dataset, device, device_id, pretrained_seq_encoder=pretrained_seq_encoder)

        # Save model
        model_path = get_unique_file_path("trained_models", f"{model.config.name}", "pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Save predictions
    if model.CREATE_SUBMISSION:
        save_predictions(model, protein_dataset, file_dir=submissions_dir)


if __name__ == '__main__':
    device_id = 1
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Create models (pretrained encoder and classification model)
    pretrained_seq_encoder = get_pretrained_encoder().to(device)  # Pretrained encoder
    if pretrained_seq_encoder is not None:
        print("Pretrained sequence encoder:", pretrained_seq_encoder)

    num_node_features = ProteinDataset.NUM_NODE_FEATURES if pretrained_seq_encoder is None else pretrained_seq_encoder.output_dim
    first_model = get_model(num_node_features).to(device)  # Classification model

    protein_dataset = ProteinDataset(
        batch_size=first_model.config.batch_size,
        num_validation_samples=first_model.config.num_validation_samples,
        pretrained_seq_encoder=pretrained_seq_encoder,
        transforms=first_model.transforms if hasattr(first_model, 'transforms') else None,
        pca_dim=first_model.PCA_DIM,
    )

    # Train models
    for model_num in range(NUM_MODELS_TRAIN):
        main_loop(model_num, model=first_model if model_num == 0 else None)
