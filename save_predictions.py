import csv

import torch
from tqdm import tqdm
from datetime import datetime

from dataset import ProteinDataset
from models.BaseProteinModel import BaseProteinModel
from utils import get_unique_file_path


@torch.no_grad()
def save_predictions(model: BaseProteinModel, dataset: ProteinDataset, file_dir="submissions", file_name=None):
    # Get predictions
    model.eval()
    label_probas = []
    for batch in tqdm(dataset.test_loader, desc='Generating predictions'):
        sequences, graphs, _ = batch
        # print(sequences, graphs)
        sequences = [s.to(model.device) for s in sequences]
        graphs = graphs.to(model.device)
        label_probas.append(model.predict_probabilities(sequences, graphs).detach().cpu())
    label_probas = torch.cat(label_probas, dim=0)

    if file_name is None:
        file_name = f'{model.config.name}_submission'
    file_path = get_unique_file_path(file_dir, file_name, 'csv')
    write_solution_file(file_path, dataset.test_protein_names, label_probas)
    print(f'  -> Predictions saved to {file_path}')


def write_solution_file(file, protein_names, label_probas):
    assert len(protein_names) == len(label_probas), f'Different number of proteins and predictions'
    assert len(protein_names) == len(label_probas), "Number of proteins and number of predictions do not match"
    assert len(label_probas[0]) == ProteinDataset.NUM_CLASSES, "Number of classes does not match"
    assert torch.all(torch.logical_and(label_probas >= 0, label_probas <= 1)), "Predictions are not valid probabilities"
    assert torch.all(torch.isclose(torch.sum(label_probas, dim=1), torch.ones(len(label_probas)))), "Predictions do not sum to 1"

    if not file.endswith(".csv"):
        file += ".csv"

    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = ['name'] + ['class' + str(i) for i in range(18)]
        writer.writerow(lst)
        for name, probas in zip(protein_names, label_probas):
            writer.writerow([name] + list(probas.numpy()))
