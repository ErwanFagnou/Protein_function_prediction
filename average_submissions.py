import csv

import torch

from save_predictions import write_solution_file
from utils import get_unique_file_path


def load_submission(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        return {row[0]: torch.tensor(list(map(float, row[1:])), dtype=torch.float32) for row in reader if row}


submissions = [
    "submissions/ESM2_8M+MHA(d=128,h=4)+query=random+10queries+dropout=0.5+labelSmoothing=0.1_submission_23-01-19_17-51-31.csv",
    "submissions/ESM2_8M+MHA(d=256,h=16)+query=random+10queries+dropout=0.5+labelSmoothing=0.2_submission_23-01-19_19-54-22.csv",
    "submissions/ESM2_8M+MHA(d=128,h=16)+query=random+10queries+dropout=0.5+labelSmoothing=0.1_submission_23-01-19_20-30-41.csv",
]

final_submission = {}
probas_dicts = [load_submission(submission) for submission in submissions]
for key in probas_dicts[0].keys():
    probas = torch.stack([probas_dict[key] for probas_dict in probas_dicts])
    probas = torch.mean(probas, dim=0)
    final_submission[key] = probas
print(final_submission['11as'])

file_path = get_unique_file_path('submissions', 'combined_submissions', 'csv')
write_solution_file(file_path, list(final_submission.keys()), torch.stack(list(final_submission.values())))

print(f'Done!\n  -> Predictions saved to {file_path}')
