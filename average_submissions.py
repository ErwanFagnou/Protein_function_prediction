import csv
import glob

import torch
from tqdm import tqdm

from save_predictions import write_solution_file
from utils import get_unique_file_path


def load_submission(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        return {row[0]: torch.tensor(list(map(float, row[1:])), dtype=torch.float32) for row in reader if row}


# submissions = [
#     "submissions/ESM2_150M+MHA(d=128,h=4)+query=random+20queries+dropout=0.2+labelSmoothing=0.05+1of5layers_submission_23-01-19_23-52-34.csv",
#     "submissions/ESM2_150M+MHA(d=128,h=4)+query=random+10queries+dropout=0.2+labelSmoothing=0.05+1of5layers_submission_23-01-19_23-12-36.csv",
# ]

# all files in "submissions/ensemble_final"
submissions = glob.glob("submissions/ensemble_final/*.csv")

aggregation = "mean"
# aggregation = "median"
# aggregation = "noOutliers"

num_outliers = 5

final_submission = {}
probas_dicts = [load_submission(submission) for submission in tqdm(submissions, desc="Loading submissions")]
for key in tqdm(probas_dicts[0].keys(), desc="Averaging submissions"):
    probas = torch.stack([probas_dict[key] for probas_dict in probas_dicts])
    if aggregation == "mean":
        probas = torch.mean(probas, dim=0)
    elif aggregation == "median":
        probas = torch.median(probas, dim=0)[0]
        probas = probas / torch.sum(probas)
    elif aggregation == "noOutliers":
        # Remove outliers
        probas = torch.sort(probas, dim=0)[0]
        probas = torch.mean(probas[num_outliers:-num_outliers], dim=0)
        probas = probas / torch.sum(probas)
    final_submission[key] = probas
print(final_submission['11as'])

# file_name = "combined_submissions"
file_name = f"combined_ensemble_final_n={len(submissions)}" #+ f"_{aggregation}5"
file_path = get_unique_file_path('submissions', file_name, 'csv')
write_solution_file(file_path, list(final_submission.keys()), torch.stack(list(final_submission.values())))

print(f'Done!\n  -> Predictions saved to {file_path}')
