import os

from joblib import load
from matplotlib import pyplot as plt

from config import (RESULTS_DIR)
from dataset import PATIENTS
from utils import make_dirs

patient_name = PATIENTS[0]
# load results
results = load(os.path.join(RESULTS_DIR, 'evaluation', '{}.joblib'.format(patient_name)))

output_path = os.path.join(RESULTS_DIR, 'plots')
make_dirs(output_path)

csp_methods = list(results[0].keys())
n_csp_components = list(results[0][csp_methods[0]].keys())
classifiers = list(results[0][csp_methods[0]][n_csp_components[0]]['classifier'].keys())

for classifier_name in classifiers:
    data = []
    labels = []
    for n_csp in n_csp_components:
        data += [[r[csp_method][n_csp]['classifier'][classifier_name]['score_test'] for r in results] for csp_method in csp_methods]
        labels += ['{} {}'.format(n_csp, csp_method) for csp_method in csp_methods]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, showmeans=True, labels=labels)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.title('{}'.format(classifier_name))
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_path, '{}.png'.format(classifier_name)))
    plt.close()
