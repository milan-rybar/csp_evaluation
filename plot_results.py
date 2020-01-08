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

csp_methods = list(results.keys())
n_csp_components = list(results[csp_methods[0]][0]['n_csp'].keys())
classifiers = list(results[csp_methods[0]][0]['n_csp'][n_csp_components[0]].keys())

for classifier_name in classifiers:
    data = []
    labels = []
    for n_csp in n_csp_components:
        data += [[result['n_csp'][n_csp][classifier_name]['score_test'] for result in results[csp_method]] for csp_method in csp_methods]
        labels += ['{} {}'.format(n_csp, csp_method) for csp_method in csp_methods]

    plt.figure()
    plt.boxplot(data, showmeans=True, labels=labels)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.title('{}'.format(classifier_name))
    # plt.show()
    plt.savefig(os.path.join(output_path, '{}.png'.format(classifier_name)))
    plt.close()
