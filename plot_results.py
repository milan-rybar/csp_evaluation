import itertools
import os

import numpy as np
import seaborn as sns
from joblib import load
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

from config import RESULTS_DIR
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


def get_results(csp_method, n_csp, classifier_name, key='score_test'):
    return [r[csp_method][n_csp]['classifier'][classifier_name][key] for r in results]


for classifier_name in classifiers:
    data = []
    labels = []
    for n_csp in n_csp_components:
        data += [get_results(csp_method, n_csp, classifier_name) for csp_method in csp_methods]
        labels += ['{} {}'.format(n_csp, csp_method) for csp_method in csp_methods]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, showmeans=True, labels=labels)
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=90)
    plt.title('{}, Stratified 5-fold cross-validation'.format(classifier_name))
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_path, '{}.png'.format(classifier_name)))
    plt.close()


def compute_stats(method_a, method_b, n_csp, classifier_name):
    a = np.array(list(itertools.chain(*get_results(method_a, n_csp, classifier_name, key='correct_test'))), dtype=np.double)
    b = np.array(list(itertools.chain(*get_results(method_b, n_csp, classifier_name, key='correct_test'))), dtype=np.double)
    return ttest_rel(a, b)


# compute and plot statistics
for classifier_name in classifiers:
    nrows = len(n_csp_components)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 10 * nrows))
    for n_csp, ax in zip(n_csp_components, axes):
        statistics = np.zeros((len(csp_methods), len(csp_methods)))

        for idx_a in range(len(csp_methods)):
            for idx_b in range(idx_a, len(csp_methods)):
                stats = compute_stats(csp_methods[idx_a], csp_methods[idx_b], n_csp, classifier_name)
                print('{} VS {}: {}'.format(csp_methods[idx_a], csp_methods[idx_b], stats))
                statistics[idx_a, idx_b] = stats.pvalue

        # Current version of matplotlib broke heatmaps. Downgrade the package to 3.1.0
        # pip install matplotlib==3.1.0
        sns.heatmap(statistics, annot=True, fmt='.6f', linewidths=0.5,
                    xticklabels=csp_methods, yticklabels=csp_methods,
                    square=True, #mask=statistics == 0,
                    cbar=False, cmap=None, ax=ax)
        ax.set_title('#CSP: {}, {}'.format(n_csp, classifier_name))
        ax.yaxis.set_tick_params(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_path, 'statistics_{}.png'.format(classifier_name)))
    plt.close()


#%%
def show_over_n_csp(method_a, method_b, classifier_name):
    print('###')
    print('{} VS {}'.format(method_a, method_b))
    for n_csp in n_csp_components:
        stats = compute_stats(method_a, method_b, n_csp, classifier_name)
        print('#CSP {}: {}'.format(n_csp, stats))
    print('###')


# CSP as generalized eigenvalue problem

# check whether unprotected GEP is different to protected GEP
show_over_n_csp('gep_no_checks', 'pca_gep', 'lda')
show_over_n_csp('gep_no_checks', 'pca_gep_no_checks', 'lda')

# different protected GEP should be same
show_over_n_csp('pca_gep', 'pca_gep_no_checks', 'lda')

# unprotected GEP vs protected GAP
show_over_n_csp('gep_no_checks', 'gap_dr', 'lda')


# CSP as geometric approach

# check whether  unprotected GAP is different
show_over_n_csp('gap_eig', 'gap_dr', 'lda')
show_over_n_csp('gap_eig', 'pca_gap_dr', 'lda')
show_over_n_csp('gap_eig', 'pca_gap_eig_dr', 'lda')
show_over_n_csp('gap_eig', 'pca_gap_eig', 'lda')

