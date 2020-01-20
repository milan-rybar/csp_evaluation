"""
Plot results for all methods.
"""

import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from dataset import PATIENTS
from visualize.common import Result


def plot_all_boxplots(results, classifier_name):
    data = []
    labels = []
    for n_csp in results.n_csp_components:
        data += [results.get_results(csp_method, n_csp, classifier_name) for csp_method in results.csp_methods]
        labels += ['{} {}'.format(n_csp, csp_method) for csp_method in results.csp_methods]

    plt.figure(figsize=(30, 6))
    plt.boxplot(data, showmeans=True, labels=labels)
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=90)
    plt.title('{}, Stratified 10-fold cross-validation'.format(classifier_name))
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(results.output_path, '{}.png'.format(classifier_name)))
    plt.close()


def plot_all_statistics(results, classifier_name):
    nrows = len(results.n_csp_components)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 15 * nrows))
    for n_csp, ax in zip(results.n_csp_components, axes):
        statistics = np.zeros((len(results.csp_methods), len(results.csp_methods)))

        for idx_a in range(len(results.csp_methods)):
            for idx_b in range(len(results.csp_methods)):  # matrix is not symmetric; using one-sided test
                stats = results.compute_stats(results.csp_methods[idx_a], results.csp_methods[idx_b], n_csp, classifier_name)
                print('{} VS {}: {}'.format(results.csp_methods[idx_a], results.csp_methods[idx_b], stats))
                statistics[idx_a, idx_b] = stats

        # Current version of matplotlib (3.1.1) broke heatmaps. Downgrade the package to 3.1.0
        # pip install matplotlib==3.1.0
        sns.heatmap(statistics, annot=True, fmt='.5f', linewidths=0.5,
                    xticklabels=results.csp_methods, yticklabels=results.csp_methods,
                    square=True,  # mask=statistics == 0,
                    cbar=False, cmap=None, ax=ax)
        ax.set_title('#CSP: {}, {}'.format(n_csp, classifier_name))
        ax.yaxis.set_tick_params(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(results.output_path, 'statistics_{}.png'.format(classifier_name)))
    plt.close()


def plot_selection(results, classifier_name):
    def plot_mean(method, name, ax):
        data = [results.get_results(method, n_csp, classifier_name) for n_csp in results.n_csp_components]
        # ax.plot(n_csp_components, [np.mean(d) for d in data], '-', label=name)
        ax.errorbar(results.n_csp_components, [np.mean(d) for d in data], [np.std(d) for d in data],
                    linestyle='-', marker='o', capsize=3, label=name)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax = axes[0]
    plot_mean('gep_no_checks', 'Python (eig)', ax)
    plot_mean('pca_gep', 'PCA → (Python (eigh), Matlab)', ax)
    plot_mean('pca_gep_no_checks', 'PCA → Python (eig)', ax)
    plot_mean('matlab_gep_no_check', 'Matlab', ax)
    plot_mean('pca_mne', 'PCA -> MNE', ax)

    ax.set_xlabel('Number of CSP components')
    ax.set_ylabel('Accuracy')
    ax.set_title('GEP, {}, {}, {}'.format(artifact_removal_name, patient_name, classifier_name))
    ax.legend()

    ax = axes[1]
    if 'gap_eig' in results.csp_methods:
        plot_mean('gap_eig', 'Python (eig)', ax)
    plot_mean('gap_dr', 'PCA → ((Python (eig, eigh), Matlab)', ax)
    plot_mean('matlab_gap_no_check', 'Matlab', ax)
    plot_mean('fieldtrip', 'Fieldtrip, BBCI', ax)

    ax.set_xlabel('Number of CSP components')
    ax.set_ylabel('Accuracy')
    ax.set_title('GAP, {}, {}, {}'.format(artifact_removal_name, patient_name, classifier_name))
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results.output_path, '{}_accuracy_{}.png'.format(artifact_removal_name, classifier_name)))
    plt.close()


removal_methods = ['manual', 'peak']

for patient_name in PATIENTS:
    for artifact_removal_name in removal_methods:
        results = Result(patient_name, artifact_removal_name)

        # plot test fold accuracies for each method
        for classifier_name in results.classifiers:
            plot_all_boxplots(results, classifier_name)

        # plot statistics of each pair of method
        for classifier_name in results.classifiers:
            plot_all_statistics(results, classifier_name)

        # plot test fold accuracies of main methods in interest
        for classifier_name in results.classifiers:
            plot_selection(results, classifier_name)
