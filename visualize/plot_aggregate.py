import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import RESULTS_DIR
from dataset import PATIENTS
from utils import make_dirs
from visualize.common import Result

removal_methods = ['peak']

# plot aggregate results
for artifact_removal_name in removal_methods:
    # load all results
    results = {}
    for patient_name in PATIENTS:
        results[patient_name] = Result(patient_name, artifact_removal_name)

    for classifier_name in results['aa'].classifiers:

        def plot_mean(method, name, ax, data_for_pandas):
            r = results['aa']

            data = []
            for n_csp in r.n_csp_components:
                data.append(np.array([
                    results[patient_name].get_results(method, n_csp, classifier_name)
                    for patient_name in results.keys()
                    if method in results[patient_name].csp_methods
                ]).reshape(-1))

                for patient_name in results.keys():
                    for r in results[patient_name].get_results(method, n_csp, classifier_name):
                        data_for_pandas.append((name, n_csp, r))

                assert data[-1].shape == (50,)

            # ax.plot(r.n_csp_components, [np.mean(d) for d in data], '-', label=name)
            # ax.errorbar(r.n_csp_components, [np.mean(d) for d in data], [np.std(d) for d in data],
            #             linestyle='-', marker='o', capsize=3, label=name)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # GEP
        ax = axes[0]
        data_for_pandas = []
        plot_mean('gep_no_checks', 'Python (eig)', ax, data_for_pandas)
        plot_mean('pca_gep', 'PCA → (Python (eigh), Matlab)', ax, data_for_pandas)
        plot_mean('pca_gep_no_checks', 'PCA → Python (eig)', ax, data_for_pandas)
        plot_mean('matlab_gep_no_check', 'Matlab', ax, data_for_pandas)
        plot_mean('pca_mne', 'PCA → MNE', ax, data_for_pandas)

        df = pd.DataFrame(data=data_for_pandas, columns=['method', 'n_csp', 'acc'])
        sns.pointplot(x='n_csp', y='acc', hue='method', data=df, ax=ax,
                      dodge=0.3, ci=95, capsize=0.2, scale=0.7, legend=False,
                      linestyles=['--', '-', '-', '--', '-'], errwidth=1)
        ymin1, ymax1 = ax.get_ylim()

        ax.set_xlabel('Number of CSP components')
        ax.set_ylabel('Accuracy')
        ax.set_title('CSP as Generalized eigenvalue problem, {}, {}'.format(artifact_removal_name, classifier_name))
        ax.legend()

        # GAP
        ax = axes[1]
        data_for_pandas = []
        plot_mean('gap_eig', 'Python (eig)', ax, data_for_pandas)
        plot_mean('gap_dr', 'PCA → ((Python (eig, eigh), Matlab)', ax, data_for_pandas)
        plot_mean('matlab_gap_no_check', 'Matlab', ax, data_for_pandas)
        plot_mean('fieldtrip', 'Fieldtrip, BBCI', ax, data_for_pandas)

        df = pd.DataFrame(data=data_for_pandas, columns=['method', 'n_csp', 'acc'])
        sns.pointplot(x='n_csp', y='acc', hue='method', data=df, ax=ax,
                      dodge=0.3, ci=95, capsize=0.2, scale=0.7, legend=False,
                      linestyles=['--', '-', '--', '-'], errwidth=1)
        ymin2, ymax2 = ax.get_ylim()

        ax.set_xlabel('Number of CSP components')
        ax.set_ylabel('Accuracy')
        ax.set_title('CSP as Geometric approach, {}, {}'.format(artifact_removal_name, classifier_name))
        ax.legend()

        # share the same Y scale
        axes[0].set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
        axes[1].set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))

        plt.tight_layout()
        # plt.show()
        output_path = os.path.join(RESULTS_DIR, 'plots', 'aggregate')
        make_dirs(output_path)
        plt.savefig(os.path.join(output_path, '{}_accuracy_{}.png'.format(artifact_removal_name, classifier_name)))
        plt.close(fig)
