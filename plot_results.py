import csv
import os
from glob import glob

import numpy as np
import seaborn as sns
from joblib import load
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon

from config import RESULTS_DIR
from dataset import PATIENTS
from utils import make_dirs


class Result(object):
    """
    Load results from all methods for particular patient and artifact removal method.
    """
    def __init__(self, patient_name, artifact_removal_name):
        self.patient_name = patient_name
        self.artifact_removal_name = artifact_removal_name

        csp_methods = []
        results = {}
        for file_name in glob(os.path.join(
                RESULTS_DIR, 'evaluation', patient_name, artifact_removal_name + '_*.joblib')):
            method_name = os.path.splitext(os.path.basename(file_name))[0][len(artifact_removal_name) + 1:]
            csp_methods.append(method_name)
            results[method_name] = load(file_name)['results']

        self.results = results
        self.csp_methods = csp_methods
        self.n_csp_components = list(results[csp_methods[0]][0].keys())
        self.classifiers = list(results[csp_methods[0]][0][self.n_csp_components[0]]['classifier'].keys())

        self.output_path = os.path.join(RESULTS_DIR, 'plots', patient_name, artifact_removal_name)
        make_dirs(self.output_path)

    def get_results(self, csp_method, n_csp, classifier_name, key='score_test'):
        return [r[n_csp]['classifier'][classifier_name][key] for r in self.results[csp_method]]

    def compute_stats(self, method_a, method_b, n_csp, classifier_name, alternative='greater'):
        a = np.array(self.get_results(method_a, n_csp, classifier_name))
        b = np.array(self.get_results(method_b, n_csp, classifier_name))
        try:
            return wilcoxon(a, b, alternative=alternative).pvalue
        except ValueError:
            # for a == b
            return 1.0
            # return np.NAN


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
    plt.title('{}, Stratified 5-fold cross-validation'.format(classifier_name))
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



#%%
# test whether 2 methods are the same or not
method_a = 'pca_mne'
method_b = 'pca_mne'
for patient_name in PATIENTS:
    for artifact_removal_name in removal_methods:
        try:
            results = Result(patient_name, artifact_removal_name)
            for n_csp in results.n_csp_components:
                for classifier_name in results.classifiers:
                    pvalue = results.compute_stats(method_a, method_b, n_csp, classifier_name, alternative='two-sided')
                    if pvalue != 1.0:
                        print(pvalue, patient_name, artifact_removal_name, n_csp, classifier_name)
                        # assert pvalue == 1.0
        except:
            print(patient_name, artifact_removal_name)


#%%
# generate table

def p_value_format(p_value):
    if p_value <= 0.0001:
        return '****'
    elif p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return '-'


output_path = os.path.join(RESULTS_DIR, 'diff')
make_dirs(output_path)

for artifact_removal_name in removal_methods:
    for classifier_name in ['lda', 'svm']:

        results = {}
        for patient_name in PATIENTS:
            results[patient_name] = Result(patient_name, artifact_removal_name)

        def get_diff_results(method, compare_method):
            row = ['{} VS {}'.format(method, compare_method)]
            for n_csp in results['aa'].n_csp_components:
                if method != 'gap_eig':
                    acc_method_a = np.array([r.get_results(method, n_csp, classifier_name) for r in results.values()]).reshape(-1)
                    acc_method_b = np.array([r.get_results(compare_method, n_csp, classifier_name) for r in results.values()]).reshape(-1)
                else:
                    acc_method_a, acc_method_b = [], []
                    for r in results.values():
                        if method in r.csp_methods:
                            acc_method_a.append(r.get_results(method, n_csp, classifier_name))
                            acc_method_b.append(r.get_results(compare_method, n_csp, classifier_name))
                    acc_method_a = np.array(acc_method_a).reshape(-1)
                    acc_method_b = np.array(acc_method_b).reshape(-1)

                diff_methods = acc_method_b - acc_method_a
                p_value = wilcoxon(acc_method_b, acc_method_a, alternative='greater').pvalue

                row.append('{0:.3f}'.format(diff_methods.mean()))
                row.append('{0:.3f}'.format(diff_methods.std()))
                row.append('{0:.3f}'.format(np.median(diff_methods)))
                # row.append('{0:.5f}'.format(p_value))
                row.append(p_value_format(p_value))
            return row

        with open(os.path.join(output_path, '{}_{}.csv'.format(artifact_removal_name, classifier_name)), 'w') as f:
            writer = csv.writer(f)

            # header
            row = ['Methods']
            for n_csp in results['aa'].n_csp_components:
                row.append('{} mean'.format(n_csp))
                row.append('{} std'.format(n_csp))
                row.append('{} med'.format(n_csp))
                # row.append('{} p-value'.format(n_csp))
                row.append(' ')
            writer.writerow(row)

            # GEP
            writer.writerow(get_diff_results('matlab_gep_no_check', 'pca_matlab_gep_no_check'))
            writer.writerow(get_diff_results('gep_no_checks', 'pca_gep_no_checks'))
            writer.writerow(get_diff_results('gep_no_checks', 'pca_gep'))
            # GAP
            writer.writerow(get_diff_results('matlab_gap_no_check', 'pca_matlab_gap_no_check'))
            writer.writerow(get_diff_results('gap_eig', 'pca_gap_eig'))
            writer.writerow(get_diff_results('gap_eig', 'gap_dr'))  # should be same as above

#%%
    #%%
    # def show_over_n_csp(method_a, method_b, classifier_name):
    #     print('###')
    #     print('{} VS {}'.format(method_a, method_b))
    #     for n_csp in n_csp_components:
    #         stats = compute_stats(method_a, method_b, n_csp, classifier_name)
    #         print('#CSP {}: {}'.format(n_csp, stats))
    #     print('###')
    #
    #
    # # CSP as generalized eigenvalue problem
    #
    # # check whether unprotected GEP is different to protected GEP
    # show_over_n_csp('gep_no_checks', 'pca_gep', 'lda')
    # show_over_n_csp('gep_no_checks', 'pca_gep_no_checks', 'lda')
    #
    # # different protected GEP should be same
    # show_over_n_csp('pca_gep', 'pca_gep_no_checks', 'lda')
    #
    # # unprotected GEP vs protected GAP
    # show_over_n_csp('gep_no_checks', 'gap_dr', 'lda')


    # CSP as geometric approach

    # check whether  unprotected GAP is different
    # show_over_n_csp('gap_eig', 'gap_dr', 'lda')
    # show_over_n_csp('gap_eig', 'pca_gap_dr', 'lda')
    # show_over_n_csp('gap_eig', 'pca_gap_eig_dr', 'lda')
    # show_over_n_csp('gap_eig', 'pca_gap_eig', 'lda')
    #
    #
    # # different protected GAP should be same
    # show_over_n_csp('gap_eig_dr', 'pca_gap_eig_dr', 'lda')
    # show_over_n_csp('gap_dr', 'pca_gap_eig_dr', 'lda')
    #
    # show_over_n_csp('gap_dr', 'gap_eig_dr', 'lda')
    #
    # show_over_n_csp('pca_gap_eig_dr', 'pca_gap_eig', 'lda')

