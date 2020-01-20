"""
Compare methods whether they are statistically different
and generate CSV with results.
"""

import csv
import os

import numpy as np
from scipy.stats import wilcoxon

from config import RESULTS_DIR
from dataset import PATIENTS
from utils import make_dirs
from visualize.common import Result


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

removal_methods = ['peak']
for artifact_removal_name in removal_methods:
    for classifier_name in ['lda', 'svm']:

        results = {}
        for patient_name in PATIENTS:
            results[patient_name] = Result(patient_name, artifact_removal_name)

        def get_diff_results(method, compare_method, alternative='greater'):
            row = ['{} VS {}'.format(method, compare_method)]
            for n_csp in results['aa'].n_csp_components:
                # if method != 'gap_eig':
                acc_method_a = np.array([r.get_results(method, n_csp, classifier_name) for r in results.values()]).reshape(-1)
                acc_method_b = np.array([r.get_results(compare_method, n_csp, classifier_name) for r in results.values()]).reshape(-1)
                # else:
                #     acc_method_a, acc_method_b = [], []
                #     for r in results.values():
                #         if method in r.csp_methods:
                #             acc_method_a.append(r.get_results(method, n_csp, classifier_name))
                #             acc_method_b.append(r.get_results(compare_method, n_csp, classifier_name))
                #     acc_method_a = np.array(acc_method_a).reshape(-1)
                #     acc_method_b = np.array(acc_method_b).reshape(-1)

                diff_methods = acc_method_b - acc_method_a
                stats = wilcoxon(acc_method_b, acc_method_a, alternative=alternative, zero_method='pratt')

                row.append('{0:.3f}'.format(diff_methods.mean() * 100))
                row.append('{0:.3f}'.format(diff_methods.std() * 100))
                row.append('{0:.3f}'.format(np.median(diff_methods) * 100))
                row.append('{0:.5f}'.format(stats.pvalue))
                row.append('{0:.5f}'.format(stats.statistic))
                row.append(p_value_format(stats.pvalue))
            return row

        with open(os.path.join(output_path, '{}_{}.csv'.format(artifact_removal_name, classifier_name)), 'w') as f:
            writer = csv.writer(f)

            # header
            row = ['Methods']
            for n_csp in results['aa'].n_csp_components:
                row.append('{} mean'.format(n_csp))
                row.append('{} std'.format(n_csp))
                row.append('{} med'.format(n_csp))
                row.append('{} pvalue'.format(n_csp))
                row.append('{} statistic'.format(n_csp))
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

            # two-sided test between Python and Matlab
            # unprotected
            writer.writerow(get_diff_results('matlab_gep_no_check', 'gep_no_checks', alternative='two-sided'))
            writer.writerow(get_diff_results('matlab_gap_no_check', 'gap_eig', alternative='two-sided'))

            # protected
            writer.writerow(get_diff_results('pca_matlab_gep_no_check', 'pca_gep_no_checks', alternative='two-sided'))

            # protected Python eigh vs eig
            writer.writerow(get_diff_results('pca_gep', 'pca_gep_no_checks', alternative='two-sided'))
