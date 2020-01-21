import numpy as np
from scipy.stats import friedmanchisquare

from dataset import PATIENTS
from visualize.common import Result

removal_methods = ['peak']

#%%
# test difference between groups
for artifact_removal_name in removal_methods:
    # load all results
    results = {}
    for patient_name in PATIENTS:
        results[patient_name] = Result(patient_name, artifact_removal_name)

    for classifier_name in ['lda']:

        def get_data(method, n_csp):
            return np.array([
                results[patient_name].get_results(method, n_csp, classifier_name)
                for patient_name in results.keys()
            ]).reshape(-1)

        # GEP Python
        for n_csp in results['aa'].n_csp_components:
            d = [
                get_data('gep_no_checks', n_csp),
                get_data('pca_gep', n_csp),
                get_data('pca_gep_no_checks', n_csp)
            ]
            pvalue = friedmanchisquare(*np.array(d))
            print(n_csp)
            print(pvalue)

        # GAP
        for n_csp in results['aa'].n_csp_components:
            d = [
                get_data('gap_eig', n_csp),
                get_data('matlab_gap_no_check', n_csp),
                get_data('gap_dr', n_csp)
            ]
            pvalue = friedmanchisquare(*np.array(d))
            print(n_csp)
            print(pvalue)

        # GEP
        for n_csp in results['aa'].n_csp_components:
            d = [
                get_data('gep_no_checks', n_csp),
                get_data('matlab_gep_no_check', n_csp),
                get_data('pca_gep', n_csp),
                get_data('pca_gep_no_checks', n_csp)
            ]
            pvalue = friedmanchisquare(*np.array(d))
            print(n_csp)
            print(pvalue)

        # # https://github.com/maximtrp/scikit-posthocs/
        # r = sp.posthoc_nemenyi_friedman(np.array(d).T)
        #
        # heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
        #                 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        # fig = sp.sign_plot(r, **heatmap_args)
        # plt.show()


#%%
# test whether 2 methods are the same or not
method_a = 'gap_dr'
method_b = 'pca_gep'
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
# inspect max difference between protected Python eig and eigh
max_difference = []
for patient_name in PATIENTS:
    for artifact_removal_name in removal_methods:
        results = Result(patient_name, artifact_removal_name)
        for split_idx in range(10):
            for n_csp in results.n_csp_components:
                a = np.real(results.results['pca_gep_no_checks'][split_idx][n_csp]['eigenvalues'])
                b = np.real(results.results['pca_gep'][split_idx][n_csp]['eigenvalues'])
                max_difference.append(np.abs(a - b).max())

# np.array(max_difference).max()
