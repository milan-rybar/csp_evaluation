import os
from glob import glob

import numpy as np
from joblib import load
from scipy.stats import wilcoxon

from config import RESULTS_DIR
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
