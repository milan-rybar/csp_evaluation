import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare

from config import RESULTS_DIR
from dataset import PATIENTS
from utils import make_dirs
from visualize.common import Result
import scikit_posthocs as sp

removal_methods = ['peak']

# plot aggregate results
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

        # protected
        d = []
        n_csp = 8
        d.append(get_data('gep_no_checks', n_csp))
        d.append(get_data('pca_gep', n_csp))
        d.append(get_data('pca_gep_no_checks', n_csp))
        d.append(get_data('matlab_gep_no_check', n_csp))

        pvalue = friedmanchisquare(*np.array(d))
        print(pvalue)

        # https://github.com/maximtrp/scikit-posthocs/
        r = sp.posthoc_nemenyi_friedman(np.array(d).T)

        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                        'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        fig = sp.sign_plot(r, **heatmap_args)
        plt.show()

