import logging
import os
from collections import defaultdict
from functools import partial

import numpy as np
from joblib import dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from artifacts_removal.ica_removal import remove_artifacts
from config import (ANALYSIS_FREQUENCY_START, ANALYSIS_FREQUENCY_END,
                    ANALYSIS_TIME_START, ANALYSIS_TIME_END, RESULTS_DIR)
from dataset import load_dataset, PATIENTS
from implementations.csp_python import (csp_wrapper, csp_gep_no_checks, csp_geometric_approach,
                                        csp_geometric_approach_no_checks)
from utils import make_dirs


def grid_evaluation(dataset, artifact_removal, csp_methods, n_csp_components, classifiers):
    """
    Evaluation for all options.
    """
    # remove artifacts
    data = artifact_removal(dataset)

    # filter data
    data = dataset.filter_data(data, fmin=ANALYSIS_FREQUENCY_START, fmax=ANALYSIS_FREQUENCY_END)

    # extract all trials (trials, channels, time)
    labels_idx = dataset.competition_training_idx + dataset.competition_test_idx
    trials = dataset.get_trials(data=data, labels_idx=labels_idx, tmin=ANALYSIS_TIME_START, tmax=ANALYSIS_TIME_END)

    results = defaultdict(list)

    # cross-validation (use always the same splits)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(trials, dataset.labels):
        X_train, X_test = trials[train_index], trials[test_index]
        y_train, y_test = dataset.labels[train_index], dataset.labels[test_index]

        # train different CSP implementations
        for csp_method_name, csp_method in csp_methods.items():
            result = _classification(X_train, X_test, y_train, y_test, csp_method, n_csp_components, classifiers)
            results[csp_method_name].append(result)

    return results


def _classification(X_train, X_test, y_train, y_test, csp_method, n_csp_components, classifiers):
    # compute CSP to get spatial filters
    W_T, eigenvalues = csp_method(X_train, y_train)

    result = {
        'W_T': W_T,
        'eigenvalues': eigenvalues,
        'n_csp': defaultdict(dict)
    }

    for n_csp in n_csp_components:
        unmixing_matrix = get_n_csp_components(W_T, n_csp)

        # project to component space and transform to log-variance
        X_train_transformed = transform_csp_components_for_classification(np.matmul(unmixing_matrix, X_train))
        X_test_transformed = transform_csp_components_for_classification(np.matmul(unmixing_matrix, X_test))

        # train different classifiers
        for classifier_name, classifier in classifiers.items():
            classifier.fit(X_train_transformed, y_train)

            score_train = classifier.score(X_train_transformed, y_train)
            score_test = classifier.score(X_test_transformed, y_test)

            result['n_csp'][n_csp][classifier_name] = {
                'score_train': score_train,
                'score_test': score_test
            }

    return result


def get_n_csp_components(W_T, n_select):
    assert len(W_T.shape) == 2  # (components, channels)

    n_sel_sources = 2 * n_select
    # select 2 * n components (n first and n last)
    selection = tuple(list(range(0, n_select)) + list(np.array(range(1, n_select + 1)) * -1))
    assert len(selection) == n_sel_sources
    logging.debug('Select subset: %s', selection)

    W_T_selected = W_T[selection, :]
    assert W_T_selected.shape == (n_sel_sources, W_T_selected.shape[1])
    return W_T_selected


def transform_csp_components_for_classification(data):
    n_trials, n_components, n_time = data.shape  # data as (trials, CSP components, time)

    # variance of components over time
    transformed_data = data.var(axis=2)
    assert transformed_data.shape == (n_trials, n_components)

    # normalize
    sum_variance = transformed_data.sum(axis=1, keepdims=True)
    assert sum_variance.shape == (n_trials, 1)
    transformed_data /= sum_variance
    assert transformed_data.shape == (n_trials, n_components)

    # logarithm (serve to approximate normal distribution)
    transformed_data = np.log(transformed_data)
    assert transformed_data.shape == (n_trials, n_components)

    return transformed_data


for patient_name in PATIENTS:
    results = grid_evaluation(
        dataset=load_dataset(patient_name),
        artifact_removal=remove_artifacts,
        csp_methods={
            # generalized eigenvalue problem approach without any checks
            'gep_no_checks': partial(csp_wrapper, csp_method=csp_gep_no_checks),
            # geometric approach
            'gap': partial(csp_wrapper, csp_method=partial(csp_geometric_approach)),
            # complex solution for first eigendecomposition (np.linalg.eig and scipy.linalg.eig behave the same)
            'rgap_eig': partial(csp_wrapper, csp_method=partial(csp_geometric_approach_no_checks, eig_method=np.linalg.eig, dim_reduction=True)),
            'gap_eig': partial(csp_wrapper, csp_method=partial(csp_geometric_approach_no_checks, eig_method=np.linalg.eig, dim_reduction=False)),
        },
        n_csp_components=[1, 2, 3, 4, 5],
        classifiers={
            'lda': LinearDiscriminantAnalysis(),
            'svm': SVC(kernel='rbf', gamma='auto', C=1.0)  # as default values
        }
    )

    output_path = os.path.join(RESULTS_DIR, 'evaluation')
    make_dirs(output_path)
    dump(results, os.path.join(output_path, '{}.joblib'.format(patient_name)))

    break
