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
from implementations.csp_matlab import matlab_wrapper
from implementations.csp_python import (csp_wrapper, csp_gep_no_checks, csp_geometric_approach,
                                        csp_geometric_approach_no_checks)
from utils import make_dirs


def grid_evaluation(dataset, artifact_removal, csp_methods, n_csp_components_list, classifiers):
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

        # run CSP for each number of CSP components separately
        # to have implementation agnostic results
        for csp_method_name, csp_method in csp_methods.items():
            split_results = {}
            for n_csp_components in n_csp_components_list:
                r = _classification(X_train, X_test, y_train, y_test, csp_method, n_csp_components, classifiers, dataset)
                split_results[n_csp_components] = r

            results[csp_method_name].append(split_results)

    # results as [csp method][cross-validation split][#csp components][classifier]
    return results


def _classification(X_train, X_test, y_train, y_test, csp_method, n_csp_components, classifiers, dataset):
    # compute CSP to get spatial filters
    unmixing_matrix, eigenvalues = csp_method(
        X=X_train, y=y_train, n_csp_components=n_csp_components, dataset=dataset)

    result = {
        'W_T': unmixing_matrix,
        'eigenvalues': eigenvalues,
        'classifier': {}
    }

    # project to component space and transform to log-variance
    X_train_transformed = transform_csp_components_for_classification(np.matmul(unmixing_matrix, X_train))
    X_test_transformed = transform_csp_components_for_classification(np.matmul(unmixing_matrix, X_test))

    # train different classifiers
    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train_transformed, y_train)

        score_train = classifier.score(X_train_transformed, y_train)
        score_test = classifier.score(X_test_transformed, y_test)

        result['classifier'][classifier_name] = {
            'score_train': score_train,
            'score_test': score_test
        }

    return result


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
            # 'gep_no_checks': partial(csp_wrapper, csp_method=csp_gep_no_checks),
            # geometric approach
            # 'gap': partial(csp_wrapper, csp_method=partial(csp_geometric_approach)),
            # # complex solution for first eigendecomposition (np.linalg.eig and scipy.linalg.eig behave the same)
            # 'rgap_eig': partial(csp_wrapper, csp_method=partial(csp_geometric_approach_no_checks, eig_method=np.linalg.eig, dim_reduction=True)),
            # 'gap_eig': partial(csp_wrapper, csp_method=partial(csp_geometric_approach_no_checks, eig_method=np.linalg.eig, dim_reduction=False)),
            'fieldtrip': partial(matlab_wrapper, csp_method='use_fieldtrip'),
        },
        n_csp_components_list=[2, 4, 6, 8, 10],
        classifiers={
            'lda': LinearDiscriminantAnalysis(),
            'svm': SVC(kernel='rbf', gamma='auto', C=1.0)  # as default values
        }
    )

    output_path = os.path.join(RESULTS_DIR, 'evaluation')
    make_dirs(output_path)
    dump(results, os.path.join(output_path, '{}.joblib'.format(patient_name)))

    break
