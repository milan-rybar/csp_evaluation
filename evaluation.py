import os
from functools import partial

import numpy as np
from joblib import dump
from jug import TaskGenerator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from artifacts_removal.ica_removal import (remove_artifacts_manual, remove_artifacts, ic_artifacts_by_kurtosis,
                                           ic_artifacts_by_peak_values)
from config import (ANALYSIS_FREQUENCY_START, ANALYSIS_FREQUENCY_END,
                    ANALYSIS_TIME_START, ANALYSIS_TIME_END, RESULTS_DIR)
from dataset import load_dataset, PATIENTS
from implementations.csp_matlab import matlab_package_wrapper, matlab_wrapper
from implementations.csp_python import (csp_wrapper, csp_gep_no_checks, csp_geometric_approach,
                                        csp_geometric_approach_no_checks, csp_gep)
from implementations.pca_dim_reduction import dim_reduction_pca
from utils import make_dirs


def grid_evaluation(dataset, artifact_removal, csp_method, n_csp_components_list, classifiers, pca_reduction=False):
    """
    Evaluation for all options.
    """
    # remove artifacts
    data, ic_artifacts = artifact_removal(dataset)

    # filter data
    data = dataset.filter_data(data, fmin=ANALYSIS_FREQUENCY_START, fmax=ANALYSIS_FREQUENCY_END)

    # extract all trials (trials, channels, time)
    labels_idx = dataset.competition_training_idx + dataset.competition_test_idx
    trials = dataset.get_trials(data=data, labels_idx=labels_idx, tmin=ANALYSIS_TIME_START, tmax=ANALYSIS_TIME_END)

    if pca_reduction:
        # reduce dimensions by PCA
        trials = dim_reduction_pca(trials)

    results = []

    # cross-validation (use always the same splits)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(trials, dataset.labels):
        X_train, X_test = trials[train_index], trials[test_index]
        y_train, y_test = dataset.labels[train_index], dataset.labels[test_index]

        split_results = {
            n_csp_components: _classification(X_train, X_test, y_train, y_test,
                                              csp_method, n_csp_components, classifiers, dataset)
            for n_csp_components in n_csp_components_list
        }
        results.append(split_results)

    return {
        'ic_artifacts': ic_artifacts,
        # results as [cross-validation split][#csp components][classifier]
        'results': results
    }


def _classification(X_train, X_test, y_train, y_test, csp_method, n_csp_components, classifiers, dataset):
    # compute CSP to get spatial filters
    unmixing_matrix, eigenvalues = csp_method(
        X=X_train, y=y_train, n_csp_components=n_csp_components, dataset=dataset)
    assert unmixing_matrix.shape == (n_csp_components, X_train.shape[1])

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

        # correct prediction on the test fold
        correct_test = classifier.predict(X_test_transformed) == y_test

        result['classifier'][classifier_name] = {
            'score_train': score_train,
            'score_test': score_test,
            'correct_test': correct_test
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


artifact_removal_methods = {
    'manual':  remove_artifacts_manual,
    'peak': partial(remove_artifacts, method=ic_artifacts_by_peak_values),
    'kurtosis': partial(remove_artifacts, method=ic_artifacts_by_kurtosis)
}

n_csp_components_list = list(range(2, 22, 2))

classifiers = {
    'lda': LinearDiscriminantAnalysis(),
    'svm': SVC(kernel='rbf', gamma='auto', C=1.0)  # as default values
}

csp_methods = {
    # generalized eigenvalue problem approach for full rank matrices
    'pca_gep': partial(csp_wrapper, csp_method=csp_gep),
    # generalized eigenvalue problem approach without any checks
    # that may have complex solution
    'gep_no_checks': partial(csp_wrapper, csp_method=csp_gep_no_checks),

    # geometric approach (with dimensionality reduction during whitening step)
    'gap_dr': partial(csp_wrapper, csp_method=partial(csp_geometric_approach)),

    # geometric approach that may have complex solution for first eigendecomposition
    # (note that `np.linalg.eig` and `scipy.linalg.eig` have the same behaviour)
    # i) with dimensionality reduction during whitening step
    'gap_eig_dr': partial(csp_wrapper, csp_method=partial(csp_geometric_approach_no_checks, eig_method=np.linalg.eig, dim_reduction=True)),
    # ii) without dimensionality reduction during whitening step
    'gap_eig': partial(csp_wrapper, csp_method=partial(csp_geometric_approach_no_checks, eig_method=np.linalg.eig, dim_reduction=False)),

    # Packages
    'fieldtrip': partial(matlab_package_wrapper, csp_method='use_fieldtrip'),
    'bbci': partial(matlab_package_wrapper, csp_method='use_bbci'),

    # our Matlab implementations
    'matlab_gep_no_check': partial(matlab_wrapper, csp_method='csp_gep_no_checks'),
    'matlab_gap_no_check': partial(matlab_wrapper, csp_method='csp_geometric_approach_no_checks')
}


@TaskGenerator
def run_task(patient_name, artifact_removal_name, csp_method_name, pca_reduction, prefix):
    output_path = os.path.join(RESULTS_DIR, 'evaluation', patient_name)
    make_dirs(output_path)

    try:
        result = grid_evaluation(
            dataset=load_dataset(patient_name),
            artifact_removal=artifact_removal_methods[artifact_removal_name],
            csp_method=csp_methods[csp_method_name],
            n_csp_components_list=n_csp_components_list,
            classifiers=classifiers,
            pca_reduction=pca_reduction
        )

        dump(result, os.path.join(output_path, '{}{}_{}.joblib'.format(prefix, artifact_removal_name, csp_method_name)))
    except Exception as e:
        # do not save any results on bad CSP implementation
        dump(e, os.path.join(output_path, '{}{}_{}.exception'.format(prefix, artifact_removal_name, csp_method_name)))
        print(e)


for patient_name in PATIENTS:
    for artifact_removal_name in artifact_removal_methods.keys():
        for csp_method_name in csp_methods.keys():
            run_task(patient_name, artifact_removal_name, csp_method_name, pca_reduction=False, prefix='')
            run_task(patient_name, artifact_removal_name, csp_method_name, pca_reduction=True, prefix='pca_')
