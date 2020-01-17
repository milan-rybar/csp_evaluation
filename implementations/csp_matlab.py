import logging
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import savemat, loadmat

from implementations.csp_python import compute_mean_normalized_spatial_covariance, get_n_csp_components


def matlab_package_wrapper(X, y, csp_method, n_csp_components, dataset):
    """
    Wrapper for different Matlab CSP implementations.
    """
    logging.debug('CSP fit X %s y %s', X.shape, y.shape)

    # directory with .m files
    source_path = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # save data to be used by Matlab CSP implementation
        data_path = os.path.join(tmp_dir_path, 'data.mat')
        save_as_mat(file_path=data_path, trials=X, labels=y, dataset=dataset)

        # file to store the result
        output_path = os.path.join(tmp_dir_path, 'result.mat')

        # run Matlab CSP implementation
        command = [
            '/usr/local/MATLAB/R2018b/bin/matlab',
            '-nodisplay', '-nosplash', '-nodesktop',
            '-r',
            'cd("{}"); {}("{}", {}, "{}"); exit;'.format(
                source_path, csp_method, data_path, n_csp_components, output_path)
        ]
        logging.info('Command: {}'.format(' '.join(command)))
        subprocess.run(command)

        # get the result
        result = loadmat(output_path)

        unmixing_matrix = result['unmixing_matrix']
        assert unmixing_matrix.dtype in [np.double, np.complex], unmixing_matrix.dtype
        logging.debug('CSP unmixing matrix %s', unmixing_matrix.shape)
        assert unmixing_matrix.shape[0] == n_csp_components, unmixing_matrix.shape

        eigenvalues = result['eigenvalues'] if 'eigenvalues' in result else None

        return unmixing_matrix, eigenvalues, None


def save_as_mat(file_path, trials, labels, dataset):
    data = {
        'trials': trials,
        'labels': labels,
        'channel_names': dataset.channel_names,
        'fs': dataset.fs
    }
    savemat(file_name=file_path, mdict=data, do_compression=True)


def matlab_wrapper(X, y, csp_method, n_csp_components, dataset):
    """
    Wrapper for our different Matlab CSP implementations.
    """
    logging.debug('CSP fit X %s y %s', X.shape, y.shape)

    # split data to 2 classes
    data_1 = X[y == 0]
    data_2 = X[y == 1]
    logging.debug('CSP class data %s and %s', data_1.shape, data_2.shape)

    # compute CSP to get spatial filters
    R_1 = compute_mean_normalized_spatial_covariance(data_1)
    R_2 = compute_mean_normalized_spatial_covariance(data_2)

    # directory with .m files
    source_path = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        data = {
            'R_1': R_1,
            'R_2': R_2
        }
        data_path = os.path.join(tmp_dir_path, 'data.mat')
        savemat(file_name=data_path, mdict=data, do_compression=True)

        # file to store the result
        output_path = os.path.join(tmp_dir_path, 'result.mat')

        # run Matlab CSP implementation
        command = [
            '/usr/local/MATLAB/R2018b/bin/matlab',
            '-nodisplay', '-nosplash', '-nodesktop',
            '-r',
            'cd("{}"); {}("{}", "{}"); exit;'.format(
                source_path, csp_method, data_path, output_path)
        ]
        logging.info('Command: {}'.format(' '.join(command)))
        subprocess.run(command)

        # get the result
        result = loadmat(output_path)

        W_T = result['unmixing_matrix']
        assert W_T.dtype in [np.double, np.complex], W_T.dtype
        logging.debug('CSP unmixing matrix %s', W_T.shape)
        assert W_T.shape == (X.shape[1], X.shape[1])

        eigenvalues = result['eigenvalues']

    logging.debug('CSP unmixing matrix %s', W_T.shape)

    # select N CSP components
    selected_W_T = get_n_csp_components(W_T, n_csp_components // 2)

    return selected_W_T, eigenvalues, W_T
