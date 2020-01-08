import logging
import os
import subprocess
import tempfile

import numpy as np
from mne.externals.pymatreader import read_mat
from scipy.io import savemat


def matlab_wrapper(X, y, csp_method, n_csp_components, dataset):
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
        result = read_mat(output_path)

        unmixing_matrix = np.array(result['unmixing_matrix'], dtype=np.double)
        logging.debug('CSP unmixing matrix %s', unmixing_matrix.shape)
        assert unmixing_matrix.shape[0] == n_csp_components, unmixing_matrix.shape

        eigenvalues = result['eigenvalues'] if 'eigenvalues' in result else None

        return unmixing_matrix, eigenvalues


def save_as_mat(file_path, trials, labels, dataset):
    data = {
        'trials': trials,
        'labels': labels,
        'channel_names': dataset.channel_names,
        'fs': dataset.fs
    }
    savemat(file_name=file_path, mdict=data, do_compression=True)
