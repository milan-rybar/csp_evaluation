"""
Remove artifacts based on our manual artifact components selection
(from pre-computed ICA, see the script `compute_ica`).
"""

import os

import mne

from config import RESULTS_DIR

ICA_ARTIFACTS = {
    'aa': [0, 15, 34, 48, 61, 75, 80, 100, 101, 103],
    'al': [5, 12, 36, 107],
    'av': [8, 9, 11, 16, 18, 19, 21, 22, 23, 26, 30, 31, 32, 43, 49, 51, 56, 65, 66, 69, 72],
    'aw': [1, 29, 32, 35, 43, 45, 52, 58, 77],
    'ay': [0, 15, 49, 67, 77, 85]
}


def remove_artifacts(dataset):
    filtered_data = dataset.filter_data(dataset.data, fmin=1.0, fmax=40.0)

    # remove ICs artifacts
    raw = dataset.get_mne_raw(filtered_data)
    ica_path = os.path.join(RESULTS_DIR, 'ica', '{}_ica.fif'.format(dataset.patient_name))
    ica = mne.preprocessing.read_ica(ica_path)

    ica.exclude = ICA_ARTIFACTS[dataset.patient_name]
    ica.apply(raw)

    return raw.get_data()
