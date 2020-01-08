"""
Remove artifacts based on our manual artifact components selection
(from pre-computed ICA, see the script `compute_ica`).
"""

import os

import mne

from config import RESULTS_DIR

ICA_ARTIFACTS = {
    'aa': [0, 15, 34, 48, 61, 69, 75, 80, 100, 101, 103, 110]
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
