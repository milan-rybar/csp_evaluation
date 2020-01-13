"""
Remove artifacts from pre-computed ICA, see the script `compute_ica`.
"""

import logging
import os

import mne
import numpy as np
from scipy.stats import kurtosis

from config import RESULTS_DIR, ANALYSIS_TIME_END

ICA_ARTIFACTS = {
    'aa': [0, 15, 34, 48, 61, 75, 80, 100, 101, 103],
    'al': [5, 12, 36, 107],
    'av': [8, 9, 11, 16, 18, 19, 21, 22, 23, 26, 30, 31, 32, 43, 49, 51, 56, 65, 66, 69, 72],
    'aw': [1, 29, 32, 35, 43, 45, 52, 58, 77],
    'ay': [0, 15, 49, 67, 77, 85]
}


def remove_artifacts_manual(dataset):
    filtered_data = dataset.filter_data(dataset.data, fmin=1.0, fmax=40.0)

    # remove ICs artifacts
    raw = dataset.get_mne_raw(filtered_data)
    ica_path = os.path.join(RESULTS_DIR, 'ica', '{}_ica.fif'.format(dataset.patient_name))
    ica = mne.preprocessing.read_ica(ica_path)

    ica.exclude = ICA_ARTIFACTS[dataset.patient_name]
    ica.apply(raw)

    return raw.get_data()


def remove_artifacts(dataset, method):
    filtered_data = dataset.filter_data(dataset.data, fmin=1.0, fmax=40.0)

    # get trials (same as for ICA computation)
    labels_idx = dataset.competition_training_idx + dataset.competition_test_idx
    trials_epochs = dataset.get_mne_epochs(
        raw=dataset.get_mne_raw(filtered_data), labels_idx=labels_idx,
        tmin=0.0, tmax=ANALYSIS_TIME_END)

    # remove ICs artifacts
    raw = dataset.get_mne_raw(filtered_data)
    ica_path = os.path.join(RESULTS_DIR, 'ica', '{}_ica.fif'.format(dataset.patient_name))
    ica = mne.preprocessing.read_ica(ica_path)

    ica.exclude = method(ica, trials_epochs)
    logging.info('Removing: {}'.format(ica.exclude))
    ica.apply(raw)

    return raw.get_data()


def ic_artifacts_by_peak_values(ica, trials_epochs):
    n_channels = trials_epochs.info['nchan']

    ic_artifacts = []
    for channel_idx in range(n_channels):
        # project one IC back to channels (NOTE: `trials_epochs` is modified inplace => copy)
        scalp_ic = ica.apply(trials_epochs.copy(), include=[channel_idx])

        # (trials, channels, time)
        data = scalp_ic.get_data()

        # threshold +-100uV
        is_artifact = np.abs([data.min(), data.max()]).max() > 100.0

        # peak-to-peak difference threshold 60uV per each trial/channel
        is_artifact |= ((data.max(axis=-1) - data.min(axis=-1)) > 60.0).any()

        if is_artifact:
            ic_artifacts.append(channel_idx)

    return ic_artifacts


def ic_artifacts_by_kurtosis(ica, trials_epochs):
    n_trials, n_channels, _ = trials_epochs.get_data().shape

    scalp_kurtosis = []
    for channel_idx in range(n_channels):
        # project one IC back to channels (NOTE: `trials_epochs` is modified inplace => copy)
        scalp_ic = ica.apply(trials_epochs.copy(), include=[channel_idx])

        # kurtosis of the scalp projection of each trial
        k = kurtosis(scalp_ic.get_data().reshape(n_trials, -1), axis=-1)
        assert k.shape == (n_trials, )
        scalp_kurtosis.append(k)
    scalp_kurtosis = np.stack(scalp_kurtosis)
    assert scalp_kurtosis.shape == (n_channels, n_trials)

    # k > (μ(k) + (0.5 × σ(k))) for each trial
    # TODO: consider 2.0 instead of 0.5
    is_artifacts = (scalp_kurtosis > (scalp_kurtosis.mean(axis=0) + (0.5 * scalp_kurtosis.std(axis=0)))).any(axis=1)
    assert is_artifacts.shape == (n_channels, )
    ic_artifacts = is_artifacts.nonzero()[0]

    return list(ic_artifacts)
