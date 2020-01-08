"""
Compute ICA for each patient and save ICA component properties.
"""

import os

import mne
import numpy as np
from matplotlib import pyplot as plt

from config import ANALYSIS_TIME_END, RESULTS_DIR
from dataset import PATIENTS, load_dataset
from utils import make_dirs

for patient_name in PATIENTS:
    dataset = load_dataset(patient_name)

    filtered_data = dataset.filter_data(dataset.data, fmin=1.0, fmax=40)

    # all trials (trials, channels, time)
    labels_idx = dataset.competition_training_idx + dataset.competition_test_idx
    trials = dataset.get_trials(data=filtered_data, labels_idx=labels_idx, tmin=0.0, tmax=ANALYSIS_TIME_END)
    n_channels = trials.shape[1]

    # get the same using MNE structures
    trials_epochs = dataset.get_mne_epochs(raw=dataset.get_mne_raw(filtered_data), labels_idx=labels_idx,
                                           tmin=0.0, tmax=ANALYSIS_TIME_END)
    # check that we have the same data
    np.testing.assert_equal(trials_epochs.get_data(), trials)

    # train ICA on all trials
    ica = mne.preprocessing.ICA(n_components=n_channels, method='fastica', max_iter=1000)
    ica.fit(trials_epochs)

    # save ICA
    output_path = os.path.join(RESULTS_DIR, 'ica')
    make_dirs(output_path)
    ica.save(os.path.join(output_path, '{}_ica.fif'.format(patient_name)))

    # save ICA component properties
    output_path = os.path.join(RESULTS_DIR, 'ica_properties', patient_name)
    make_dirs(output_path)

    layout = dataset.get_mne_layout()
    for channel_idx in range(n_channels):
        fig = ica.plot_properties(trials_epochs, picks=channel_idx, psd_args=dict(fmax=45), show=False,
                                  topomap_args=dict(layout=layout))[0]
        plt.savefig(os.path.join(output_path, 'ica_{}_properties.png'.format(channel_idx)))
        plt.close(fig)
