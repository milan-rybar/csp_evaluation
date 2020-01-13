import os

import mne
import numpy as np
from mne.externals.pymatreader import read_mat

from config import DATA_DIR
from utils import plot_scalpmaps_of_matrix_columns

# name of patients
PATIENTS = ['aa', 'al', 'av', 'aw', 'ay']


class Dataset(object):
    """
    Represents data from single participant from BCI Competition III IVa.
    """

    def __init__(self, data_set_path, true_labels_path, patient_name):
        self.data_set = read_mat(data_set_path)
        self.true_labels = read_mat(true_labels_path)
        self.patient_name = patient_name

        # data as (channels, time)
        self.data = 0.1 * np.array(self.data_set['cnt'], dtype=np.double)  # convert to uV values
        self.data = self.data.swapaxes(0, 1)  # (time, channels) to (channels, time)

        labels = self.data_set['mrk']['y']
        # training set used in the competition (exclude labels with NaN values)
        self.competition_training_idx = np.isfinite(labels)
        # test set used in the competition (labels with NaN values)
        self.competition_test_idx = np.isnan(labels)

        # number of channels
        self.n_channels, _ = self.data.shape
        # sampling frequency
        self.fs = self.data_set['nfo']['fs']
        # channel names
        self.channel_names = self.data_set['nfo']['clab']

        # use 0, 1 as labels (instead of 1, 2)
        self.labels = self.true_labels['true_y'] - 1
        # frame indexes of corresponding labels
        self.labels_frame = self.data_set['mrk']['pos']
        assert len(self.labels) == len(self.labels_frame)

    def filter_data(self, data, fmin, fmax):
        return mne.filter.filter_data(
            data=data, sfreq=self.fs, l_freq=fmin, h_freq=fmax,
            fir_design='firwin')

    def get_trials(self, data, tmin, tmax, labels_idx=None):
        if labels_idx is None:
            labels_idx = np.ones(len(self.labels_frame), dtype=np.bool)

        # get trials (trials, channels, time)
        trials_data = np.stack([
            data[:, int(label_time + tmin * self.fs):int(label_time + tmax * self.fs)]
            for label_time in self.labels_frame[labels_idx]
        ])
        assert trials_data.shape == (np.sum(labels_idx), self.n_channels, int((tmax - tmin) * self.fs))
        return trials_data

    def plot_scalpmaps_of_matrix_columns(self, matrix, title=None, same_scale=True):
        return plot_scalpmaps_of_matrix_columns(
            matrix=matrix,
            ch_names=self.channel_names,
            pos=np.array(list(zip(self.data_set['nfo']['xpos'], self.data_set['nfo']['ypos']))),
            title=title,
            same_scale=same_scale
        )

    def get_mne_raw(self, data):
        info = mne.create_info(ch_names=self.channel_names, sfreq=self.fs, ch_types='eeg', montage=None)
        return mne.io.RawArray(data=data, info=info)

    def get_mne_epochs(self, raw, tmin, tmax, labels_idx):
        n_events = len(labels_idx)
        events = np.zeros((n_events, 3), dtype=np.int)
        for trial_index, label_time in enumerate(self.labels_frame[labels_idx]):
            events[trial_index, 0] = label_time  # frame
            events[trial_index, 2] = 1  # event code

        # NOTE: MNE Epochs has `tmax` inclusive => change to exclusive
        return mne.Epochs(raw=raw, events=events, tmin=tmin, tmax=tmax - 1.0 / self.fs,
                          baseline=None, detrend=None, proj=False, preload=True)

    def get_mne_layout(self):
        pos = np.array(list(zip(self.data_set['nfo']['xpos'], self.data_set['nfo']['ypos'])))
        return mne.channels.generate_2d_layout(pos, ch_names=self.channel_names)


def load_dataset(patient_name, dir_path=DATA_DIR):
    data_set_path = os.path.join(dir_path, 'data_set_IVa_{}.mat'.format(patient_name))
    true_labels_path = os.path.join(dir_path, 'true_labels_{}.mat'.format(patient_name))

    return Dataset(data_set_path, true_labels_path, patient_name)
