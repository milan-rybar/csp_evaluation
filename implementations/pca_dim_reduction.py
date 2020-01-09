import numpy as np
from sklearn.decomposition import PCA

from implementations.csp_python import compute_mean_normalized_spatial_covariance


def dim_reduction_pca(trials):
    rank = np.linalg.matrix_rank(trials[0])  # matrix rank of the first trial
    assert rank == np.linalg.matrix_rank(compute_mean_normalized_spatial_covariance(trials))

    # reduce dimensions by PCA
    pca = PCA(n_components=rank, whiten=False)
    pca.fit(transform_for_scikit(trials))

    trials_subspace = pca_transform(pca, trials)
    assert trials_subspace.shape == (trials.shape[0], rank, trials.shape[2])

    return trials_subspace


def transform_for_scikit(data):
    assert len(data.shape) == 3  # (trials, channels, time)
    n_channels = data.shape[1]
    # concatenate trials and time dimension
    data = data.swapaxes(0, 1)  # (channels, trials, time)
    data = data.reshape((n_channels, -1))  # (channels, trials x time)
    assert len(data.shape) == 2 and data.shape[0] == n_channels
    data = data.swapaxes(0, 1)  # (trials x time, channels) as (n_samples, n_features) for scikit
    return data


def transform_from_scikit(data, n_trials, n_components):
    assert len(data.shape) == 2 and data.shape[1] == n_components  # (trials x time, components)
    data = data.swapaxes(0, 1)  # (components, trials x time)
    data = data.reshape((n_components, n_trials, -1))  # (components, trials, time)
    data = data.swapaxes(0, 1)  # (trials, channels, time)
    return data


def pca_transform(pca, data):
    pca_data = pca.transform(transform_for_scikit(data))
    pca_data = transform_from_scikit(pca_data, n_trials=data.shape[0], n_components=pca.n_components)
    assert pca_data.shape == (data.shape[0], pca.n_components, data.shape[2])
    return pca_data
