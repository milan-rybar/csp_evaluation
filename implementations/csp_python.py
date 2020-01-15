import logging

import numpy as np
import scipy.linalg


def compute_mean_normalized_spatial_covariance(data):
    """
    Compute mean of normalized spatial covariance.

    :param data: data of shape (trials, channels, times)
    :type data: `numpy.ndarray`
    :return: mean of normalized covariances of shape (channels, channels)
    :rtype: `numpy.ndarray`
    """
    n_trials, n_channels, _ = data.shape

    # normalized spatial covariances
    R = np.zeros((n_channels, n_channels), dtype=data.dtype)
    for trial_idx in range(n_trials):
        # covariance of the trial
        cov = np.matmul(data[trial_idx, :, :], data[trial_idx, :, :].T)
        # normalize covariance by its trace
        cov /= np.trace(cov)
        R += cov
    # mean normalized spatial covariance
    R_mean = R / n_trials

    assert R_mean.shape == (n_channels, n_channels)  # shape (channels, channels)
    np.testing.assert_equal(R_mean, R_mean.T)  # symmetric real matrix

    return R_mean


def csp_gep(R_1, R_2):
    """
    Compute CSP (common spatial patterns) as a generalized eigenvalue problem.

    This method works only for covariance matrices with full rank!

    :param R_1: covariance matrix for the first class as (channels, channels)
    :type R_1: `numpy.ndarray`
    :param R_2: covariance matrix for the second class as (channels, channels)
    :type R_2: `numpy.ndarray`
    :return: decomposition matrix W^T of shape (channels, channels)
        where spatial filters are in rows
        and their corresponding eigenvalues D_1
    :rtype: (`numpy.ndarray`, `numpy.ndarray`)
    """
    # covariances are real symmetric matrices with full rank
    np.testing.assert_equal(R_1, R_1.T)  # real symmetric matrix
    np.testing.assert_equal(R_2, R_2.T)  # real symmetric matrix
    assert R_1.shape == R_2.shape
    assert np.linalg.matrix_rank(R_1) == R_1.shape[0]  # full rank
    assert np.linalg.matrix_rank(R_2) == R_2.shape[0]  # full rank
    n_channels = R_1.shape[0]

    # generalized eigenvalue problem for real symmetric matrices
    D_1, W = scipy.linalg.eigh(R_1, R_1 + R_2, type=1)  # for real symmetric matrices
    assert not np.any(np.iscomplex(D_1)) and not np.any(np.iscomplex(W))
    np.testing.assert_equal(D_1, sorted(D_1))  # ascending order

    # transpose as eigenvectors (spatial filters) are in columns of W
    W_T = W.T
    assert W_T.shape == (n_channels, n_channels)

    return W_T, D_1


def csp_gep_no_checks(R_1, R_2):
    """
    CSP as generalized eigenvalue problem approach without any checks.

    :param R_1: covariance matrix for the first class as (channels, channels)
    :type R_1: `numpy.ndarray`
    :param R_2: covariance matrix for the second class as (channels, channels)
    :type R_2: `numpy.ndarray`
    :return: decomposition matrix W^T of shape (channels, channels)
        where spatial filters are in rows
        and their corresponding eigenvalues D_1
    :rtype: (`numpy.ndarray`, `numpy.ndarray`)
    """
    # generalized eigenvalue problem
    D_1, W = scipy.linalg.eig(R_1, R_1 + R_2)

    # eigenvalues are not ordered
    sort_idx = np.argsort(D_1)  # ascending order
    D_1_sorted = D_1[sort_idx]
    np.testing.assert_equal(D_1_sorted, sorted(D_1))
    # eigenvectors sorted in ascending order by corresponding eigenvalues
    W_sorted = W[:, sort_idx]

    # transpose as eigenvectors (spatial filters) are in columns of W
    return W_sorted.T, D_1_sorted


def csp_geometric_approach(R_1, R_2):
    """
    Compute CSP (common spatial patterns) in geometric approach.

    This method also works for singular covariance matrices (e.g., due to ICA-based artifacts removal).
    The spatial projection will be to a space with a (possible lower) dimension,
    which is equal to the rank of the composite covariance matrix (R_1 + R_2).

    :param R_1: covariance matrix for the first class as (channels, channels)
    :type R_1: `numpy.ndarray`
    :param R_2: covariance matrix for the second class as (channels, channels)
    :type R_2: `numpy.ndarray`
    :return: decomposition matrix W^T of shape (n_sources, channels)
        where spatial filters are in rows and
        n_sources is the rank of the composite covariance matrix (R_1 + R_2)
        and their corresponding eigenvalues D_1
    :rtype: (`numpy.ndarray`, `numpy.ndarray`)
    """
    # covariances are real symmetric matrices
    np.testing.assert_equal(R_1, R_1.T)  # real symmetric matrix
    np.testing.assert_equal(R_2, R_2.T)  # real symmetric matrix
    assert R_1.shape == R_2.shape
    n_channels = R_1.shape[0]

    # composite spatial covariance matrix
    R_c = R_1 + R_2

    # factorize R_c into eigenvalues and eigenvectors
    F, E = np.linalg.eigh(R_c)  # for real symmetric matrix
    np.testing.assert_equal(F, sorted(F))  # ascending order

    # covariance matrix can be singular (e.g., due to ICA-based artifacts removal) =>
    # exclude axis (eigenvectors) with zero-eigenvalues => dimensionality reduction at this point
    valid_axis = np.nonzero(F >= 1e-14)[0]
    assert len(valid_axis) == np.linalg.matrix_rank(R_c), (len(valid_axis), np.linalg.matrix_rank(R_c))
    n_sources = len(valid_axis)
    logging.info('Composite covariance matrix has %d zero-eigenvalues', n_channels - n_sources)

    F_sub = F[valid_axis]
    E_sub = E[:, valid_axis]  # eigenvectors are in columns

    # whitening transformation matrix
    U = np.matmul(np.diag(1.0 / np.sqrt(F_sub)), E_sub.T)  # shape (n_sources, n_channels)
    assert not np.any(np.isnan(U))  # check for numerical problems
    assert U.shape == (n_sources, n_channels)

    # whiten spatial covariance matrix R_1
    S_1 = np.matmul(np.matmul(U, R_1), U.T)  # shape (n_sources, n_sources)

    # S_1 is real symmetric matrix with full rank of shape (n_sources, n_sources)
    np.testing.assert_almost_equal(S_1, S_1.T, decimal=12)
    assert S_1.shape == (n_sources, n_sources)
    assert np.linalg.matrix_rank(S_1) == n_sources, (np.linalg.matrix_rank(S_1), n_sources)

    # factorize S_1 into eigenvalues and eigenvectors
    D_1, P = np.linalg.eigh(S_1)  # for real symmetric matrix
    np.testing.assert_equal(D_1, sorted(D_1))  # ascending order

    # spatial filters are in rows, shape (n_sources, n_channels)
    W_T = np.matmul(P.T, U)
    assert W_T.shape == (n_sources, n_channels)

    return W_T, D_1


def csp_geometric_approach_no_checks(R_1, R_2, eig_method, dim_reduction=True):
    """
    Compute CSP (common spatial patterns) in geometric approach without any checks.

    See method `csp_geometric_approach` for more information.

    :param R_1: covariance matrix for the first class as (channels, channels)
    :type R_1: `numpy.ndarray`
    :param R_2: covariance matrix for the second class as (channels, channels)
    :type R_2: `numpy.ndarray`
    :param eig_method: eigendecomposition method to use
    :param dim_reduction: whether to remove dimensions with zero eingevalues or not
    :return: decomposition matrix W^T of shape (n_sources, channels)
        where spatial filters are in rows
        and their corresponding eigenvalues D_1
    :rtype: (`numpy.ndarray`, `numpy.ndarray`)
    """
    # composite spatial covariance matrix
    R_c = R_1 + R_2

    # factorize R_c into eigenvalues and eigenvectors
    F, E = eig_method(R_c)

    # eigenvalues may not be ordered
    sort_idx = np.argsort(F)  # ascending order
    np.testing.assert_equal(F[sort_idx], sorted(F))  # ascending order
    F = F[sort_idx]
    # eigenvectors sorted in ascending order by corresponding eigenvalues
    E = E[:, sort_idx]

    if dim_reduction:
        # covariance matrix can be singular (e.g., due to ICA-based artifacts removal) =>
        # exclude axis (eigenvectors) with zero-eigenvalues => dimensionality reduction at this point
        valid_axis = np.nonzero(F >= 1e-14)[0]
        assert len(valid_axis) == np.linalg.matrix_rank(R_c), (len(valid_axis), np.linalg.matrix_rank(R_c))

        F = F[valid_axis]
        E = E[:, valid_axis]  # eigenvectors are in columns

    # whitening transformation matrix
    U = np.matmul(np.diag(1.0 / np.sqrt(F)), E.T)  # shape (n_sources, n_channels)
    assert not np.any(np.isnan(U))  # check for numerical problems

    # whiten spatial covariance matrix R_1
    S_1 = np.matmul(np.matmul(U, R_1), U.T)  # shape (n_sources, n_sources)

    # factorize S_1 into eigenvalues and eigenvectors
    D_1, P = eig_method(S_1)  # for real symmetric matrix

    # eigenvalues may not be ordered
    sort_idx = np.argsort(D_1)  # ascending order
    D_1 = D_1[sort_idx]
    # eigenvectors sorted in ascending order by corresponding eigenvalues
    P = P[:, sort_idx]

    # spatial filters are in rows, shape (n_sources, n_channels)
    W_T = np.matmul(P.T, U)

    return W_T, D_1


def csp_wrapper(X, y, csp_method, n_csp_components, dataset):
    """
    Wrapper for different CSP implementations.
    """
    logging.debug('CSP fit X %s y %s', X.shape, y.shape)

    # split data to 2 classes
    data_1 = X[y == 0]
    data_2 = X[y == 1]
    logging.debug('CSP class data %s and %s', data_1.shape, data_2.shape)

    # compute CSP to get spatial filters
    R_1 = compute_mean_normalized_spatial_covariance(data_1)
    R_2 = compute_mean_normalized_spatial_covariance(data_2)

    W_T, eigenvalues = csp_method(R_1, R_2)
    logging.debug('CSP unmixing matrix %s', W_T.shape)

    # select N CSP components
    unmixing_matrix = get_n_csp_components(W_T, n_csp_components // 2)

    return unmixing_matrix, eigenvalues


def get_n_csp_components(W_T, n_select):
    assert len(W_T.shape) == 2  # (components, channels)

    n_sel_sources = 2 * n_select
    # select 2 * n components (n first and n last)
    selection = tuple(list(range(0, n_select)) + list(np.array(range(1, n_select + 1)) * -1))
    assert len(selection) == n_sel_sources
    logging.debug('Select subset: %s', selection)

    W_T_selected = W_T[selection, :]
    assert W_T_selected.shape == (n_sel_sources, W_T_selected.shape[1])
    return W_T_selected
