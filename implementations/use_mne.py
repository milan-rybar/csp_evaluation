import logging

import mne


def use_mne(X, y, n_csp_components, dataset):
    _, n_channels, _ = X.shape
    logging.debug('CSP fit X %s y %s', X.shape, y.shape)

    csp = mne.decoding.CSP(n_components=n_csp_components, reg=None, norm_trace=True, cov_est='epoch')
    csp.fit(X, y)

    # according to inner implementation
    # NOTE: MNE uses different selection criterion:
    #   # sort eigenvectors
    #   ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
    W_T = csp.filters_[:csp.n_components]
    assert W_T.shape == (n_csp_components, n_channels)

    # eigenvalues are not stored
    eigenvalues = None

    return W_T, eigenvalues
