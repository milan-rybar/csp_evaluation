import itertools
import logging
import os

import mne
import numpy as np
from matplotlib import pyplot as plt


def plot_scalpmaps_of_matrix_columns(matrix, ch_names, pos, title=None, same_scale=False):
    """
    Plot scalpmaps for columns of the provided (transformation) matrix.

    The matrix should be an inverse matrix of the linear transformation,
    if one is interested in most important channels.
    If the linear transformation itself is provided,
    it represents most important components.

    :param matrix: linear transformation matrix of shape (channels, components)
    :type matrix: `numpy.ndarray`
    :param ch_names: list of channel names
    :type ch_names: list of str
    :param pos: list of 2D sensor positions as (sensors, 2), see `mne.viz.plot_topomap`
    :type pos: `numpy.ndarray` | `mne.io.Info`
    :param title: plot title
    :type title: None|str
    :param same_scale: Indicate whether to plot scalpmaps with the same range or not.
    :type same_scale: bool
    :return: figure
    """
    ncols = 8
    nrows = int(np.ceil(matrix.shape[1] / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(1.8 * ncols, 1.8 * nrows))
    if nrows > 1:
        axes = itertools.chain(*axes)  # flatten axes array

    if same_scale:
        vmax = np.abs(matrix).max()
        vmin = -vmax

    for i, ax in enumerate(axes):
        if i >= matrix.shape[1]:
            break
        if not same_scale:
            vmax = np.abs(matrix[:, i]).max()
            vmin = -vmax
        mne.viz.plot_topomap(
            data=matrix[:, i],
            pos=pos,
            names=ch_names,
            vmin=vmin,
            vmax=vmax,
            show_names=False,
            sensors=True,
            outlines='head',
            cmap='RdBu_r',
            axes=ax,
            show=False
        )
        ax.set_title('{}'.format(i))

    plt.tight_layout(2.0)
    if title is not None:
        plt.suptitle(title)

    return fig


def make_dirs(path):
    if not os.path.exists(path):
        try:  # in case that in parallel multiple workers try to do the dir
            os.makedirs(path)
            logging.debug('created %s', path)
        except:
            logging.warning('Attempt to create dir "%s" that already exists.', path)
