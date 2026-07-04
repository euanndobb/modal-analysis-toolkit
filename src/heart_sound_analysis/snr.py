"""Signal-to-noise-ratio (SNR) helpers for heart-sound amplitude analysis.

The SNR regime normalises heart-sound amplitudes by dividing a "signal" PSD
(e.g. the S1 peak) by a "noise" PSD (e.g. diastole), so that amplitudes can be
compared across chest positions independently of background noise.
"""

from __future__ import annotations

import numpy as np


def compute_snr(signal_psd: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
    """Compute the element-wise signal-to-noise ratio of two PSDs.

    Parameters
    ----------
    signal_psd : numpy.ndarray
        PSD of the signal of interest (e.g. the S1 peak).
    noise_psd : numpy.ndarray
        PSD of the noise reference (e.g. diastole). Must broadcast against
        ``signal_psd``.

    Returns
    -------
    numpy.ndarray
        The ratio ``signal_psd / noise_psd``.
    """
    return np.asarray(signal_psd) / np.asarray(noise_psd)


def average_over_band(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average a per-record spectrum down to a single value per record.

    Parameters
    ----------
    data : numpy.ndarray
        Array whose rows are spectra/bands to be collapsed, of shape
        ``(n_records, n_bins)`` by default.
    axis : int, optional
        Axis over which to average. Defaults to the last axis, giving one
        mean value per record.

    Returns
    -------
    numpy.ndarray
        The mean of ``data`` along ``axis``.
    """
    return np.asarray(data).mean(axis=axis)
