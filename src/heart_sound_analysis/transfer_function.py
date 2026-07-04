"""Frequency-response / transfer-function estimation for vibration tests.

For the forced-vibration experiments an input (hammer impulse or shaker force)
and one or more output (accelerometer) signals are recorded. These functions
estimate the transfer function between input and output either directly from
the FFTs of the raw signals, or via Welch-averaged PSDs, and provide the
coherence used to judge measurement quality.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft
from scipy import signal

DEFAULT_SAMPLING_RATE = 4000
"""Default acquisition sampling rate in Hz for the vibration rigs."""


def fft_frequencies(n_samples: int, sampling_rate: int) -> np.ndarray:
    """Return the (two-sided) FFT frequency axis for ``n_samples`` samples.

    Parameters
    ----------
    n_samples : int
        Number of samples in the time-domain signal.
    sampling_rate : int
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Frequency values in Hz, ``arange(n_samples) / (n_samples / fs)``.
    """
    period = n_samples / sampling_rate
    return np.arange(n_samples) / period


def fft_transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the transfer function directly from the signals' FFTs.

    Computes ``H(f) = FFT(output) / FFT(input)``.

    Parameters
    ----------
    input_signal, output_signal : numpy.ndarray
        Time-domain input (forcing) and output (response) signals. Must be the
        same length.
    sampling_rate : int, optional
        Sampling rate in Hz. Defaults to :data:`DEFAULT_SAMPLING_RATE`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The frequency axis in Hz and the complex transfer function ``H(f)``.
    """
    input_spectrum = fft(input_signal)
    output_spectrum = fft(output_signal)
    frequencies = fft_frequencies(len(input_spectrum), sampling_rate)
    transfer_function = output_spectrum / input_spectrum
    return frequencies, transfer_function


def welch_transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    seg_size: int,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    **welch_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the transfer function from Welch-averaged PSDs.

    Computes ``H(f) = PSD(output) / PSD(input)`` where each PSD is estimated
    with :func:`scipy.signal.welch`. Averaging reduces the influence of
    background noise relative to the raw-FFT estimate.

    Parameters
    ----------
    input_signal, output_signal : numpy.ndarray
        Time-domain input (forcing) and output (response) signals.
    seg_size : int
        Welch segment length (``nperseg``).
    sampling_rate : int, optional
        Sampling rate in Hz. Defaults to :data:`DEFAULT_SAMPLING_RATE`.
    **welch_kwargs
        Additional keyword arguments forwarded to :func:`scipy.signal.welch`
        (e.g. ``window``, ``noverlap``, ``detrend``).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The frequency axis in Hz and the transfer function ``H(f)``.
    """
    frequencies, input_psd = signal.welch(
        input_signal, fs=sampling_rate, nperseg=seg_size, **welch_kwargs
    )
    _, output_psd = signal.welch(
        output_signal, fs=sampling_rate, nperseg=seg_size, **welch_kwargs
    )
    return frequencies, np.asarray(output_psd) / np.asarray(input_psd)


def coherence(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    seg_size: int,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    noverlap: int | None = None,
    **coherence_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the magnitude-squared coherence between input and output.

    Coherence near 1 indicates a linear, noise-free relationship at that
    frequency; lower values flag noise or nonlinearity.

    Parameters
    ----------
    input_signal, output_signal : numpy.ndarray
        Time-domain input and output signals.
    seg_size : int
        Welch segment length (``nperseg``).
    sampling_rate : int, optional
        Sampling rate in Hz. Defaults to :data:`DEFAULT_SAMPLING_RATE`.
    noverlap : int, optional
        Number of overlapping samples between segments. If ``None`` (default),
        half the segment size is used.
    **coherence_kwargs
        Additional keyword arguments forwarded to
        :func:`scipy.signal.coherence`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The frequency axis in Hz and the coherence estimate.
    """
    if noverlap is None:
        noverlap = seg_size // 2
    return signal.coherence(
        input_signal,
        output_signal,
        fs=sampling_rate,
        window="hann",
        nperseg=seg_size,
        noverlap=noverlap,
        **coherence_kwargs,
    )


def to_decibels(values: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Convert a ratio/PSD to a decibel scale.

    Parameters
    ----------
    values : numpy.ndarray
        Values to convert (may be complex, matching the original analysis
        which took ``10*log10`` of a complex transfer function before taking
        the magnitude).
    factor : float, optional
        Decibel factor. ``10`` (default) for power-like quantities, ``20`` for
        amplitude-like quantities.

    Returns
    -------
    numpy.ndarray
        ``factor * log10(values)``.
    """
    return factor * np.log10(values)
