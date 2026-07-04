"""Power-spectral-density helpers for heart-sound analysis.

These functions cover the signal-processing steps used to turn segmented,
time-series heart-sound recordings into averaged power spectral densities
(PSDs) via Welch's method, and to isolate the frequency band of interest.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

DEFAULT_SAMPLING_RATE = 4000
"""Default acquisition sampling rate in Hz used throughout the project."""


def clean_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    min_segment_size: int,
    trim_ends: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop segments that are shorter than a minimum length.

    Given paired ``starts``/``ends`` sample indices describing segments of a
    signal, remove any segment whose length (``end - start``) is smaller than
    ``min_segment_size``. Optionally trim the first and last segment first,
    which are often partial/edge segments produced by a segmentation step.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Sample indices of the start and end of each segment. Must be the same
        length and ordered such that ``ends[i]`` corresponds to ``starts[i]``.
    min_segment_size : int
        Minimum allowed segment length in samples. Segments shorter than this
        are discarded.
    trim_ends : bool, optional
        If ``True`` (default), drop the first and last segment before
        filtering. This reproduces the original behaviour where edge segments
        are assumed unreliable.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The filtered ``(starts, ends)`` arrays.
    """
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    if trim_ends and len(starts) > 2:
        starts = starts[1:-1]
        ends = ends[1:-1]

    keep = (ends - starts) >= min_segment_size
    return starts[keep], ends[keep]


def average_psd(
    all_frequencies: np.ndarray,
    all_psd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a stack of PSDs computed on a common frequency grid.

    Parameters
    ----------
    all_frequencies : numpy.ndarray
        Array of shape ``(n_segments, n_bins)`` of frequency values. All rows
        are assumed identical, so the first row is returned as the frequency
        axis.
    all_psd : numpy.ndarray
        Array of shape ``(n_segments, n_bins)`` of PSD values to be averaged
        across the segment axis.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The frequency axis (``all_frequencies[0]``) and the mean PSD across
        segments.
    """
    all_frequencies = np.asarray(all_frequencies)
    all_psd = np.asarray(all_psd)
    return all_frequencies[0], all_psd.mean(axis=0)


def welch_psd(
    data: np.ndarray,
    start: int,
    end: int,
    seg_size: int,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    **welch_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the PSD of a slice of a signal using Welch's method.

    Parameters
    ----------
    data : numpy.ndarray
        The full time-series signal.
    start, end : int
        Sample indices bounding the slice of interest (``data[start:end]``).
    seg_size : int
        Length of each Welch segment (``nperseg``).
    sampling_rate : int, optional
        Sampling rate in Hz. Defaults to :data:`DEFAULT_SAMPLING_RATE`.
    **welch_kwargs
        Additional keyword arguments forwarded to :func:`scipy.signal.welch`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The frequency axis and the PSD estimate for the requested slice.
    """
    segment = data[start:end]
    frequencies, psd = signal.welch(
        segment, fs=sampling_rate, nperseg=seg_size, **welch_kwargs
    )
    return frequencies, psd


def extract_peak_band(
    data: Sequence[Sequence[np.ndarray]],
    seg_size: int,
    low_freq: float = 0.0,
    high_freq: float = 100.0,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    n_channels: int | None = None,
) -> list[np.ndarray]:
    """Slice a frequency band of interest out of per-channel PSD data.

    Each element of ``data`` is expected to hold one PSD array per channel
    (e.g. S1 peak, systole, S2 peak, diastole). This function converts the
    requested frequency band into bin indices given ``seg_size`` and
    ``sampling_rate`` and returns the sliced PSDs grouped by channel.

    This unifies the two ``peak_data`` implementations that previously lived in
    ``functions.py`` and ``compute_snr.py`` (which differed only in their
    hard-coded frequency band and return type).

    Parameters
    ----------
    data : sequence
        Iterable of records, where each record is indexable by channel and
        yields a PSD array, i.e. ``data[i][channel]`` is a 1-D PSD.
    seg_size : int
        Welch segment length used to produce the PSDs. Sets the frequency
        resolution (``sampling_rate / seg_size``).
    low_freq, high_freq : float, optional
        Lower and upper bounds of the frequency band in Hz. Defaults span
        0-100 Hz, the range relevant to heart sounds.
    sampling_rate : int, optional
        Sampling rate in Hz. Defaults to :data:`DEFAULT_SAMPLING_RATE`.
    n_channels : int, optional
        Number of channels per record. If ``None`` (default), inferred from
        the length of the first record.

    Returns
    -------
    list[numpy.ndarray]
        One array per channel of shape ``(n_records, n_bins_in_band)``.
    """
    freq_res = sampling_rate / seg_size
    low_bin = int(np.floor(low_freq / freq_res))
    high_bin = int(np.ceil(high_freq / freq_res) + 1)

    data = list(data)
    if not data:
        return []
    if n_channels is None:
        n_channels = len(data[0])

    channels = [
        np.array([record[channel][low_bin:high_bin] for record in data])
        for channel in range(n_channels)
    ]
    return channels
