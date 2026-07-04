"""Compare Welch-averaged transfer functions and coherence across tests.

Loads force / accelerometer / far-field-accelerometer CSVs for several repeat
tests and overlays their frequency response functions (and coherence) on a
single figure. Three plotting helpers are provided at increasing fidelity:
raw-FFT, Welch-averaged, and Welch-averaged with a coherence twin axis.

Run with ``uv run python scripts/tf_comp.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from heart_sound_analysis.io import load_channel_csv
from heart_sound_analysis.transfer_function import (
    coherence,
    fft_transfer_function,
    to_decibels,
    welch_transfer_function,
)

# --- Parameters ---------------------------------------------------------------
SAMPLING_RATE = 5120
SEG_SIZE = 5000
FREQ_LIMITS = (10, 200)

# Marker -> label mapping used by the raw-FFT plot helper.
MARKER_LABELS = {
    "x": "Test 1 - Small",
    "1": "Test 2 - Small",
    "2": "Test 3 - Small",
    "3": "Test 1 - 50g",
    ".": "Test 2 - 50g",
}
DEFAULT_LABEL = "Test 3 - 50g"


def load_test(force_path: str, accel_path: str, add_accel_path: str):
    """Load the force, accelerometer and far-field channels for one test.

    Parameters
    ----------
    force_path, accel_path, add_accel_path : str
        CSV paths for the shaker force input, driving-point accelerometer, and
        far-field accelerometer respectively.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The ``(force, accel, add_accel)`` time-series arrays.
    """
    return (
        load_channel_csv(force_path),
        load_channel_csv(accel_path),
        load_channel_csv(add_accel_path),
    )


def plot_tf(force_time_data, accel_time_data, add_accel_time_data, marker):
    """Plot the far-field transfer function from raw FFTs.

    Parameters
    ----------
    force_time_data, accel_time_data, add_accel_time_data : numpy.ndarray
        Time-series force, driving-point and far-field acceleration signals.
    marker : str
        Matplotlib marker; also selects the series label via ``MARKER_LABELS``.
    """
    frequencies, tf_ff = fft_transfer_function(
        force_time_data, add_accel_time_data, SAMPLING_RATE
    )
    tf_ff_db = to_decibels(tf_ff)

    label = MARKER_LABELS.get(marker, DEFAULT_LABEL)
    plt.title("Transfer Functions")
    plt.scatter(frequencies, np.abs(tf_ff_db), label=label, marker=marker, s=20)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("TF Amplitude (dB)")
    plt.legend()
    plt.xlim([0, 200])


def plot_tf_welch(
    force_time_data,
    accel_time_data,
    add_accel_time_data,
    segsize,
    test_number,
):
    """Plot the Welch-averaged far-field transfer function.

    Parameters
    ----------
    force_time_data, accel_time_data, add_accel_time_data : numpy.ndarray
        Time-series force, driving-point and far-field acceleration signals.
    segsize : int
        Welch segment length (``nperseg``).
    test_number : int or str
        Identifier used in the series label.
    """
    frequencies, tf_ff = welch_transfer_function(
        force_time_data,
        add_accel_time_data,
        seg_size=segsize,
        sampling_rate=SAMPLING_RATE,
        window="hann",
    )
    tf_ff_db = to_decibels(tf_ff)

    plt.title("Transfer Functions")
    plt.plot(frequencies, np.abs(tf_ff_db), label=f"Test {test_number}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("TF Amplitude (dB)")
    plt.legend()
    plt.xlim([10, 200])
    plt.ylim([20, 65])


def plot_tf_welch_coherence(
    force_time_data,
    accel_time_data,
    add_accel_time_data,
    segsize,
    test_number,
    ax1,
    ax2,
):
    """Plot the Welch driving-point transfer function with coherence overlaid.

    Draws the driving-point transfer function on ``ax1`` (log-frequency, dB)
    and the input/output coherence on the twin axis ``ax2``.

    Parameters
    ----------
    force_time_data, accel_time_data, add_accel_time_data : numpy.ndarray
        Time-series force, driving-point and far-field acceleration signals.
    segsize : int
        Welch segment length (``nperseg``); 50% overlap is used.
    test_number : str
        Label for the transfer-function series.
    ax1, ax2 : matplotlib.axes.Axes
        Primary (transfer function) and twin (coherence) axes.

    Returns
    -------
    matplotlib.axes.Axes
        The coherence axis ``ax2``.
    """
    frequencies, tf_dp = welch_transfer_function(
        force_time_data,
        accel_time_data,
        seg_size=segsize,
        sampling_rate=SAMPLING_RATE,
        window="hann",
        noverlap=segsize // 2,
    )
    tf_dp_db = to_decibels(tf_dp)

    coh_freq, coh_dp = coherence(
        force_time_data, accel_time_data, seg_size=segsize, sampling_rate=SAMPLING_RATE
    )

    ax1.set_xlabel("Frequency / Hz")
    ax1.set_ylabel("Transfer Function Amplitude / dB")
    ax1.semilogx(frequencies, np.abs(tf_dp_db), label=f"{test_number} (dp)")
    ax1.tick_params(axis="y")
    ax1.legend(loc="lower right")

    ax2.set_ylabel("Coherence")
    ax2.semilogx(coh_freq, coh_dp, linestyle="dotted", linewidth=2)
    ax2.tick_params(axis="y")

    return ax2


def main() -> None:
    """Overlay the transfer function and coherence for the long-exposure tests."""
    prefix = "long_exposure_again_smallg_413g"
    tests = [
        load_test(
            f"{prefix}_{i}_force.csv",
            f"{prefix}_{i}_accel.csv",
            f"{prefix}_{i}_add_accel.csv",
        )
        for i in (1, 2, 3)
    ]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # second axes sharing the same x-axis

    for i, (force, accel, add_accel) in enumerate(tests, start=1):
        plot_tf_welch_coherence(force, accel, add_accel, SEG_SIZE, f"Test {i}", ax1, ax2)

    plt.xlim(list(FREQ_LIMITS))
    ax2.set_ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Reference: alternative datasets used during the project. Swap the ``prefix``
# in ``main`` (or load explicitly) to reproduce these comparisons.
#
#   "chest_50g_413g_{1,2,3}_{force,accel,add_accel}.csv"   # 50 g preload
#   "arm_smallg_413g_{1,2,3}_{force,accel,add_accel}.csv"  # arm, small head
#
# The raw-FFT (`plot_tf`) and Welch (`plot_tf_welch`) helpers above can be used
# in place of `plot_tf_welch_coherence` for quicker, lower-fidelity comparisons.
# -----------------------------------------------------------------------------
