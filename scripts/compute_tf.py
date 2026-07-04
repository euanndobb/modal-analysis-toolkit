"""Plot FFT-based transfer functions for hammer-impulse vibration tests.

For each test, the hammer (input) and accelerometer (output) CSVs are loaded,
localised to the impulse window, and their transfer function is estimated from
the raw FFTs and plotted in dB against frequency.

Run with ``uv run python scripts/compute_tf.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from heart_sound_analysis.io import load_channel_csv
from heart_sound_analysis.transfer_function import fft_transfer_function, to_decibels

# --- Parameters ---------------------------------------------------------------
SAMPLING_RATE = 4000
FILENAME_PREFIX = "leg_6cm"
FREQ_LIMITS = (0, 250)

# Marker -> (test label, marker size) for the scatter plot.
MARKER_STYLES = {
    "x": ("Test 1", 50),
    "1": ("Test 2", 80),
    "+": ("Test 3", 80),
    "2": ("Test 4", 80),
}
DEFAULT_STYLE = ("Test 4", 80)


def plot_transfer_function(
    start: float,
    finish: float,
    test: int,
    marker: str,
    sampling_rate: int = SAMPLING_RATE,
) -> None:
    """Load one test's CSVs and add its transfer function to the current plot.

    Parameters
    ----------
    start, finish : float
        Start and end times in seconds of the impulse window to analyse.
    test : int
        Test number, used to build the CSV filenames.
    marker : str
        Matplotlib marker; also selects the label/size via ``MARKER_STYLES``.
    sampling_rate : int, optional
        Sampling rate in Hz. Defaults to :data:`SAMPLING_RATE`.
    """
    impulse_start = int(sampling_rate * start)
    impulse_finish = int(sampling_rate * finish)

    hammer = load_channel_csv(
        f"{FILENAME_PREFIX}{test}hammer.csv", start=impulse_start, end=impulse_finish
    )
    accel = load_channel_csv(
        f"{FILENAME_PREFIX}{test}accel.csv", start=impulse_start, end=impulse_finish
    )

    frequencies, transfer_function = fft_transfer_function(hammer, accel, sampling_rate)
    transfer_function_db = to_decibels(transfer_function)

    label, size = MARKER_STYLES.get(marker, DEFAULT_STYLE)
    plt.scatter(
        frequencies, np.abs(transfer_function_db), label=label, marker=marker, s=size
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("TF Amplitude")
    plt.legend()
    plt.xlim(list(FREQ_LIMITS))


def main() -> None:
    """Plot the transfer functions for the configured set of tests."""
    plot_transfer_function(4.4, 4.8, 1, "x")
    plot_transfer_function(5.4, 6.1, 1, "1")
    plot_transfer_function(9.4, 9.7, 1, "+")
    plot_transfer_function(6.85, 7.05, 1, "3")
    plt.show()


if __name__ == "__main__":
    main()
