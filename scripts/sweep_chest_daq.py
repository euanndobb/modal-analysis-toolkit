"""Drive a shaker frequency sweep and record chest-phantom responses.

Generates a linear chirp, plays it out through a shaker (DAQ output) while
recording force and accelerometer inputs, and saves each channel to CSV for
every position/test combination. Requires connected NI DAQ hardware.

Run with ``uv run python scripts/sweep_chest_daq.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.signal import chirp

from heart_sound_analysis.daq import DAQInput, DAQOutput, run_daq

# --- Acquisition parameters ---------------------------------------------------
SAMPLING_RATE = 5120
RUNTIME_S = 40  # Should match the length of the frequency sweep.
FREQ_MIN = 10
FREQ_MAX = 500

# Output filename metadata.
NAME = "chest_full_1"
SIZE = "19mm"
PRELOAD = "419g"

POSITIONS = [3, 4, 5, 6, 7, 8, 9]
TESTS = [1, 2, 3, 4, 5]

SETTLE_BEFORE_S = 40  # Wait before starting the sweep sequence.
SETTLE_BETWEEN_POSITIONS_S = 15


def build_chirp(sampling_rate: int, runtime_s: float, freq_min: float, freq_max: float) -> np.ndarray:
    """Return a linear chirp sweeping ``freq_min`` -> ``freq_max``.

    Parameters
    ----------
    sampling_rate : int
        Sampling rate in Hz.
    runtime_s : float
        Duration of the chirp in seconds.
    freq_min, freq_max : float
        Start and end frequencies of the sweep in Hz.

    Returns
    -------
    numpy.ndarray
        The chirp waveform.
    """
    t = np.arange(0, runtime_s * sampling_rate) / sampling_rate
    return chirp(t, f0=freq_min, f1=freq_max, t1=t.max(), method="linear")


def main() -> None:
    """Run the full position/test sweep and save each channel to CSV."""
    accel_input = DAQInput(name="Sens_1", module="2", channel="ai1", ch_type="iepe")
    force_input = DAQInput(name="Sens_2", module="2", channel="ai2", ch_type="iepe")
    add_accel_input = DAQInput(name="Sens_3", module="2", channel="ai3", ch_type="iepe")

    arr = build_chirp(SAMPLING_RATE, RUNTIME_S, FREQ_MIN, FREQ_MAX)
    shaker_output = DAQOutput(module="3", channel="ao0", signal=arr)

    time.sleep(SETTLE_BEFORE_S)

    for position in POSITIONS:
        for test in TESTS:
            run_daq(
                [accel_input, force_input, add_accel_input],
                output=shaker_output,
                fs=SAMPLING_RATE,
                duration=None,
            )

            accel_input_arr = np.array(accel_input.signal)
            force_input_arr = np.array(force_input.signal)
            add_accel_input_arr = np.array(add_accel_input.signal)

            stem = f"{NAME}_{SIZE}_{PRELOAD}_{position}_{test}"
            pd.DataFrame(force_input_arr).to_csv(f"{stem}_force.csv")
            pd.DataFrame(accel_input_arr).to_csv(f"{stem}_accel.csv")
            pd.DataFrame(add_accel_input_arr).to_csv(f"{stem}_add_accel.csv")

        time.sleep(SETTLE_BETWEEN_POSITIONS_S)


if __name__ == "__main__":
    main()
