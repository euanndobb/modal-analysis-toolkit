"""Build a pydvma dataset from recorded hammer/accelerometer CSVs.

Loads an input (hammer) and output (accelerometer) channel, localises them to
an impulse window, packages them as a :class:`pydvma.datastructure.TimeData`
inside a :class:`pydvma.datastructure.DataSet`, and saves it to disk.

Run with ``uv run python scripts/construct_dataset.py``.
"""

from __future__ import annotations

import datetime

import numpy as np
import pydvma as dvma
from pydvma import datastructure

from heart_sound_analysis.io import load_channel_csv

# --- Parameters ---------------------------------------------------------------
SAMPLING_RATE = 4000
INPUT_CSV = "arm1hammer.csv"
OUTPUT_CSV = "arm1accel.csv"
OUTPUT_DATASET = "arm1_3.npy"
IMPULSE_START_S = 8.4
IMPULSE_FINISH_S = 8.9


def main() -> None:
    """Assemble the two channels into a pydvma dataset and save it."""
    impulse_start = int(SAMPLING_RATE * IMPULSE_START_S)
    impulse_finish = int(SAMPLING_RATE * IMPULSE_FINISH_S)

    time_data_input = load_channel_csv(INPUT_CSV, start=impulse_start, end=impulse_finish)
    time_data_output = load_channel_csv(OUTPUT_CSV, start=impulse_start, end=impulse_finish)

    time_samp = len(time_data_input) / SAMPLING_RATE
    time_axis = np.arange(0, time_samp, 1 / SAMPLING_RATE)

    settings = dvma.MySettings(
        channels=2,
        fs=SAMPLING_RATE,
        stored_time=3,
        pretrig_samples=100,
        device_driver="nidaq",
    )

    t = datetime.datetime.now()
    timestring = (
        f"_{t.year}_{t.month}_{t.day}_at_{t.hour}_{t.minute}_{t.second}"
    )

    timedata = datastructure.TimeData(
        time_axis,
        np.stack([time_data_input, time_data_output]).T,
        settings,
        timestamp=t,
        timestring=timestring,
        test_name="Test Input",
    )

    dataset = datastructure.DataSet()
    dataset.add_to_dataset(timedata)
    dvma.save_data(dataset, OUTPUT_DATASET)
    print(f"Saved dataset to {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()
