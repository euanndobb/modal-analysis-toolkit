"""Compute per-position SNR averages from averaged heart-sound PSDs.

Loads a stack of averaged PSDs (one record per chest position, each holding a
PSD per channel), extracts the heart-sound frequency band, divides the S1-peak
PSD by the diastole PSD to form an SNR, then averages over the band to give a
single SNR value per position.

Run with ``uv run python scripts/compute_snr.py`` after adjusting the paths and
parameters below.
"""

from __future__ import annotations

import numpy as np

from heart_sound_analysis.psd import extract_peak_band
from heart_sound_analysis.snr import average_over_band, compute_snr

# --- Parameters (edit these for your dataset) ---------------------------------
INPUT_PATH = "Euan5all_average_psds.npy"
OUTPUT_PATH = "Euan_5_snr_averages.npy"
SAMPLING_RATE = 4000
SEG_SIZE = 390
LOW_FREQ = 10.0
HIGH_FREQ = 140.0
# Channel indices within each record: (S1 peak, systole, S2 peak, diastole).
SIGNAL_CHANNEL = 0
NOISE_CHANNEL = 3


def main() -> None:
    """Load PSDs, compute the SNR band average, and save the result."""
    all_average_data = np.load(INPUT_PATH, allow_pickle=True)

    channels = extract_peak_band(
        all_average_data,
        seg_size=SEG_SIZE,
        low_freq=LOW_FREQ,
        high_freq=HIGH_FREQ,
        sampling_rate=SAMPLING_RATE,
    )

    signal_psd = channels[SIGNAL_CHANNEL]
    noise_psd = channels[NOISE_CHANNEL]

    snr = compute_snr(signal_psd, noise_psd)
    snr_band_average = average_over_band(snr)

    np.save(OUTPUT_PATH, snr_band_average)
    print(f"Saved {snr_band_average.shape} SNR averages to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
