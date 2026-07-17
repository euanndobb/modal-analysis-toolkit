# Modal Analysis Toolkit

Analytical tools from my MEng project, *Understanding heart-sound propagation in the human chest*. They cover experimental control for vibrational testing, noise reduction, frequency-domain analysis - for the purposes of vibrational modal analysis [What is Modal Analysis?](https://dewesoft.com/blog/what-is-modal-analysis).

> Heart sound region segmenation (systole, diastole) and VHD classification models are omitted from this repository

## Contents

**Vibrational Data Acquisition**
- Drive a shaker frequency sweep (linear chirp) while recording force and accelerometer channels via an NI DAQ.
- Save each channel to CSV for every chest position / test combination.
- Assemble recorded channels into `pydvma` datasets ready for analysis.

**Heart Sound Noise Reduction & Visualisation**
- Segment time-series recordings into systole, diastole, and the S1/S2 peaks
- Compute a normalised signal-to-noise ratio (SNR) for comparing amplitudes
- Generate heart-sound amplitude heatmaps across chest positions

**Vibrational Characterisation**
- Compute complex power spectral densities (PSDs) for the forcing input and accelerometer response.
- Produce frequency response functions (FRFs) and coherence plots for series of 1D tests.
- Perform regression of 2D resonant frequencies, mode shapes and damping using modal analysis techniques (for small amplitude oscillation)

## Layout

```
src/heart_sound_analysis/   # reusable library
  psd.py                    # Welch PSD estimation and band extraction
  snr.py                    # SNR normalisation
  transfer_function.py      # FRF and coherence estimation
  io.py                     # load recorded CSV channel data
  daq.py                    # NI DAQ acquisition (requires hardware)
scripts/                    # runnable acquisition / analysis scripts
  sweep_chest_daq.py        # drive a shaker sweep and record responses
  construct_dataset.py      # build a pydvma dataset from CSVs
  compute_snr.py            # per-position SNR averages from PSDs
  compute_tf.py             # FFT transfer functions (hammer tests)
  tf_comp.py                # Welch transfer functions + coherence
```

## Usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                # create .venv and install deps
uv run python scripts/compute_snr.py   # run a script
```

Each script has a parameter block at the top - edit the file paths, sampling
rate, segment size, and frequency band to point at your own data.
