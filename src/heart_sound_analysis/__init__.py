"""Analytical tools for the MEng project on heart-sound propagation.

The package is split by concern:

- :mod:`heart_sound_analysis.psd` -- Welch PSD estimation and band extraction.
- :mod:`heart_sound_analysis.snr` -- signal-to-noise-ratio normalisation.
- :mod:`heart_sound_analysis.transfer_function` -- frequency-response and
  coherence estimation for the forced-vibration tests.
- :mod:`heart_sound_analysis.io` -- loading recorded CSV channel data.
- :mod:`heart_sound_analysis.daq` -- NI DAQ acquisition (requires hardware).

``daq`` is intentionally not imported here so the analysis modules can be used
without the ``nidaqmx`` package/hardware present.
"""

from __future__ import annotations

from . import io, psd, snr, transfer_function

__all__ = ["io", "psd", "snr", "transfer_function"]
__version__ = "0.1.0"
