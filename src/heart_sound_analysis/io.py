"""Input/output helpers for loading recorded measurement data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_channel_csv(
    path: str,
    column: int = 1,
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    """Load a single data column from a measurement CSV as a NumPy array.

    The acquisition scripts store each channel as a CSV whose first column is
    an index and whose second column holds the samples. This helper reads that
    column and optionally slices it to a sample window.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    column : int, optional
        Zero-based column index to extract. Defaults to ``1`` (the second
        column), matching the acquisition format.
    start, end : int, optional
        Optional sample indices to slice the data (``data[start:end]``). Use to
        localise an impulse or steady-state window.

    Returns
    -------
    numpy.ndarray
        The requested column, sliced to ``[start:end]`` if given.
    """
    frame = pd.read_csv(path)
    values = np.array(frame.iloc[:, column])
    if start is not None or end is not None:
        values = values[start:end]
    return values
