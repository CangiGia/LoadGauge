"""Input/output: read STMs and strain time series from disk or from memory.

Two public entry points are provided, both dispatching on the file extension:

* :func:`load_stm`    — returns a :class:`~loadgauge.signals.STM`.
* :func:`load_strain` — returns a :class:`~loadgauge.signals.Signals`.

Supported formats
-----------------
+----------------+---------------------------+-------------------------------+
| Extension      | Parser                    | Extra kwargs accepted         |
+================+===========================+===============================+
| ``.txt``,      | :func:`numpy.loadtxt`     | ``delimiter``, ``skiprows``,  |
| ``.dat``       |                           | ``usecols``, …                |
+----------------+---------------------------+-------------------------------+
| ``.csv``       | :func:`pandas.read_csv`   | ``sep``, ``header``,          |
|                |                           | ``skiprows``, ``usecols``, …  |
+----------------+---------------------------+-------------------------------+
| ``.xlsx``,     | :func:`pandas.read_excel` | ``sheet_name``, ``header``,   |
| ``.xls``       |                           | ``skiprows``, ``usecols``, …  |
+----------------+---------------------------+-------------------------------+

All format-specific keyword arguments can be forwarded through ``**kwargs``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from loadgauge.signals import STM, Signals

_TXT_EXT = {".txt", ".dat"}
_CSV_EXT = {".csv"}
_XLS_EXT = {".xlsx", ".xls"}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _read_table(path: Path, **kwargs) -> tuple[NDArray, list[str] | None]:
    """Read a tabular file and return ``(values, column_names_or_None)``."""
    ext = path.suffix.lower()
    if ext in _TXT_EXT:
        values = np.loadtxt(path, **kwargs)
        return np.atleast_2d(values), None
    if ext in _CSV_EXT:
        df = pd.read_csv(path, **kwargs)
        return df.to_numpy(dtype=float), list(df.columns.astype(str))
    if ext in _XLS_EXT:
        df = pd.read_excel(path, **kwargs)
        return df.to_numpy(dtype=float), list(df.columns.astype(str))
    raise ValueError(
        f"Unsupported file extension '{ext}'. "
        f"Supported: {_TXT_EXT | _CSV_EXT | _XLS_EXT}"
    )


# ---------------------------------------------------------------------------
# STM loader
# ---------------------------------------------------------------------------


def load_stm(
    source: NDArray | str | Path,
    strain_channels: list[str],
    load_channels: list[str],
    name: str = "",
    **kwargs,
) -> STM:
    """Load a strain-transfer matrix from memory or from a file.

    Parameters
    ----------
    source : NDArray or str or Path
        * :class:`numpy.ndarray` — used directly (shape validated by
          :class:`~loadgauge.signals.STM`).
        * Path-like — file read via the extension-based dispatcher (see
          module docstring for supported formats and extra kwargs).
    strain_channels : list of str
        Names for the rows (strain measurement points).
        Length must match ``matrix.shape[0]``.
    load_channels : list of str
        Names for the columns (load components).
        Length must match ``matrix.shape[1]``.
    name : str, optional
        Free-form identifier for the returned :class:`~loadgauge.signals.STM`.
    **kwargs
        Forwarded to :func:`numpy.loadtxt`, :func:`pandas.read_csv`, or
        :func:`pandas.read_excel` depending on the file extension.

    Returns
    -------
    STM
        Strain-transfer matrix with row and column labels.

    Raises
    ------
    ValueError
        If the file extension is not supported, or if the matrix shape does
        not match the supplied channel lists.

    Examples
    --------
    From a NumPy array::

        stm = load_stm(np.eye(4, 3),
                       strain_channels=["SG1","SG2","SG3","SG4"],
                       load_channels=["Fx","Fy","Fz"])

    From a CSV file (header row ignored, only numeric block used)::

        stm = load_stm("stm.csv",
                       strain_channels=["SG1","SG2","SG3","SG4"],
                       load_channels=["Fx","Fy","Fz"],
                       header=None)

    From a TXT file with ``#`` comments::

        stm = load_stm("stm.txt",
                       strain_channels=["SG1","SG2","SG3","SG4"],
                       load_channels=["Fx","Fy","Fz"],
                       comments="#")
    """
    if isinstance(source, np.ndarray):
        matrix = np.asarray(source, dtype=float)
    else:
        matrix, _ = _read_table(Path(source), **kwargs)
        matrix = matrix.astype(float, copy=False)

    return STM(
        matrix=matrix,
        strain_channels=list(strain_channels),
        load_channels=list(load_channels),
        name=name,
    )


# ---------------------------------------------------------------------------
# Strain time-series loader
# ---------------------------------------------------------------------------


def load_strain(
    source: str | Path,
    time_column: int | str | None = 0,
    channels: list[str] | None = None,
    units: list[str] | None = None,
    **kwargs,
) -> Signals:
    """Load a strain time series from a file.

    Parameters
    ----------
    source : str or Path
        Path to the file.  Extension determines the parser (see module
        docstring).
    time_column : int, str, or None, optional
        Specifies the time column:

        * **int** — zero-based column index (works for all formats).
        * **str** — column name (only for ``.csv`` / ``.xlsx`` with a header).
        * **None** — no time column; ``Signals.time`` will be ``None``.

        Default ``0`` (first column).
    channels : list of str or None, optional
        Channel names for the data columns (excluding the time column).
        When ``None``:

        * ``.csv`` / ``.xlsx`` — names are taken from the file header.
        * ``.txt`` / ``.dat``  — generic names ``"ch0"``, ``"ch1"``, …
          are generated.
    units : list of str or None, optional
        Physical unit per data channel.  Length must equal the number of
        data channels.
    **kwargs
        Forwarded to the underlying reader.  Useful arguments:

        * ``.txt`` / ``.dat``: ``delimiter``, ``skiprows``, ``comments``.
        * ``.csv``: ``sep``, ``decimal``, ``skiprows``, ``usecols``.
        * ``.xlsx``: ``sheet_name``, ``skiprows``, ``usecols``.

    Returns
    -------
    Signals
        Time-series container with ``data`` of shape
        ``(n_channels, n_samples)``.

    Raises
    ------
    ValueError
        If *time_column* is a string but the file has no header, or if the
        column name is not found, or if *channels* / *units* lengths are
        inconsistent with the data.

    Examples
    --------
    Excel file with time in the first column and header row::

        strain = load_strain("acq.xlsx",
                             time_column=0,
                             units=["µε"]*4,
                             sheet_name="Run1")

    Plain-text file with no header, first column = time::

        strain = load_strain("acq.txt",
                             time_column=0,
                             channels=["SG1","SG2","SG3","SG4"],
                             delimiter=",")

    CSV with a named time column::

        strain = load_strain("acq.csv",
                             time_column="Time_s",
                             units=["µε"]*3)
    """
    path = Path(source)
    values, header = _read_table(path, **kwargs)
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    # Resolve time_column to an integer index (or None)
    if time_column is None:
        time_idx: int | None = None
    elif isinstance(time_column, int):
        time_idx = time_column
    else:  # string name
        if header is None:
            raise ValueError(
                f"time_column='{time_column}' is a column name, but the "
                f"file has no header (use an integer index instead)"
            )
        if time_column not in header:
            raise ValueError(
                f"time_column '{time_column}' not found in header: {header}"
            )
        time_idx = header.index(time_column)

    # Split time vector / data columns
    if time_idx is None:
        time = None
        data = values.T
        data_header = header
    else:
        time = values[:, time_idx]
        data_cols = np.delete(values, time_idx, axis=1)
        data = data_cols.T
        data_header = (
            [h for i, h in enumerate(header) if i != time_idx]
            if header is not None
            else None
        )

    # Resolve channel names
    if channels is not None:
        ch = list(channels)
    elif data_header is not None:
        ch = list(data_header)
    else:
        ch = [f"ch{i}" for i in range(data.shape[0])]

    return Signals(data=data, channels=ch, time=time, units=units)
