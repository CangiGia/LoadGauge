"""Data containers for LoadGauge.

Two lightweight dataclasses model the entities in the reconstruction workflow:

* :class:`STM` — the strain-transfer matrix (typically obtained from a
  finite-element analysis), together with row and column labels.
* :class:`Signals` — a time-series container used both for the measured strain
  input and for the reconstructed loads output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class STM:
    """Strain-transfer matrix.

    Encapsulates the linear map

    .. math::

        \\boldsymbol{\\varepsilon} = \\mathbf{C}\\,\\mathbf{f}

    together with the human-readable names of its rows (strain channels) and
    columns (load components).

    Parameters
    ----------
    matrix : NDArray, shape (m, n)
        Numerical matrix mapping the *n* load components onto the *m* strain
        channels.
    strain_channels : list of str
        Names of the *m* strain measurement points (rows of *matrix*).
    load_channels : list of str
        Names of the *n* load components (columns of *matrix*).
    name : str, optional
        Free-form identifier for the STM (e.g. project or model name).

    Raises
    ------
    ValueError
        If *matrix* is not 2-D, or if the lengths of *strain_channels* /
        *load_channels* do not match the matrix dimensions.

    Notes
    -----
    Reconstruction requires :math:`m \\geq n`.  This is **not** enforced here
    so that ill-sized matrices can still be inspected.  Use
    :func:`loadgauge.core.check_stm` for validation.

    Examples
    --------
    >>> import numpy as np
    >>> from loadgauge.signals import STM
    >>> C = np.eye(3, 2)
    >>> stm = STM(C, strain_channels=["SG1","SG2","SG3"], load_channels=["Fx","Fy"])
    >>> stm.n_strain, stm.n_loads
    (3, 2)
    """

    matrix: NDArray
    strain_channels: list[str]
    load_channels: list[str]
    name: str = ""

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError(
                f"matrix must be 2-D, got shape {self.matrix.shape}"
            )
        m, n = self.matrix.shape
        if len(self.strain_channels) != m:
            raise ValueError(
                f"len(strain_channels) = {len(self.strain_channels)} "
                f"does not match number of rows ({m})"
            )
        if len(self.load_channels) != n:
            raise ValueError(
                f"len(load_channels) = {len(self.load_channels)} "
                f"does not match number of columns ({n})"
            )

    @property
    def n_strain(self) -> int:
        """Number of strain channels (rows of the matrix)."""
        return self.matrix.shape[0]

    @property
    def n_loads(self) -> int:
        """Number of load components (columns of the matrix)."""
        return self.matrix.shape[1]


@dataclass
class Signals:
    """Time-series container for strain measurements or reconstructed loads.

    The same class is reused for both the input (measured strain) and the
    output (reconstructed loads) of the reconstruction pipeline.  The channel
    names and units carry the semantic meaning.

    Parameters
    ----------
    data : NDArray, shape (n_channels, n_samples)
        Sample values, one row per channel.
    channels : list of str
        Channel names; ``len(channels)`` must equal ``data.shape[0]``.
    time : NDArray of shape (n_samples,) or None, optional
        Timestamps.  ``None`` when no time vector is available; downstream
        code will fall back to sample indices.
    units : list of str or None, optional
        Physical unit per channel; when provided, length must equal
        ``data.shape[0]``.

    Raises
    ------
    ValueError
        If *data* is not 2-D, or if any supplied length is inconsistent with
        ``data.shape``.

    Examples
    --------
    Measured strain::

        Signals(data=eps, channels=["SG1","SG2","SG3"],
                time=t, units=["µε","µε","µε"])

    Reconstructed loads::

        Signals(data=f, channels=["Fx","Fy","Mz"],
                time=t, units=["N","N","N·m"])
    """

    data: NDArray
    channels: list[str]
    time: NDArray | None = None
    units: list[str] | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(
                f"data must be 2-D (n_channels, n_samples), got shape {self.data.shape}"
            )
        n_ch, n_s = self.data.shape
        if len(self.channels) != n_ch:
            raise ValueError(
                f"len(channels) = {len(self.channels)} does not match "
                f"data.shape[0] = {n_ch}"
            )
        if self.time is not None:
            self.time = np.asarray(self.time, dtype=float)
            if self.time.ndim != 1 or self.time.shape[0] != n_s:
                raise ValueError(
                    f"time must be 1-D with length {n_s}, got shape {self.time.shape}"
                )
        if self.units is not None and len(self.units) != n_ch:
            raise ValueError(
                f"len(units) = {len(self.units)} does not match "
                f"n_channels = {n_ch}"
            )

    @property
    def n_channels(self) -> int:
        """Number of channels (rows of *data*)."""
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of time samples (columns of *data*)."""
        return self.data.shape[1]
