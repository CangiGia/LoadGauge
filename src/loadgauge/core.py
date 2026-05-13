"""Domain logic: STM validation and load reconstruction.

The forward model is

.. math::

    \\boldsymbol{\\varepsilon}(t) = \\mathbf{C}\\,\\mathbf{f}(t)

with :math:`\\mathbf{C}` the strain-transfer matrix (STM).
Reconstruction inverts the relation through the Moore-Penrose pseudo-inverse:

.. math::

    \\mathbf{f}(t) = \\mathbf{C}^{+}\\,\\boldsymbol{\\varepsilon}(t).
"""

from __future__ import annotations

from loadgauge import linalg
from loadgauge.signals import STM, Signals


def check_stm(stm: STM, cond_threshold: float = 1e3) -> dict:
    """Diagnose whether the STM is suitable for load reconstruction.

    Three checks are performed:

    1. The system must not be under-sensed (``m >= n``).
    2. The STM must have full column rank.
    3. The condition number must be below *cond_threshold*.

    Parameters
    ----------
    stm : STM
        Strain-transfer matrix to evaluate.
    cond_threshold : float, optional
        Maximum acceptable condition number.  Default ``1e3`` is a common
        rule-of-thumb for quasi-static load reconstruction.

    Returns
    -------
    dict
        cond : float
            Condition number of ``stm.matrix``.
        singular_values : NDArray
            Singular values in descending order.
        numerical_rank : int
            Number of non-negligible singular values.
        n_strain : int
            Number of strain channels (rows).
        n_loads : int
            Number of load components (columns).
        usable : bool
            ``True`` only when all three checks pass.
        reason : str
            Human-readable verdict: ``"OK"`` or a description of the failure.
    """
    diag = linalg.condition_report(stm.matrix)
    m, n = stm.n_strain, stm.n_loads

    reasons: list[str] = []
    usable = True

    if m < n:
        usable = False
        reasons.append(
            f"under-sensed: {m} strain channels < {n} load components"
        )

    if diag["numerical_rank"] < n:
        usable = False
        reasons.append(
            f"rank deficient: numerical rank {diag['numerical_rank']} < {n}"
        )

    if diag["cond"] > cond_threshold:
        usable = False
        reasons.append(
            f"ill-conditioned: cond = {diag['cond']:.3e} > {cond_threshold:.3e}"
        )

    return {
        "cond": diag["cond"],
        "singular_values": diag["singular_values"],
        "numerical_rank": diag["numerical_rank"],
        "n_strain": m,
        "n_loads": n,
        "usable": usable,
        "reason": "OK" if usable else "; ".join(reasons),
    }


def reconstruct(
    stm: STM,
    strain: Signals,
    load_units: list[str] | None = None,
) -> Signals:
    """Reconstruct the load history from measured strain via the STM.

    Computes

    .. math::

        \\mathbf{f}(t) = \\mathbf{C}^{+}\\,\\boldsymbol{\\varepsilon}(t)

    where :math:`\\mathbf{C}^{+}` is the Moore-Penrose pseudo-inverse of
    ``stm.matrix``.  The pseudo-inverse is computed once and applied to all
    samples in a single matrix product.

    Parameters
    ----------
    stm : STM
        Strain-transfer matrix to invert.  ``stm.load_channels`` labels the
        channels of the returned :class:`~loadgauge.signals.Signals`.
    strain : Signals
        Measured strain time series.
        ``strain.n_channels`` must equal ``stm.n_strain``.
    load_units : list of str or None, optional
        Units to attach to the reconstructed load channels.  ``None`` leaves
        unit metadata unset.

    Returns
    -------
    Signals
        Reconstructed loads, shape ``(stm.n_loads, strain.n_samples)``.
        ``channels`` is set to ``stm.load_channels``; ``time`` is copied
        from *strain*.

    Raises
    ------
    ValueError
        If ``strain.n_channels != stm.n_strain``.
    """
    if strain.n_channels != stm.n_strain:
        raise ValueError(
            f"strain has {strain.n_channels} channels but STM expects "
            f"{stm.n_strain} ({stm.strain_channels})"
        )

    C_plus = linalg.pseudoinverse(stm.matrix)
    loads_data = C_plus @ strain.data  # (n_loads, n_samples)

    return Signals(
        data=loads_data,
        channels=list(stm.load_channels),
        time=strain.time,
        units=list(load_units) if load_units is not None else None,
    )
