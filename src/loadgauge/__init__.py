"""LoadGauge — Load reconstruction from strain measurements.

Quick-start
-----------
>>> from loadgauge import io, core
>>> stm    = io.load_stm("stm.csv",
...                      strain_channels=["SG1","SG2","SG3","SG4"],
...                      load_channels=["Fx","Fy","Fz","Mz"])
>>> print(core.check_stm(stm))
>>> strain = io.load_strain("acquisition.xlsx", time_column=0)
>>> loads  = core.reconstruct(stm, strain)
"""

from __future__ import annotations

from loadgauge.signals import STM, Signals

__version__ = "0.1.0.dev0"
__all__ = ["STM", "Signals"]
