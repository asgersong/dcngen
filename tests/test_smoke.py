"""Environment smoke test: the WNTR toolchain runs a vanilla hydraulic simulation.

Infrastructure canary, not a domain seam: proves the conda env, the editable
install, and the bundled EPANET solver all work before any dcngen physics exists.
"""

from pathlib import Path

NET1_INP = Path(__file__).parent / "fixtures" / "Net1.inp"


def test_vanilla_wntr_simulation_completes(tmp_path):
    import numpy as np
    import wntr

    wn = wntr.network.WaterNetworkModel(str(NET1_INP))
    results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(tmp_path / "epanet"))

    pressure = results.node["pressure"]
    assert len(pressure) > 1, "expected a multi-timestep horizon"
    assert np.isfinite(pressure.to_numpy()).all()


def test_dcngen_package_is_importable():
    import dcngen

    assert dcngen.__version__
