import numpy as np
import pytest

pytest.importorskip('pyvista', reason='requires the [gui] extra')

import pyvista as pv

from flimkit_zstack_explorer.viewer import show


def test_show_builds_offscreen_without_error(monkeypatch):
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)

    rng = np.random.default_rng(0)
    intensity = rng.poisson(50, size=(4, 12, 16)).astype(np.float32)
    lifetime = rng.uniform(1.5, 3.5, size=(4, 12, 16)).astype(np.float32)
    lifetime[0] = np.nan

    show(intensity, lifetime, voxel_size_um=(2.0, 0.5, 0.5))
