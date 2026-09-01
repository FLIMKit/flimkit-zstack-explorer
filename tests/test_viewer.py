import numpy as np
import pytest

pytest.importorskip('pyvista', reason='requires the [gui] extra')

import pyvista as pv

from flimkit_zstack_explorer.viewer import (
    COLORMAPS, _masked_grid, _set_lifetime_colormap, show,
)


def test_masked_grid_drops_exact_cell_count():
    shape = (10, 20, 24)
    dims = tuple(np.array(shape)[::-1] + 1)
    rng = np.random.default_rng(0)

    intensity = rng.poisson(50, size=shape).astype(np.float32)
    zero_idx = rng.choice(intensity.size, 500, replace=False)
    intensity.flat[zero_idx] = 0
    grid = _masked_grid(intensity, intensity > 0, 'intensity', dims, (1.0, 1.0, 1.0))
    assert grid.n_cells == intensity.size - 500

    lifetime = rng.uniform(1.5, 3.5, size=shape).astype(np.float32)
    lifetime[3] = np.nan  # a whole slice with no per-pixel fit
    grid = _masked_grid(lifetime, np.isfinite(lifetime), 'lifetime_ns', dims, (1.0, 1.0, 1.0))
    assert grid.n_cells == lifetime.size - shape[1] * shape[2]


def test_masked_grid_all_dropped_yields_empty_grid():
    shape = (2, 4, 4)
    dims = tuple(np.array(shape)[::-1] + 1)
    values = np.zeros(shape, dtype=np.float32)
    grid = _masked_grid(values, values > 0, 'intensity', dims, (1.0, 1.0, 1.0))
    assert grid.n_cells == 0


def test_show_handles_all_zero_intensity_and_all_nan_lifetime(monkeypatch):
    # Must not crash even when every voxel is masked out in one or both
    # volumes (add_volume errors on an empty grid, so show() has to skip it).
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)

    intensity = np.zeros((3, 8, 8), dtype=np.float32)
    lifetime = np.full((3, 8, 8), np.nan, dtype=np.float32)

    show(intensity, lifetime)


def _spy_on_add_volume(monkeypatch):
    calls = []
    original = pv.Plotter.add_volume

    def spy(self, grid, **kwargs):
        actor = original(self, grid, **kwargs)
        calls.append((grid, kwargs, actor))
        return actor

    monkeypatch.setattr(pv.Plotter, 'add_volume', spy)
    return calls


def test_show_builds_offscreen_without_error(monkeypatch):
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)

    rng = np.random.default_rng(0)
    intensity = rng.poisson(50, size=(4, 12, 16)).astype(np.float32)
    lifetime = rng.uniform(1.5, 3.5, size=(4, 12, 16)).astype(np.float32)
    lifetime[0] = np.nan

    show(intensity, lifetime, voxel_size_um=(2.0, 0.5, 0.5))


def test_intensity_is_clipped_to_99th_percentile(monkeypatch):
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)
    calls = _spy_on_add_volume(monkeypatch)

    rng = np.random.default_rng(1)
    intensity = rng.poisson(50, size=(3, 20, 20)).astype(np.float32)
    expected_clip = float(np.percentile(intensity, 99))
    intensity[0, 0, 0] = 1_000_000.0  # a hot pixel outlier
    lifetime = np.full((3, 20, 20), 2.0, dtype=np.float32)

    show(intensity, lifetime)

    intensity_grid = calls[0][0]
    assert intensity_grid.cell_data['intensity'].max() <= expected_clip + 1e-3
    assert intensity_grid.cell_data['intensity'].min() >= 0


def test_lifetime_scalar_range_defaults_to_2nd_98th_percentile(monkeypatch):
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)
    calls = _spy_on_add_volume(monkeypatch)

    rng = np.random.default_rng(2)
    intensity = np.full((3, 16, 16), 10.0, dtype=np.float32)
    lifetime = rng.uniform(1.0, 4.0, size=(3, 16, 16)).astype(np.float32)
    expected_vmin = float(np.percentile(lifetime, 2))
    expected_vmax = float(np.percentile(lifetime, 98))

    show(intensity, lifetime)

    lifetime_actor = calls[1][2]
    lo, hi = lifetime_actor.mapper.scalar_range
    assert lo == pytest.approx(expected_vmin, abs=1e-3)
    assert hi == pytest.approx(expected_vmax, abs=1e-3)


def test_set_lifetime_colormap_swaps_colors_and_keeps_opacity_ramp():
    grid = pv.ImageData()
    grid.dimensions = (5, 5, 5)
    grid.spacing = (1, 1, 1)
    grid.cell_data['lifetime_ns'] = np.linspace(0, 1, 64).astype(np.float32)

    plotter = pv.Plotter(off_screen=True)
    actor = plotter.add_volume(grid, scalars='lifetime_ns', cmap='viridis',
                                opacity='linear')
    original_colors = actor.mapper.lookup_table.values.copy()

    for name in COLORMAPS:
        _set_lifetime_colormap(actor, name)
        assert actor.mapper.lookup_table is not None
        # Opacity ramps from transparent to opaque across the table, same
        # shape 'linear' produces for every colormap.
        alpha = actor.mapper.lookup_table.values[:, -1]
        assert alpha[0] < alpha[-1]

    _set_lifetime_colormap(actor, 'hot')
    assert not np.array_equal(actor.mapper.lookup_table.values, original_colors)
    plotter.close()
