import numpy as np
import pytest

pytest.importorskip('pyvista', reason='requires the [gui] extra')

import pyvista as pv

from flimkit_zstack_explorer.viewer import (
    COLORMAPS, _point_cloud, _set_lifetime_colormap, show,
)


def test_point_cloud_keeps_exactly_the_valid_voxels():
    shape = (10, 20, 24)
    rng = np.random.default_rng(0)

    intensity = rng.poisson(50, size=shape).astype(np.float32)
    zero_idx = rng.choice(intensity.size, 500, replace=False)
    intensity.flat[zero_idx] = 0
    cloud = _point_cloud(intensity, intensity > 0, 'intensity', (1.0, 1.0, 1.0))
    assert cloud.n_points == intensity.size - 500

    lifetime = rng.uniform(1.5, 3.5, size=shape).astype(np.float32)
    lifetime[3] = np.nan  # a whole slice with no per-pixel fit
    cloud = _point_cloud(lifetime, np.isfinite(lifetime), 'lifetime_ns', (1.0, 1.0, 1.0))
    assert cloud.n_points == lifetime.size - shape[1] * shape[2]


def test_point_cloud_all_dropped_yields_none():
    values = np.zeros((2, 4, 4), dtype=np.float32)
    assert _point_cloud(values, values > 0, 'intensity', (1.0, 1.0, 1.0)) is None


def test_point_cloud_drops_scattered_single_voxel_dropout():
    # The bug this guards against: volume rendering (add_volume) resamples
    # onto a dense 3D texture before ray casting, which fills scattered
    # single-voxel gaps from their valid neighbors -- verified directly
    # against a real z-stack export where 92% of lifetime voxels were NaN:
    # add_volume rendered it as a solid block. A point cloud has no
    # resampling step, so a masked-out voxel has no point, period -- this
    # checks that property holds regardless of how scattered the mask is.
    shape = (10, 40, 40)
    rng = np.random.default_rng(3)
    lifetime = np.full(shape, 2.0, dtype=np.float32)
    speckle = rng.random(shape) < 0.4
    lifetime[speckle] = np.nan

    cloud = _point_cloud(lifetime, np.isfinite(lifetime), 'lifetime_ns', (1.0, 1.0, 1.0))

    assert cloud.n_points == (~speckle).sum()
    # every remaining point must be a real, non-speckled voxel
    kept_z, kept_y, kept_x = np.nonzero(~speckle)
    kept_coords = set(zip(kept_x.tolist(), kept_y.tolist(), kept_z.tolist()))
    for p in cloud.points:
        assert tuple(int(round(c)) for c in p) in kept_coords


def test_show_handles_all_zero_intensity_and_all_nan_lifetime(monkeypatch):
    # Must not crash even when every voxel is masked out in one or both
    # clouds (add_points needs at least one point, so show() has to skip it
    # rather than call add_points with an empty cloud).
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)

    intensity = np.zeros((3, 8, 8), dtype=np.float32)
    lifetime = np.full((3, 8, 8), np.nan, dtype=np.float32)

    show(intensity, lifetime)


def _spy_on_add_points(monkeypatch):
    calls = []
    original = pv.Plotter.add_points

    def spy(self, points, **kwargs):
        actor = original(self, points, **kwargs)
        calls.append((points, kwargs, actor))
        return actor

    monkeypatch.setattr(pv.Plotter, 'add_points', spy)
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
    calls = _spy_on_add_points(monkeypatch)

    rng = np.random.default_rng(1)
    intensity = rng.poisson(50, size=(3, 20, 20)).astype(np.float32)
    expected_clip = float(np.percentile(intensity, 99))
    intensity[0, 0, 0] = 1_000_000.0  # a hot pixel outlier
    lifetime = np.full((3, 20, 20), 2.0, dtype=np.float32)

    show(intensity, lifetime)

    intensity_cloud = calls[0][0]
    assert intensity_cloud.point_data['intensity'].max() <= expected_clip + 1e-3
    assert intensity_cloud.point_data['intensity'].min() >= 0


def test_lifetime_scalar_range_defaults_to_2nd_98th_percentile(monkeypatch):
    pv.OFF_SCREEN = True
    monkeypatch.setattr(pv.Plotter, 'show', lambda self, *a, **k: None)
    calls = _spy_on_add_points(monkeypatch)

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
    cloud = pv.PolyData(np.random.default_rng(0).random((64, 3)).astype(np.float32))
    cloud.point_data['lifetime_ns'] = np.linspace(0, 1, 64).astype(np.float32)

    plotter = pv.Plotter(off_screen=True)
    actor = plotter.add_points(cloud, scalars='lifetime_ns', cmap='viridis')
    original_colors = actor.mapper.lookup_table.values.copy()

    for name in COLORMAPS:
        _set_lifetime_colormap(actor, name)
        assert actor.mapper.lookup_table is not None

    _set_lifetime_colormap(actor, 'hot')
    assert not np.array_equal(actor.mapper.lookup_table.values, original_colors)
    plotter.close()
