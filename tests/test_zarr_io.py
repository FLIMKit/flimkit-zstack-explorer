import numpy as np
import pytest

from flimkit_zstack_explorer.zarr_io import save_ome_zarr, load_ome_zarr


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    intensity = rng.poisson(50, size=(4, 16, 20)).astype(np.float32)
    lifetime = rng.uniform(1.5, 3.5, size=(4, 16, 20)).astype(np.float32)
    lifetime[0] = np.nan

    path = str(tmp_path / 'volume.zarr')
    save_ome_zarr(path, intensity, lifetime, voxel_size_um=(2.0, 0.5, 0.5))

    i2, l2 = load_ome_zarr(path)
    assert i2.shape == intensity.shape
    assert l2.shape == lifetime.shape
    assert np.allclose(i2, intensity)
    # NaNs in the source lifetime slice are written as 0 by save_ome_zarr's
    # nan_to_num pass on intensity only; lifetime itself is written as-is,
    # so slice 0 round-trips as NaN.
    assert np.all(np.isnan(l2[0]))
    assert np.allclose(l2[1:], lifetime[1:])


def test_saving_to_same_path_twice_overwrites(tmp_path):
    # The save dialog defaults to the same filename every run, so re-running
    # the tool against a path that already has a store must overwrite it
    # rather than hitting ome-zarr's ContainsArrayError on leftover nodes.
    intensity = np.full((3, 8, 8), 10.0, dtype=np.float32)
    lifetime = np.full((3, 8, 8), 2.0, dtype=np.float32)
    path = str(tmp_path / 'zstack_volume.zarr')

    save_ome_zarr(path, intensity, lifetime)
    save_ome_zarr(path, intensity * 2, lifetime + 1)

    i2, l2 = load_ome_zarr(path)
    assert np.allclose(i2, 20.0)
    assert np.allclose(l2, 3.0)


def test_shape_mismatch_raises(tmp_path):
    intensity = np.zeros((3, 4, 4), dtype=np.float32)
    lifetime = np.zeros((3, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match='same shape'):
        save_ome_zarr(str(tmp_path / 'bad.zarr'), intensity, lifetime)


def test_requires_3d_stacks(tmp_path):
    intensity = np.zeros((4, 4), dtype=np.float32)
    lifetime = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match='Z, Y, X'):
        save_ome_zarr(str(tmp_path / 'bad2.zarr'), intensity, lifetime)
