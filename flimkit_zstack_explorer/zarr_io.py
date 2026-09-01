from __future__ import annotations
import os
import shutil
from typing import Optional, Sequence

import numpy as np

CHANNELS = ('intensity', 'lifetime_ns')


def save_ome_zarr(
    path: str,
    intensity_stack: np.ndarray,
    lifetime_stack: np.ndarray,
    *,
    voxel_size_um: Optional[Sequence[float]] = None,
) -> str:
    """Write intensity + lifetime volumes as a 2-channel OME-Zarr store.

    `intensity_stack` and `lifetime_stack` must both be (Z, Y, X) and share
    a shape. Channel 0 is intensity, channel 1 is lifetime in ns (NaN where
    a slice had no fit). `voxel_size_um`, if given, is (z, y, x) physical
    spacing used for the store's scale metadata; otherwise voxels are unit
    spacing. Returns `path`.
    """
    if intensity_stack.shape != lifetime_stack.shape:
        raise ValueError(
            f'intensity {intensity_stack.shape} and lifetime {lifetime_stack.shape} '
            'stacks must have the same shape')
    if intensity_stack.ndim != 3:
        raise ValueError(f'expected (Z, Y, X) stacks, got shape {intensity_stack.shape}')

    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    # ome-zarr's 'w' mode doesn't clear an existing store, so writing to the
    # same path twice (the save dialog defaults to the same filename every
    # run) hits leftover array nodes from the previous write.
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

    data = np.stack([
        np.nan_to_num(intensity_stack, nan=0.0).astype(np.float32),
        lifetime_stack.astype(np.float32),
    ])

    store = parse_url(str(path), mode='w').store
    root = zarr.group(store=store)

    nz, ny, nx = intensity_stack.shape
    chunks = (1, 1, min(256, ny), min(256, nx))
    write_image(image=data, group=root, axes='czyx', scaler=None,
                storage_options=dict(chunks=chunks))

    scale = [1.0] + [float(v) for v in (voxel_size_um or (1.0, 1.0, 1.0))]
    root.attrs['flimkit_zstack_explorer'] = {
        'channels': list(CHANNELS),
        'voxel_size_um': scale[1:],
    }
    return str(path)


def load_ome_zarr(path: str):
    """Read back a store written by `save_ome_zarr`.

    Returns (intensity_stack, lifetime_stack), each (Z, Y, X) numpy arrays.
    """
    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    reader = Reader(parse_url(str(path)))
    node = next(iter(reader()))
    data = np.asarray(node.data[0])
    if data.shape[0] != 2:
        raise ValueError(f'expected a 2-channel store (intensity, lifetime), got shape {data.shape}')
    return data[0], data[1]
