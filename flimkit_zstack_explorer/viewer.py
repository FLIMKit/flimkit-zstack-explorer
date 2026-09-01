"""Standalone popout 3D viewer. Runs in its own process (see plugin.py),
never imports tkinter/flimkit, so it doesn't fight FLIMKit's Tk mainloop
for the main thread.

Run directly with: python -m flimkit_zstack_explorer.viewer --zarr <path>
"""
from __future__ import annotations
import argparse
import sys

import numpy as np


def _missing_pyvista():
    sys.stderr.write(
        'flimkit-zstack-explorer: pyvista is required for the 3D viewer.\n'
        'Install it with:\n\n'
        '    pip install "flimkit-zstack-explorer[gui]"\n\n')
    sys.exit(1)


def show(intensity_stack: np.ndarray, lifetime_stack: np.ndarray, *,
         voxel_size_um=(1.0, 1.0, 1.0), title: str = 'FLIMKit 3D Z-stack Explorer'):
    try:
        import pyvista as pv
    except ImportError:
        _missing_pyvista()
        return

    spacing = tuple(float(v) for v in voxel_size_um)

    intensity_grid = pv.ImageData()
    intensity_grid.dimensions = np.array(intensity_stack.shape)[::-1] + 1
    intensity_grid.spacing = spacing[::-1]
    intensity_grid.cell_data['intensity'] = intensity_stack.flatten(order='F')

    # Background (no-fit / no-signal) voxels are pinned to 0, which is below
    # any real lifetime in the data, so the 'linear' opacity transfer
    # function used below fades them out automatically without needing a
    # second per-voxel opacity field (PyVista's add_volume only accepts an
    # opacity transfer function keyed by the volume's own scalar range).
    lifetime_grid = pv.ImageData()
    lifetime_grid.dimensions = np.array(lifetime_stack.shape)[::-1] + 1
    lifetime_grid.spacing = spacing[::-1]
    finite_lifetime = np.where(np.isfinite(lifetime_stack), lifetime_stack, 0.0)
    lifetime_grid.cell_data['lifetime_ns'] = finite_lifetime.flatten(order='F')

    plotter = pv.Plotter(shape=(1, 2), title=title)

    plotter.subplot(0, 0)
    plotter.add_volume(intensity_grid, scalars='intensity', cmap='inferno',
                        opacity='linear', name='intensity')
    plotter.add_text('Intensity', font_size=12)
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_volume(lifetime_grid, scalars='lifetime_ns', cmap='turbo',
                        opacity='linear', name='lifetime')
    plotter.add_scalar_bar('lifetime (ns)')
    plotter.add_text('FLIM (lifetime)', font_size=12)
    plotter.add_axes()

    plotter.link_views()
    plotter.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr', required=True, help='Path to an OME-Zarr store '
                         'written by flimkit_zstack_explorer.zarr_io.save_ome_zarr')
    args = parser.parse_args(argv)

    from flimkit_zstack_explorer.zarr_io import load_ome_zarr
    intensity, lifetime = load_ome_zarr(args.zarr)
    show(intensity, lifetime, title=f'FLIMKit 3D Z-stack Explorer - {args.zarr}')


if __name__ == '__main__':
    main()
