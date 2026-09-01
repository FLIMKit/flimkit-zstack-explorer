"""Standalone popout 3D viewer. Runs in its own process (see plugin.py),
never imports tkinter/flimkit, so it doesn't fight FLIMKit's Tk mainloop
for the main thread.

Run directly with: python -m flimkit_zstack_explorer.viewer --zarr <path>
"""
from __future__ import annotations
import argparse
import sys

import numpy as np

# Same names, same order, as flimkit.utils.display.COLORMAPS, so the FLIM
# volume offers the same palette as FLIMKit's own 2D lifetime view.
COLORMAPS = ['hsv', 'viridis', 'cool', 'hot', 'twilight']


def _missing_pyvista():
    sys.stderr.write(
        'flimkit-zstack-explorer: pyvista is required for the 3D viewer.\n'
        'Install it with:\n\n'
        '    pip install "flimkit-zstack-explorer[gui]"\n\n')
    sys.exit(1)


def _set_lifetime_colormap(actor, name):
    import pyvista as pv
    lut = pv.LookupTable(cmap=name)
    lut.apply_opacity('linear')
    actor.mapper.lookup_table = lut


def show(intensity_stack: np.ndarray, lifetime_stack: np.ndarray, *,
         voxel_size_um=(1.0, 1.0, 1.0), title: str = 'FLIMKit 3D Z-stack Explorer'):
    try:
        import pyvista as pv
    except ImportError:
        _missing_pyvista()
        return

    spacing = tuple(float(v) for v in voxel_size_um)

    # Clipped to the 99th percentile, same as FLIMKit's own 2D intensity
    # view (flimkit/UI/fov_preview.py), so a handful of hot pixels don't
    # wash out the whole volume's opacity ramp.
    intensity_clip = np.percentile(intensity_stack, 99) if intensity_stack.size else 1.0
    intensity_clipped = np.clip(intensity_stack, 0, intensity_clip)

    intensity_grid = pv.ImageData()
    intensity_grid.dimensions = np.array(intensity_stack.shape)[::-1] + 1
    intensity_grid.spacing = spacing[::-1]
    intensity_grid.cell_data['intensity'] = intensity_clipped.flatten(order='F')

    # Background (no-fit / no-signal) voxels are pinned to 0, which is below
    # any real lifetime in the data, so the 'linear' opacity transfer
    # function fades them out automatically without needing a second
    # per-voxel opacity field.
    lifetime_grid = pv.ImageData()
    lifetime_grid.dimensions = np.array(lifetime_stack.shape)[::-1] + 1
    lifetime_grid.spacing = spacing[::-1]
    finite_lifetime = np.where(np.isfinite(lifetime_stack), lifetime_stack, 0.0)
    lifetime_grid.cell_data['lifetime_ns'] = finite_lifetime.flatten(order='F')

    valid_lifetime = lifetime_stack[np.isfinite(lifetime_stack)]
    if valid_lifetime.size:
        lt_min, lt_max = float(valid_lifetime.min()), float(valid_lifetime.max())
        # 2nd/98th percentile default, same as FLIMKit's "Auto" scale button.
        vmin = float(np.percentile(valid_lifetime, 2))
        vmax = float(np.percentile(valid_lifetime, 98))
    else:
        lt_min, lt_max, vmin, vmax = 0.0, 1.0, 0.0, 1.0
    if lt_max <= lt_min:
        lt_max = lt_min + 1.0
    if vmax <= vmin:
        vmin, vmax = lt_min, lt_max

    plotter = pv.Plotter(shape=(1, 2), title=title)

    plotter.subplot(0, 0)
    plotter.add_volume(intensity_grid, scalars='intensity', cmap='inferno',
                        opacity='linear', name='intensity')
    plotter.add_text('Intensity', font_size=12)
    plotter.add_axes()

    plotter.subplot(0, 1)
    lifetime_actor = plotter.add_volume(
        lifetime_grid, scalars='lifetime_ns', cmap='viridis',
        opacity='linear', name='lifetime', show_scalar_bar=False)
    lifetime_actor.mapper.scalar_range = (vmin, vmax)
    plotter.add_scalar_bar('lifetime (ns)', mapper=lifetime_actor.mapper)
    plotter.add_text('FLIM (lifetime)', font_size=12)
    plotter.add_axes()

    def set_vmin(value):
        _, hi = lifetime_actor.mapper.scalar_range
        lo = min(float(value), hi)
        lifetime_actor.mapper.scalar_range = (lo, hi)
        plotter.update_scalar_bar_range((lo, hi))

    def set_vmax(value):
        lo, _ = lifetime_actor.mapper.scalar_range
        hi = max(float(value), lo)
        lifetime_actor.mapper.scalar_range = (lo, hi)
        plotter.update_scalar_bar_range((lo, hi))

    def set_cmap(name):
        _set_lifetime_colormap(lifetime_actor, name)

    plotter.add_slider_widget(
        set_vmin, rng=(lt_min, lt_max), value=vmin, title='lifetime min (ns)',
        pointa=(0.52, 0.92), pointb=(0.98, 0.92), style='modern')
    plotter.add_slider_widget(
        set_vmax, rng=(lt_min, lt_max), value=vmax, title='lifetime max (ns)',
        pointa=(0.52, 0.80), pointb=(0.98, 0.80), style='modern')
    plotter.add_text_slider_widget(
        set_cmap, data=COLORMAPS, value=COLORMAPS.index('viridis'),
        pointa=(0.52, 0.68), pointb=(0.98, 0.68), style='modern')

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
