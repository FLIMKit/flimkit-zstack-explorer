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


def _masked_grid(values, mask, name, dimensions, spacing):
    """Build an ImageData for `values` and drop every cell where `mask` is
    False, via a threshold filter.

    A volume mapper's own opacity/NaN handling doesn't reliably drop
    individual voxels (tested: NaN cells with nan_opacity=0 still render as
    solid fill), so masked-out voxels are removed from the geometry outright
    rather than made transparent.
    """
    import pyvista as pv
    grid = pv.ImageData()
    grid.dimensions = dimensions
    grid.spacing = spacing
    grid.cell_data[name] = np.where(mask, values, 0).flatten(order='F')
    grid.cell_data['_mask'] = mask.astype(np.float32).flatten(order='F')
    return grid.threshold(0.5, scalars='_mask')


def show(intensity_stack: np.ndarray, lifetime_stack: np.ndarray, *,
         voxel_size_um=(1.0, 1.0, 1.0), title: str = 'FLIMKit 3D Z-stack Explorer'):
    try:
        import pyvista as pv
    except ImportError:
        _missing_pyvista()
        return

    spacing = tuple(float(v) for v in voxel_size_um)[::-1]
    dimensions = tuple(np.array(intensity_stack.shape)[::-1] + 1)

    # Clipped to the 99th percentile, same as FLIMKit's own 2D intensity
    # view (flimkit/UI/fov_preview.py), so a handful of hot pixels don't
    # wash out the whole volume's opacity ramp. Zero-intensity voxels (no
    # signal) are dropped from the geometry entirely.
    intensity_clip = np.percentile(intensity_stack, 99) if intensity_stack.size else 1.0
    intensity_clipped = np.clip(intensity_stack, 0, intensity_clip)
    intensity_grid = _masked_grid(
        intensity_clipped, intensity_stack > 0, 'intensity', dimensions, spacing)

    # Voxels with no per-pixel fit (NaN lifetime) are dropped the same way.
    lifetime_grid = _masked_grid(
        lifetime_stack, np.isfinite(lifetime_stack), 'lifetime_ns', dimensions, spacing)

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
    if intensity_grid.n_cells:
        plotter.add_volume(intensity_grid, scalars='intensity', cmap='inferno',
                            opacity='linear', name='intensity')
    else:
        plotter.add_text('No intensity signal', font_size=10, color='grey')
    plotter.add_text('Intensity', font_size=12)
    plotter.add_axes()

    plotter.subplot(0, 1)
    if lifetime_grid.n_cells:
        lifetime_actor = plotter.add_volume(
            lifetime_grid, scalars='lifetime_ns', cmap='viridis',
            opacity='linear', name='lifetime', show_scalar_bar=False)
        lifetime_actor.mapper.scalar_range = (vmin, vmax)
        plotter.add_scalar_bar('lifetime (ns)', mapper=lifetime_actor.mapper)

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
    else:
        plotter.add_text('No per-pixel fit data', font_size=10, color='grey')
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
