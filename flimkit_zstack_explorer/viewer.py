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

DEFAULT_POINT_SIZE = 4


def _missing_pyvista():
    sys.stderr.write(
        'flimkit-zstack-explorer: pyvista is required for the 3D viewer.\n'
        'Install it with:\n\n'
        '    pip install "flimkit-zstack-explorer[gui]"\n\n')
    sys.exit(1)


def _set_lifetime_colormap(actor, name):
    # No opacity ramp here: every point already represents a real, valid
    # voxel (masking is handled by the point simply not existing), so
    # fading by lifetime value would misleadingly make short lifetimes look
    # less "real" than long ones. Points are fully opaque.
    import pyvista as pv
    actor.mapper.lookup_table = pv.LookupTable(cmap=name)


def _point_cloud(values, mask, scalar_name, spacing_xyz):
    """Build a point cloud of only the voxels where `mask` is True.

    Real FLIM z-stacks are dominated by background: in one exported stack,
    92% of lifetime voxels had no per-pixel fit. Volume rendering
    (add_volume) resamples the data onto a dense 3D texture before ray
    casting, and that resampling fills scattered single-voxel gaps from
    their valid neighbors -- verified directly: masking those voxels out of
    an UnstructuredGrid and volume-rendering it still showed a solid block
    for scattered per-pixel dropout, even though a single large contiguous
    dropped region rendered correctly as empty. A point cloud has no
    interpolation step to paper over gaps with -- a voxel with no data
    simply has no point, so there is nothing to render there.
    """
    import pyvista as pv
    z, y, x = np.nonzero(mask)
    if z.size == 0:
        return None
    sx, sy, sz = spacing_xyz
    points = np.column_stack([x * sx, y * sy, z * sz]).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud.point_data[scalar_name] = values[mask].astype(np.float32)
    return cloud


def show(intensity_stack: np.ndarray, lifetime_stack: np.ndarray, *,
         voxel_size_um=(1.0, 1.0, 1.0), title: str = 'FLIMKit 3D Z-stack Explorer'):
    try:
        import pyvista as pv
    except ImportError:
        _missing_pyvista()
        return

    sz, sy, sx = (float(v) for v in voxel_size_um)

    # Clipped to the 99th percentile, same as FLIMKit's own 2D intensity
    # view (flimkit/UI/fov_preview.py), so a handful of hot pixels don't
    # wash out the color scale. Zero-intensity voxels (no signal) have no
    # point at all.
    intensity_clip = np.percentile(intensity_stack, 99) if intensity_stack.size else 1.0
    intensity_clipped = np.clip(intensity_stack, 0, intensity_clip)
    intensity_cloud = _point_cloud(
        intensity_clipped, intensity_stack > 0, 'intensity', (sx, sy, sz))

    # Voxels with no per-pixel fit (NaN lifetime) get no point either.
    lifetime_cloud = _point_cloud(
        lifetime_stack, np.isfinite(lifetime_stack), 'lifetime_ns', (sx, sy, sz))

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
    if intensity_cloud is not None:
        plotter.add_points(intensity_cloud, scalars='intensity', cmap='inferno',
                            point_size=DEFAULT_POINT_SIZE,
                            render_points_as_spheres=False, name='intensity')
    else:
        plotter.add_text('No intensity signal', font_size=10, color='grey')
    plotter.add_text('Intensity', font_size=12)
    plotter.add_axes()

    plotter.subplot(0, 1)
    if lifetime_cloud is not None:
        lifetime_actor = plotter.add_points(
            lifetime_cloud, scalars='lifetime_ns', cmap='viridis',
            point_size=DEFAULT_POINT_SIZE, render_points_as_spheres=False,
            name='lifetime', show_scalar_bar=False)
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
