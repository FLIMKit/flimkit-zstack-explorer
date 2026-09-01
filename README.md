# flimkit-zstack-explorer

3D intensity and FLIM volume explorer for FLIMKit z-stacks. Takes a z-stack
you've already loaded or fitted in FLIMKit, stacks its per-slice intensity
and lifetime maps into `(Z, Y, X)` volumes, saves them as an OME-Zarr store,
and opens a popout 3D viewer with the two volumes side by side.

## What it does

- Adds **Tools > 3D Z-stack Explorer...** to FLIMKit.
- Only does anything once a z-stack is loaded (FOV mode > Z-stack); otherwise
  it tells you to load one first.
- Walks every slice in the loaded z-stack through the FOV preview panel's own
  display code (the same path used when you drag the z-slider), so the
  volume matches what's on screen slice by slice.
- Saves a 2-channel OME-Zarr store (channel 0 = intensity, channel 1 =
  lifetime in ns, NaN where a slice had no per-pixel fit).
- Opens a [PyVista](https://pyvista.org) volume viewer in its own process —
  intensity on the left, FLIM lifetime on the right, camera-linked — so it
  doesn't compete with FLIMKit's Tkinter UI for the main thread.

## Install

```
pip install 'flimkit-zstack-explorer[gui]'
```

Quote it — in zsh (macOS's default shell), an unquoted `[gui]` is a glob
pattern and the command fails with "no matches found" without installing
anything.

`[gui]` adds PyVista, which does the actual 3D rendering. Without it, the
volume-building and OME-Zarr saving still work, but the viewer step will
tell you it's missing and how to install it.

Install it into the same Python environment FLIMKit itself runs from —
a separate environment or a different `python`/`pip` on your `PATH` will
install cleanly but leave FLIMKit unable to see it. Verify with:

```bash
python -c "import flimkit_zstack_explorer, pyvista; print('ok')"
```

run from the same terminal you launch FLIMKit from.

FLIMKit discovers this through the `flimkit.plugins` entry point, so there's
nothing else to wire up. On start it appears under Tools.

## From code

```python
from flimkit_zstack_explorer import build_zstack_volumes, save_ome_zarr, load_ome_zarr

# app is FLIMKit's running App instance, with a z-stack loaded in its
# FOV preview panel
intensity_stack, lifetime_stack = build_zstack_volumes(app)
save_ome_zarr('volume.zarr', intensity_stack, lifetime_stack,
              voxel_size_um=(2.0, 0.5, 0.5))

intensity_stack, lifetime_stack = load_ome_zarr('volume.zarr')
```

To view a store outside FLIMKit:

```
python -m flimkit_zstack_explorer.viewer --zarr volume.zarr
```

## Why a separate process for the viewer

FLIMKit's UI is Tkinter; PyVista's is VTK. Running VTK's render loop on a
background thread inside a Tkinter process is unreliable, especially on
macOS, where GUI toolkits generally expect to own the main thread. Building
the volume and saving to disk happens inside FLIMKit; the popout viewer is a
plain `python -m flimkit_zstack_explorer.viewer` subprocess reading the
saved store, so it gets its own main thread and doesn't touch Tkinter at
all.

## Development

```
pip install -e .[gui,test]
pytest
```

`tests/test_volumes.py` exercises the z-stack-to-array logic against a fake
FOV preview panel, so it runs without Tkinter or a real FLIMKit install.
`tests/test_zarr_io.py` round-trips real arrays through zarr/ome-zarr.
