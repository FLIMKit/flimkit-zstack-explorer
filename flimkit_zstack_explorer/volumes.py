from __future__ import annotations
import threading
from typing import Callable, Optional, Tuple

import numpy as np


def _run_on_ui_thread(app, callback: Callable):
    """Run `callback` on the Tkinter main thread and block for its result.

    FLIMKit's own FOV preview panel is Tkinter/matplotlib state and is not
    thread-safe, so anything that touches it (loading a slice, reading back
    the rendered maps) has to happen on the main thread even when this
    function is called from a worker thread.
    """
    if threading.current_thread() is threading.main_thread():
        return callback()
    root = getattr(app, 'root', None)
    if root is None or not callable(getattr(root, 'after', None)):
        raise RuntimeError('FLIMKit UI is not available')
    done = threading.Event()
    outcome = {}

    def run():
        try:
            outcome['value'] = callback()
        except BaseException as error:
            outcome['error'] = error
        finally:
            done.set()

    root.after(0, run)
    done.wait()
    if 'error' in outcome:
        raise outcome['error']
    return outcome.get('value')


def build_zstack_volumes(
    app,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (intensity_stack, lifetime_stack) arrays, each shape (Z, Y, X),
    from the z-stack currently loaded in FLIMKit's FOV preview panel.

    Each z-slice is displayed through the panel's own `display_fit_results` /
    `load_fov` (the same code path the app uses when you drag the z-slider)
    so the maps match what you'd see on screen, rather than re-deriving the
    intensity/lifetime math here. The panel is left showing whatever slice it
    was on before this ran. Raises RuntimeError if no z-stack is loaded, or
    if slices don't all share one shape.

    Safe to call from any thread.
    """
    def get_zstack():
        panel = getattr(app, '_fov_preview', None)
        zstack = getattr(panel, '_zstack', None) if panel is not None else None
        return panel, (list(zstack) if zstack else None)

    panel, zstack = _run_on_ui_thread(app, get_zstack)
    if not zstack:
        raise RuntimeError('No z-stack is loaded in FLIMKit')

    def show_slice(desc):
        fit_result = desc.get('fit_result')
        if fit_result is not None:
            panel.display_fit_results(desc.get('ptu_path'), fit_result, _keep_zstack=True)
        else:
            panel.load_fov(desc.get('ptu_path'), _keep_zstack=True)
        intensity = getattr(panel, '_intensity_map', None)
        lifetime = getattr(panel, '_lifetime_map', None)
        intensity = (np.array(intensity, dtype=np.float32, copy=True)
                     if intensity is not None else None)
        lifetime = (np.array(lifetime, dtype=np.float32, copy=True)
                    if lifetime is not None else None)
        return intensity, lifetime

    original_i = _run_on_ui_thread(app, lambda: getattr(panel, '_z_i', 0))
    intensity_slices = []
    lifetime_slices = []
    try:
        for i, desc in enumerate(zstack):
            intensity, lifetime = _run_on_ui_thread(app, lambda desc=desc: show_slice(desc))
            if intensity is None:
                raise RuntimeError(f'z-slice {i} ({desc.get("z", i)}) has no intensity image')
            intensity_slices.append(intensity)
            lifetime_slices.append(
                lifetime if lifetime is not None
                else np.full(intensity.shape[:2], np.nan, dtype=np.float32))
            if progress is not None:
                progress(i + 1, len(zstack))
    finally:
        def restore():
            panel._z_i = original_i
            panel._show_zstack_slice(original_i)
            sync = getattr(panel, '_sync_z_slider', None)
            if callable(sync):
                sync()
        _run_on_ui_thread(app, restore)

    shape = intensity_slices[0].shape
    for name, slices in (('intensity', intensity_slices), ('lifetime', lifetime_slices)):
        for i, arr in enumerate(slices):
            if arr.shape != shape:
                raise RuntimeError(
                    f'{name} slice {i} has shape {arr.shape}, expected {shape} '
                    '(all z-slices must be the same size)')

    return np.stack(intensity_slices), np.stack(lifetime_slices)
