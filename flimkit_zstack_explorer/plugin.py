from flimkit.plugins import tool


def _parent(app):
    return getattr(app, 'root', None) or app


def _deps_ok():
    try:
        import zarr  # noqa: F401
        import ome_zarr  # noqa: F401
    except ImportError:
        return False
    return True


@tool(id='zstack_3d_explorer', label='3D Z-stack Explorer...', menu='Tools', order=850)
def open_3d_explorer(app):
    from tkinter import filedialog, messagebox

    parent = _parent(app)
    panel = getattr(app, '_fov_preview', None)
    if not getattr(panel, '_zstack', None):
        messagebox.showinfo(
            'No z-stack loaded',
            'This tool builds a 3D volume from a loaded z-stack.\n\n'
            'Load or fit a z-stack first (FOV mode > Z-stack), then run '
            'this tool again.',
            parent=parent)
        return

    if not _deps_ok():
        messagebox.showerror(
            'Missing dependency',
            'The 3D Z-stack Explorer needs zarr and ome-zarr.\n\n'
            'Install with:\n\n    pip install flimkit-zstack-explorer',
            parent=parent)
        return

    out_path = filedialog.asksaveasfilename(
        parent=parent, title='Save 3D volume as OME-Zarr',
        defaultextension='.zarr', initialfile='zstack_volume.zarr',
        filetypes=[('OME-Zarr store', '*.zarr'), ('All files', '*')])
    if not out_path:
        return

    _run_build_and_view(app, parent, out_path)


def _run_build_and_view(app, parent, out_path):
    import threading
    from flimkit.UI.progress_window import ProgressWindow

    n_slices = len(app._fov_preview._zstack)
    win = ProgressWindow(parent, task_name='Building 3D volume')
    win.set_progress(0, maximum=n_slices)

    def on_progress(i, n):
        app.root.after(0, lambda: win.set_progress(i, maximum=n))

    def worker():
        from flimkit_zstack_explorer.volumes import build_zstack_volumes
        from flimkit_zstack_explorer.zarr_io import save_ome_zarr

        try:
            intensity, lifetime = build_zstack_volumes(app, progress=on_progress)
            app.root.after(0, lambda: win.set_status('Saving OME-Zarr...'))
            save_ome_zarr(out_path, intensity, lifetime)
        except Exception as exc:
            app.root.after(0, win.close)
            app.root.after(0, lambda exc=exc: _show_error(parent, exc))
            return
        app.root.after(0, win.close)
        app.root.after(0, lambda: _launch_viewer(out_path, parent))

    threading.Thread(target=worker, daemon=True).start()


def _show_error(parent, exc):
    from tkinter import messagebox
    messagebox.showerror('3D Z-stack Explorer failed',
                          f'{type(exc).__name__}: {exc}', parent=parent)


def _launch_viewer(zarr_path, parent):
    import subprocess
    import sys
    from tkinter import messagebox

    try:
        subprocess.Popen([
            sys.executable, '-m', 'flimkit_zstack_explorer.viewer',
            '--zarr', str(zarr_path),
        ])
    except Exception as exc:
        messagebox.showerror(
            '3D Z-stack Explorer',
            f'Volume saved to {zarr_path}, but the viewer failed to start:\n'
            f'{type(exc).__name__}: {exc}\n\n'
            'Install the viewer with: pip install "flimkit-zstack-explorer[gui]"',
            parent=parent)
