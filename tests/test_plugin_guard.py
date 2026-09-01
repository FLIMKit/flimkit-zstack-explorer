"""Verifies the Tools menu entry is a *guarded* action (per the approved
design: always visible, but refuses to run without a loaded z-stack)
rather than actually disabled/hidden."""
import pytest

flimkit = pytest.importorskip('flimkit', reason='requires a FLIMKit install')

from flimkit_zstack_explorer.plugin import open_3d_explorer, open_saved_volume


class NoZStackApp:
    root = None
    _fov_preview = None


class EmptyZStackApp:
    root = None

    class _fov_preview:
        _zstack = []


def test_guard_shows_message_when_no_zstack(monkeypatch):
    calls = []
    monkeypatch.setattr('tkinter.messagebox.showinfo',
                         lambda *a, **k: calls.append((a, k)))

    open_3d_explorer(NoZStackApp())

    assert len(calls) == 1
    assert 'z-stack' in calls[0][0][0].lower() or 'z-stack' in calls[0][0][1].lower()


def test_guard_shows_message_when_zstack_empty(monkeypatch):
    calls = []
    monkeypatch.setattr('tkinter.messagebox.showinfo',
                         lambda *a, **k: calls.append((a, k)))

    open_3d_explorer(EmptyZStackApp())

    assert len(calls) == 1


def test_does_not_prompt_for_save_path_without_zstack(monkeypatch):
    monkeypatch.setattr('tkinter.messagebox.showinfo', lambda *a, **k: None)

    def fail_if_called(*a, **k):
        raise AssertionError('should not prompt for a save path without a z-stack')

    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', fail_if_called)

    open_3d_explorer(NoZStackApp())


def test_open_saved_volume_rejects_non_zarr_dir(tmp_path, monkeypatch):
    plain_dir = tmp_path / 'not_a_store'
    plain_dir.mkdir()
    monkeypatch.setattr('tkinter.filedialog.askdirectory', lambda **k: str(plain_dir))

    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror',
                         lambda *a, **k: errors.append((a, k)))

    launched = []
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: launched.append((a, k)))

    open_saved_volume(NoZStackApp())

    assert len(errors) == 1
    assert not launched


def test_open_saved_volume_launches_viewer_for_a_real_store(tmp_path, monkeypatch):
    from flimkit_zstack_explorer.zarr_io import save_ome_zarr
    import numpy as np

    store = tmp_path / 'volume.zarr'
    save_ome_zarr(str(store), np.zeros((2, 4, 4), dtype=np.float32),
                  np.zeros((2, 4, 4), dtype=np.float32))

    monkeypatch.setattr('tkinter.filedialog.askdirectory', lambda **k: str(store))

    launched = []
    monkeypatch.setattr('subprocess.Popen', lambda args, **k: launched.append(args))

    open_saved_volume(NoZStackApp())

    assert len(launched) == 1
    assert '--zarr' in launched[0]
    assert str(store) in launched[0]


def test_open_saved_volume_no_op_when_dialog_cancelled(monkeypatch):
    monkeypatch.setattr('tkinter.filedialog.askdirectory', lambda **k: '')

    launched = []
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: launched.append((a, k)))

    open_saved_volume(NoZStackApp())

    assert not launched
