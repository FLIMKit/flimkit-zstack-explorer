"""Verifies the Tools menu entry is a *guarded* action (per the approved
design: always visible, but refuses to run without a loaded z-stack)
rather than actually disabled/hidden."""
import pytest

flimkit = pytest.importorskip('flimkit', reason='requires a FLIMKit install')

from flimkit_zstack_explorer.plugin import open_3d_explorer


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
