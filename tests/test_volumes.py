import numpy as np
import pytest

from flimkit_zstack_explorer.volumes import build_zstack_volumes


class FakePanel:
    """Stands in for flimkit.UI.fov_preview.FOVPreviewPanel: enough of its
    z-stack surface for build_zstack_volumes to drive without Tkinter."""

    def __init__(self, zstack):
        self._zstack = zstack
        self._z_i = 0
        self._intensity_map = None
        self._lifetime_map = None
        self.shown = []

    def display_fit_results(self, ptu_path, fit_result, _keep_zstack=False):
        self._intensity_map = fit_result['intensity']
        self._lifetime_map = fit_result.get('lifetime')
        self.shown.append(ptu_path)

    def load_fov(self, ptu_path, _keep_zstack=False):
        self._intensity_map = np.ones((4, 4), dtype=np.float32)
        self._lifetime_map = None
        self.shown.append(ptu_path)

    def _show_zstack_slice(self, i):
        self._z_i = i
        desc = self._zstack[i]
        if desc.get('fit_result') is not None:
            self.display_fit_results(desc.get('ptu_path'), desc['fit_result'], _keep_zstack=True)
        else:
            self.load_fov(desc.get('ptu_path'), _keep_zstack=True)


class FakeApp:
    def __init__(self, panel):
        self._fov_preview = panel
        self.root = None


def _slice(z, intensity, lifetime=None):
    fit_result = {'intensity': np.array(intensity, dtype=np.float32)}
    if lifetime is not None:
        fit_result['lifetime'] = np.array(lifetime, dtype=np.float32)
    return {'z': z, 'ptu_path': f'slice_{z}.ptu', 'fit_result': fit_result}


def test_builds_stack_from_zstack_slices():
    intensity = np.full((3, 3), 10.0)
    lifetime = np.full((3, 3), 2.5)
    zstack = [_slice(0, intensity, lifetime), _slice(1, intensity * 2, lifetime + 1)]
    panel = FakePanel(zstack)
    app = FakeApp(panel)

    intensity_stack, lifetime_stack = build_zstack_volumes(app)

    assert intensity_stack.shape == (2, 3, 3)
    assert lifetime_stack.shape == (2, 3, 3)
    assert np.allclose(intensity_stack[0], 10.0)
    assert np.allclose(intensity_stack[1], 20.0)
    assert np.allclose(lifetime_stack[1], 3.5)


def test_missing_lifetime_becomes_nan_plane():
    intensity = np.full((2, 2), 5.0)
    zstack = [_slice(0, intensity, lifetime=None)]
    panel = FakePanel(zstack)
    app = FakeApp(panel)

    _, lifetime_stack = build_zstack_volumes(app)

    assert lifetime_stack.shape == (1, 2, 2)
    assert np.all(np.isnan(lifetime_stack[0]))


def test_no_zstack_raises():
    panel = FakePanel(zstack=None)
    app = FakeApp(panel)

    with pytest.raises(RuntimeError, match='No z-stack'):
        build_zstack_volumes(app)


def test_mismatched_slice_shapes_raise():
    zstack = [
        _slice(0, np.zeros((3, 3))),
        _slice(1, np.zeros((4, 4))),
    ]
    panel = FakePanel(zstack)
    app = FakeApp(panel)

    with pytest.raises(RuntimeError, match='same size'):
        build_zstack_volumes(app)


def test_restores_original_slice_after_building():
    zstack = [_slice(0, np.zeros((2, 2))), _slice(1, np.ones((2, 2)))]
    panel = FakePanel(zstack)
    panel._z_i = 1
    app = FakeApp(panel)

    build_zstack_volumes(app)

    assert panel._z_i == 1
