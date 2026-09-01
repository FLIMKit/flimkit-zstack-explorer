FLIMKIT_PLUGIN_API = 1

from flimkit_zstack_explorer.volumes import build_zstack_volumes
from flimkit_zstack_explorer.zarr_io import save_ome_zarr, load_ome_zarr

registered = False
register_error = None
try:
    from flimkit_zstack_explorer import plugin as _plugin
    registered = True
except ImportError as exc:
    register_error = exc

__all__ = [
    'FLIMKIT_PLUGIN_API',
    'build_zstack_volumes',
    'save_ome_zarr',
    'load_ome_zarr',
]
