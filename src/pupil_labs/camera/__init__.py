"""Top-level entry-point for the <project_name> package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # noqa

try:
    __version__ = version("pupil_labs.camera")
except PackageNotFoundError:
    # package is not installed
    pass

from .radial import CameraRadial

__all__ = ["__version__", "CameraRadial"]
