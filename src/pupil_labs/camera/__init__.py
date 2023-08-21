"""Top-level entry-point for the pupil_labs.camera package"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.camera")
except PackageNotFoundError:
    # package is not installed
    pass

from .radial import CameraRadial

__all__ = ["__version__", "CameraRadial"]
