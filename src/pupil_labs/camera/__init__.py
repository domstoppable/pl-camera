"""Top-level entry-point for the pupil_labs.camera package"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("pupil_labs.camera")

from .radial import CameraRadial

__all__ = ["CameraRadial", "__version__"]
