import abc
import typing as T
import warnings
from pathlib import Path

import numpy as np

from . import types as CT


class CameraBase(abc.ABC):
    pixel_width: int
    pixel_height: int
    camera_matrix: CT.CameraMatrix
    distortion_coefficients: T.Optional[CT.DistortionCoefficientsLike] = None

    @property
    def focal_length(self) -> float:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        return (fx + fy) / 2

    @abc.abstractmethod
    def undistort_image(self, image: CT.Image) -> CT.Image:
        raise NotImplementedError()

    @abc.abstractmethod
    def undistort_points(
        self, points_2d: CT.Points3DLike, use_distortion: bool = True
    ) -> CT.Points3D:
        raise NotImplementedError()

    @abc.abstractmethod
    def project_points(
        self, points_3d: CT.Points3DLike, use_distortion: bool = True
    ) -> CT.Points2D:
        raise NotImplementedError()

    def undistort_points_on_image_plane(
        self, points_2d: CT.Points2DLike
    ) -> CT.Points2D:
        points_3d = self.undistort_points(points_2d, use_distortion=True)
        points_2d = self.project_points(points_3d, use_distortion=False)
        return points_2d

    def distort_points_on_image_plane(self, points_2d: CT.Points2DLike) -> CT.Points2D:
        points_3d = self.undistort_points(points_2d, use_distortion=False)
        points_2d = self.project_points(points_3d, use_distortion=True)
        return points_2d
