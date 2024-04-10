from typing import Any, NamedTuple, Optional

import cv2
import numpy as np
from pupil_labs.camera.utils import to_np_point_array

from . import types as CT


def undistort_image(
    image: CT.Image,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike,
) -> CT.Image:
    distortion_coefficients = coalesce_distortion_coefficients(distortion_coefficients)
    return cv2.undistort(image, camera_matrix, distortion_coefficients)


def undistort_points(
    points_2d: CT.Points2DLike,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
    new_camera_matrix: Optional[CT.CameraMatrixLike] = None,
) -> CT.Points3D:
    coalesced_distortion_coefficients = coalesce_distortion_coefficients(
        distortion_coefficients
    )
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    np_points_2d = to_np_point_array(points_2d).squeeze()
    np_undistorted_points_2d = cv2.undistortPointsIter(
        src=np_points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=coalesced_distortion_coefficients,
        R=None,
        P=new_camera_matrix,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.0001),
    )

    np_points_3d = convert_points_to_homogeneous(np_undistorted_points_2d)
    return np_points_3d.squeeze()


def convert_points_to_homogeneous(points: CT.Points2DLike) -> CT.Points3D:
    np_points_3d = to_np_point_array(points)

    if len(np_points_3d.shape) == 3 and np_points_3d.shape[1] > 1:
        np_points_3d = np_points_3d.squeeze()

    if np_points_3d.shape[1] != 3:
        np_points_3d = cv2.convertPointsToHomogeneous(np_points_3d)

    return np_points_3d


def _project_points(
    points_3d: CT.Points3DLike,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
):
    rvec = tvec = np.zeros((1, 1, 3))

    coalesced_distortion_coefficients = coalesce_distortion_coefficients(
        distortion_coefficients
    )
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    np_points_3d = convert_points_to_homogeneous(points_3d)

    projected, jacobian = cv2.projectPoints(
        objectPoints=np_points_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=coalesced_distortion_coefficients,
    )
    return np.array(projected).astype(np.float64).squeeze(), jacobian


def project_points(
    points_3d: CT.Points3DLike,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
) -> CT.Points2D:
    return _project_points(points_3d, camera_matrix, distortion_coefficients)[0]


def project_points_with_jacobian(
    points_3d: CT.Points3DLike,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
) -> tuple[CT.Points2D, np.ndarray]:
    return _project_points(points_3d, camera_matrix, distortion_coefficients)


def undistort_rectify_map(
    camera_matrix: CT.CameraMatrixLike,
    width: int,
    height: int,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
    new_camera_matrix: Optional[CT.CameraMatrixLike] = None,
) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
    distortion_coefficients = coalesce_distortion_coefficients(distortion_coefficients)

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,  # type: ignore
    )
    return map1, map2


def coalesce_distortion_coefficients(
    distortion_coefficients: CT.DistortionCoefficientsLike,
):
    if distortion_coefficients is None:
        distortion_coefficients = []
    return np.asarray(distortion_coefficients, dtype=np.float64)
