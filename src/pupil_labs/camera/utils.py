from typing import Optional

import numpy as np
import numpy.typing as npt
import pupil_labs.camera.types as CT
from numpy.lib.recfunctions import structured_to_unstructured

# import cv2
# import numpy as np
# camera_matrix = np.array(
#     [
#         [766.3037717610379, 0.0, 559.7158729463123],
#         [0.0, 765.4514012936911, 537.2187571096966],
#         [0.0, 0.0, 1.0],
#     ]
# )
# dist_coeffs = np.array(
#     [
#         -1.25717871e-01,
#         1.00917472e-01,
#         4.06447571e-04,
#         -1.77695080e-04,
#         1.73092861e-02,
#         2.04495899e-01,
#         8.64089826e-03,
#         6.42843389e-02,
#     ]
# )

# # an array of pixel coords at edges of world image:
# distorted = np.array([(2.0, 2.0), (1088.0, 1080.0)])

# # inaccurate undistorted version
# undistorted = cv2.undistortPoints(
#     distorted, camera_matrix, dist_coeffs
# )
# redistorted = cv2.projectPoints(
#     cv2.convertPointsToHomogeneous(undistorted_pixel_coords),
#     np.zeros((1, 1, 3)),
#     np.zeros((1, 1, 3)),
#     camera_matrix,
#     dist_coeffs,
# )[0]
# # +/-10 px compared to original
# # array([[[  10.20868463,    9.95140237]],
# #        [[1081.74248024, 1073.62690134]]])

# # accurate undistorted version
# better_undistorted = cv2.undistortPointsIter(
#     distorted,
#     camera_matrix,
#     dist_coeffs,
#     None,
#     None,
#     (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 0.0001),
# )
# better_redistorted = cv2.projectPoints(
#     cv2.convertPointsToHomogeneous(better_undistorted),
#     np.zeros((1, 1, 3)),
#     np.zeros((1, 1, 3)),
#     camera_matrix,
#     dist_coeffs,
# )[0]

# # +/-0.00001px compared to original
# # array([[[   2.00006271,    2.00006079]],
# #        [[1087.99993641, 1079.99993528]]])


def apply_distortion_model(point, dist_coeffs):
    x, y = point
    r = np.linalg.norm([x, y])

    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs

    scale = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    scale /= 1 + k4 * r**2 + k5 * r**4 + k6 * r**6

    x_dist = scale * x + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_dist = scale * y + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    return np.asarray([x_dist, y_dist])


def to_np_point_array(
    points: CT.Points2DLike | CT.Points3DLike,
    n_coords: int = 2,
) -> npt.NDArray[np.float64]:
    """
    Convert a python, numpy or structured array of points into unstructured

    Examples:
        >>> to_np_point_array([1, 10])
        array([[ 1., 10.]])
        >>> to_np_point_array([(1, 10), (2, 20)])
        array([[ 1., 10.],
               [ 2., 20.]])
        >>> to_np_point_array([(1, 10, 100), (2, 20, 200)])
        array([[ 1., 10.],
               [ 2., 20.]])
        >>> to_np_point_array([(1, 10, 100), (2, 20, 200)], n_coords=3)
        array([[  1.,  10., 100.],
               [  2.,  20., 200.]])
        >>> to_np_point_array([1, 10])
        array([[ 1., 10.]])
        >>> to_np_point_array(
        ...     np.array([(1, 10), (2, 20)],
        ...     dtype=[("x", np.int32), ("y", np.int32)])
        ... )
        array([[ 1., 10.],
               [ 2., 20.]])

    """
    if not len(points):
        return np.array([], dtype=np.float64).reshape((-1, n_coords))

    if hasattr(points, "dtype") and points.dtype.names is not None:
        if n_coords > len(points.dtype.names):
            raise ValueError(
                f"can not convert {len(points.dtype.names)}D points to {n_coords}D"
            )
        np_points = structured_to_unstructured(points, dtype=np.float64)[:, :n_coords]
    else:
        np_points = np.asarray(points, dtype=np.float64).squeeze()
        data_n_coords = (
            np_points.shape[0] if np_points.ndim == 1 else np_points.shape[1]
        )
        if n_coords > data_n_coords:
            raise ValueError(f"can not convert {data_n_coords}D points to {n_coords}D")
        if np_points.ndim == 1:
            np_points = np_points.reshape((-1, len(np_points)))
        np_points = np_points[:, :n_coords]

    return np_points
