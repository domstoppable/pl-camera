import cv2

from . import custom_types as CT


def undistort_points(
    points_2d: CT.Points2D,
    camera_matrix: CT.CameraMatrix,
    distortion_coefficients: CT.DistortionCoefficients | None = None,
    new_camera_matrix: CT.CameraMatrix | None = None,
) -> CT.Points3D:
    points_3d = cv2.undistortPointsIter(
        src=points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
        R=None,
        P=new_camera_matrix,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.0001),
    )

    # Convert to homogeneous coordinates to obtain proper 3D vectors
    points_3d = cv2.convertPointsToHomogeneous(points_3d)

    # Remove unnecessary dimension introduced by OpenCV
    points_3d = points_3d[:, 0, :]

    return points_3d
