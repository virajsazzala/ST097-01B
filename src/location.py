import cv2
import numpy as np


def find_displacement(kp1, kp2, k, match, depth, max_depth=3000):
    """
    Estimate the movement of the camera sequentially per frame.

    args:
        kp1 - keypoints in the first image
        kp2 - keypoints in the second image
        k - intrinsic calibration matrix
        match - matched features

        optional:
            depth - depth map of init frame
            max_depth - threshold to ignore matched features

    returns:
        r - estimated 3x3 rotation matrix
        t - estimated 3x1 translation vector
        points1 -- matched feature pixel coordinates in the first image.
        points2 -- matched feature pixel coordinates in the second image.
    """
    r = np.eye(3)
    t = np.zeros((3, 1))
    object_points = np.zeros((0, 3))

    points1 = np.float32([kp1[m.queryIdx].pt for m in match])
    points2 = np.float32([kp2[m.trainIdx].pt for m in match])

    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]

    delete = []
    for i, (u, v) in enumerate(points1):
        z = depth[int(round(v)), int(round(u))]

        if z > max_depth:
            delete.append(i)
            continue

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    points1 = np.delete(points1, delete, 0)
    points2 = np.delete(points2, delete, 0)

    _, rv, t, inlines = cv2.solvePnPRansac(object_points, points2, k, None)
    r = cv2.Rodrigues(rv)[0]

    return r, t, points1, points2

