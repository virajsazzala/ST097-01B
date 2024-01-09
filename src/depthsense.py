import cv2
import numpy as np


def generate_disparity_map(limg, rimg):
    """
    Generates the disparity map for the left image.

    args:
        limg - left image from stereo camera
        rimg - right image from stereo camera

    returns:
        disp_map - disparity map for the left image
    """
    sad_window = 6
    block_size = 11
    P1 = 8 * 1 * block_size**2
    P2 = 32 * 1 * block_size**2
    disparities_count = sad_window * 16
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    # using SGBM instead of BM
    matcher = cv2.StereoSGBM_create(
        numDisparities=disparities_count,
        minDisparity=0,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        mode=mode,
    )

    disp_map = matcher.compute(limg, rimg).astype(np.float32) / 16

    return disp_map


def generate_depth_map(tl, tr, kl, disp_map, rectified=True):
    """
    Generate a depth map.

    args:
        tl - translation vector for left camera
        tr - translation vector for right camera
        kl - intrinsic matrix for left camera
        disp_map - disparity map from left camera

        optional:
            rectified(T)- rectification to find baseline

    returns:
        depth_map - depth map for left camera
    """
    if rectified:
        base = tr[0] - tl[0]
    else:
        base = tl[0] - tr[0]

    focal = kl[0][0]

    # correcting for the non-overlapping zone
    disp_map[disp_map == 0.0] = 0.1
    disp_map[disp_map == -1.0] = 0.1

    depth_map = np.ones(disp_map.shape)
    depth_map = focal * base / disp_map

    return depth_map


def extract_depth(limg, rimg, P0, P1, rectified=True):
    """
    Generate the depth map for left camera from stereo images.

    args:
        limg - left image from stereo camera
        rimg - right image from stereo camera
        P0 - projection matrix for left camera
        P1 - projection matrix for right camera

        optional:
            rectified(T)- rectification to find baseline

    returns:
        depth_map - depth map for left camera
    """
    disp_map = generate_disparity_map(limg, rimg)

    kl, rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
    tl = (tl / tl[3])[:3]

    kr, rr, tr, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    tr = (tr / tr[3])[:3]

    depth = generate_depth_map(tl, tr, kl, disp_map)

    return depth
