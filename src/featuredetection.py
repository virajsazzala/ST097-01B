import cv2
import numpy as np


def extract_features(img, mask=None):
    """
    Extract keypoints and descriptors from the image.

    args:
        img - a grayscale image

    returns:
        keypoints - extracted features from the image
        descriptors - keypoint descriptors from the image
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, mask)

    return keypoints, descriptors


def match_features(d1, d2, filter=True, threshold=0.5, k=2):
    """
    Match features from images.

    args:
        d1 - keypoint descriptors in first image
        d2 - keypoint descriptors in second image

        optional:
            filter - get the best features, given threshold
            dthreshold - max allowed relative distance b/w best matches (0.0, 1.0)
                       - good threshold (~0.5)
            k - total number of neighbors to match each feature

    returns:
        best_matches - top matches based on given threshold
    """
    matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)

    matches = matcher.knnMatch(d1, d2, k=k)
    matches = sorted(matches, key=lambda x: x[0].distance)

    if filter:
        best_matches = []
        for i, j in matches:
            if i.distance <= threshold * j.distance:
                best_matches.append(i)

        matches = best_matches

    return matches
