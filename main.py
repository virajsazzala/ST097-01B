import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.datahandler import Datahandler
from src.depthsense import extract_depth
from src.location import find_displacement
from src.plotting import show_computed_path
from src.featuredetection import extract_features, match_features

handler = Datahandler(sequence='01')
frame_count = handler.frame_count

show_computed_path(plt, handler)

# create homogenous transformation matrix
tot = np.eye(4)
trajectory = np.zeros((frame_count, 3, 4))
trajectory[0] = tot[:3, :]
imheight = handler.img_height
imwidth = handler.img_width

# set frames
handler.reset_frames()
next_img = next(handler.imgs_left)

kl, rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(handler.P0)
tl = (tl / tl[3])[:3]

# video
impath = handler.seq_dir + "image_0/"
cv2.namedWindow("drive", cv2.WINDOW_NORMAL)

start_time = datetime.now()
for i in range(frame_count - 1):
    
    left_img = next_img
    next_img = next(handler.imgs_left)
    right_img = next(handler.imgs_right)

    cv2.imshow("drive", left_img)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

    depth = extract_depth(left_img, right_img, handler.P0, handler.P1)

    # masking non-overlapping zone
    mask = np.zeros(depth.shape, dtype=np.uint8)
    ymax = depth.shape[0]
    xmax = depth.shape[1]
    cv2.rectangle(mask, (96, 0), (xmax, ymax), (255), thickness=-1)

    kp0, des0 = extract_features(left_img, mask)
    kp1, des1 = extract_features(next_img, mask)
    matches = match_features(des0, des1, filter=True)

    # distance from nth frame to n+1th frame
    r, t, p1, p2 = find_displacement(kp0, kp1, kl, matches, depth)

    tmat = np.eye(4)
    tmat[:3, :3] = r
    tmat[:3, 3] = t.T
    tot = tot.dot(np.linalg.inv(tmat))
    trajectory[i + 1, :, :] = tot[:3, :]

    # plotting
    xs = trajectory[: i + 2, 0, 3]
    ys = trajectory[: i + 2, 1, 3]
    zs = trajectory[: i + 2, 2, 3]
    plt.plot(xs, ys, zs, c="green")
    plt.pause(1e-32)

end_time = datetime.now()
print(f"Processing Time {i + 1}: {end_time - start_time}")

plt.close()
cv2.destroyAllWindows()
