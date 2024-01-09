import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.datahandler import Datahandler
from src.depthsense import extract_depth
from src.location import find_displacement
from src.featuredetection import extract_features, match_features

from src.helpers import display_plot

handler = Datahandler("05", sample=False)
frame_count = handler.frame_count

# computed plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=-20, azim=270)
xs = handler.gt[:, 0, 3]
ys = handler.gt[:, 1, 3]
zs = handler.gt[:, 2, 3]
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
ax.plot(xs, ys, zs, c="grey")

# create homogenous transformation matrix
T_tot = np.eye(4)
trajectory = np.zeros((frame_count, 3, 4))
trajectory[0] = T_tot[:3, :]
imheight = handler.img_height
imwidth = handler.img_width

# set frames
handler.reset_frames()
next_img = next(handler.imgs_left)

kl, rl, tl, _, _, _, _ = cv2.decomposeProjectionMatrix(handler.P0)
tl = (tl / tl[3])[:3]

# video
impath = f"./dataset/sequences/{handler.sequence}/image_0"
cv2.namedWindow("drive", cv2.WINDOW_NORMAL)

for i in range(frame_count - 1):
    start_time = datetime.now()

    left_img = next_img
    next_img = next(handler.imgs_left)
    right_img = next(handler.imgs_right)

    imp = os.path.join(impath, handler.limgs[i])
    fr = cv2.imread(imp)
    cv2.imshow("drive", fr)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

    # get depth map
    depth = extract_depth(left_img, right_img, handler.P0, handler.P1)

    # masking non-overlapping zone
    mask = np.zeros(depth.shape, dtype=np.uint8)
    ymax = depth.shape[0]
    xmax = depth.shape[1]
    cv2.rectangle(mask, (96, 0), (xmax, ymax), (255), thickness=-1)

    # get features from img
    kp0, des0 = extract_features(left_img, mask)
    kp1, des1 = extract_features(next_img, mask)
    matches = match_features(des0, des1, filter=True)

    # distance from nth frame to n+1th frame
    r, t, p1, p2 = find_displacement(kp0, kp1, kl, matches, depth)

    tmat = np.eye(4)
    tmat[:3, :3] = r
    tmat[:3, 3] = t.T
    T_tot = T_tot.dot(np.linalg.inv(tmat))
    trajectory[i + 1, :, :] = T_tot[:3, :]

    end_time = datetime.now()

    print("Time to compute frame {}:".format(i + 1), end_time - start_time)

    # plotting
    xs = trajectory[: i + 2, 0, 3]
    ys = trajectory[: i + 2, 1, 3]
    zs = trajectory[: i + 2, 2, 3]
    plt.plot(xs, ys, zs, c="green")
    plt.pause(1e-32)

plt.close()
cv2.destroyAllWindows()
