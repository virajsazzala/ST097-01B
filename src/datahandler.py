import os
import cv2
import numpy as np
import pandas as pd


class Datahandler:
    """
    A class to extract required data from the KITTI Dataset.
    """

    def __init__(self, sample=True, sequence='00'):
        """
        optional args:
            sequence - sequence directory number from KITTI Ds.
            sample - uses the sample dataset
        """
        # dirs for poses and ground truths
        if sample:
            self.seq_dir = f"./data/sequences/"
            self.poses_dir = f"./data/poses/poses.txt"
        else:
            # track sequence
            self.sequence = sequence
            self.seq_dir = f"./dataset/sequences/{self.sequence}/"
            self.poses_dir = f"./dataset/poses/{self.sequence}.txt"

        poses = pd.read_csv(self.poses_dir, delimiter=" ", header=None)

        self.limgs = os.listdir(self.seq_dir + "image_0")
        self.rimgs = os.listdir(self.seq_dir + "image_1")
        self.frame_count = len(self.limgs)

        # camera calibration details (grayscale only - P0 & P1)
        # reshape flattened matrices -> 3x4
        calib = pd.read_csv(
            self.seq_dir + "calib.txt", delimiter=" ", header=None, index_col=0
        )
        self.P0 = np.array(calib.iloc[0]).reshape((3, 4))
        self.P1 = np.array(calib.iloc[1]).reshape((3, 4))

        # times & ground truth poses -> arrays
        self.times = np.array(
            pd.read_csv(self.seq_dir + "times.txt", delimiter=" ", header=None)
        )

        # reshape gt flattened matrices -> 3x4
        self.gt = np.zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

        # set images w/ generators for saving memory
        self.reset_frames()
        self.first_limg = cv2.imread(self.seq_dir + "image_0/" + self.limgs[0], 0)
        self.first_rimg = cv2.imread(self.seq_dir + "image_1/" + self.rimgs[0], 0)
        self.second_limg = cv2.imread(self.seq_dir + "image_0/" + self.limgs[1], 0)

        self.img_height, self.img_width = self.first_limg.shape[:2]

    def reset_frames(self):
        """
        Resets all generators to the inital frame.
        """
        self.imgs_left = (
            cv2.imread(self.seq_dir + "image_0/" + name, 0) for name in self.limgs
        )
        self.imgs_right = (
            cv2.imread(self.seq_dir + "image_1/" + name, 0) for name in self.rimgs
        )
