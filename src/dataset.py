import os
import json
import cv2
import numpy as np

from .settings import *


class SceneDataset:

    def __init__(self):

        self.images = self.load_images()
        self.poses = self.load_poses()
        self.K = self.camera_matrix()

    def load_images(self):

        frames = {}

        image_files = sorted([
            file for file in os.listdir(IMAGE_FOLDER)
            if file.endswith('.png')
        ])

        for file in image_files:

            idx = int(
                file.split('_')[1].split('.')[0]
            )

            path = os.path.join(
                IMAGE_FOLDER,
                file
            )

            image = cv2.imread(path)

            frames[idx] = image

        return frames

    def load_poses(self):

        with open(POSE_FILE, 'r') as f:
            raw = json.load(f)

        poses = {}

        for k, v in raw.items():
            poses[int(k)] = np.array(v)

        return poses

    def camera_matrix(self):

        return np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0, 0, 1]
        ])
