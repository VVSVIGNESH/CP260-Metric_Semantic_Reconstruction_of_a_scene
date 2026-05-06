import cv2
import numpy as np
import open3d as o3d

from itertools import combinations


class SparseMapper:

    def __init__(self, K):

        self.K = K

        self.detector = cv2.AKAZE_create()

        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    def generate_sparse_cloud(
        self,
        images,
        poses
    ):

        all_points = []

        frame_ids = sorted(images.keys())

        for id1, id2 in combinations(frame_ids, 2):

            img1 = cv2.cvtColor(
                images[id1],
                cv2.COLOR_BGR2GRAY
            )

            img2 = cv2.cvtColor(
                images[id2],
                cv2.COLOR_BGR2GRAY
            )

            kp1, desc1 = self.detector.detectAndCompute(
                img1,
                None
            )

            kp2, desc2 = self.detector.detectAndCompute(
                img2,
                None
            )

            if desc1 is None or desc2 is None:
                continue

            desc1 = np.float32(desc1)
            desc2 = np.float32(desc2)

            matches = self.matcher.knnMatch(
                desc1,
                desc2,
                k=2
            )

            filtered = []

            for pair in matches:

                if len(pair) != 2:
                    continue

                m, n = pair

                if m.distance < 0.72 * n.distance:
                    filtered.append(m)

            if len(filtered) < 20:
                continue

            pts1 = np.float32([
                kp1[m.queryIdx].pt
                for m in filtered
            ])

            pts2 = np.float32([
                kp2[m.trainIdx].pt
                for m in filtered
            ])

            P1 = self.K @ np.linalg.inv(
                poses[id1]
            )[:3]

            P2 = self.K @ np.linalg.inv(
                poses[id2]
            )[:3]

            X = cv2.triangulatePoints(
                P1,
                P2,
                pts1.T,
                pts2.T
            )

            X = (X[:3] / X[3]).T

            finite_mask = np.isfinite(X).all(axis=1)

            X = X[finite_mask]

            if len(X) > 0:
                all_points.append(X)

        all_points = np.concatenate(all_points)

        cloud = o3d.geometry.PointCloud()

        cloud.points = o3d.utility.Vector3dVector(
            all_points
        )

        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=1.5
        )

        return cloud
