import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN


class GeometryProcessor:

    def __init__(self):
        pass

    def cluster_surface(self, points):

        if len(points) < 20:
            return points

        labels = DBSCAN(
            eps=0.03,
            min_samples=5
        ).fit(points).labels_

        valid = labels >= 0

        filtered = points[valid]

        # fallback if DBSCAN removes everything
        if len(filtered) < 10:
            return points

        return filtered

    def estimate_obb(self, points):

        # fallback for extremely sparse points
        if len(points) < 4:

            center = np.mean(points, axis=0)

            return {

                'center': center.tolist(),

                'extent': [0.04, 0.015, 0.006],

                'rotation': np.eye(3).tolist()
            }

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            points
        )

        try:

            obb = pcd.get_minimal_oriented_bounding_box()

        except:

            center = np.mean(points, axis=0)

            return {

                'center': center.tolist(),

                'extent': [0.04, 0.015, 0.006],

                'rotation': np.eye(3).tolist()
            }

        extent = obb.extent.tolist()

        # enforce realistic connector depth
        extent[2] = 0.006

        return {

            'center': obb.center.tolist(),

            'extent': extent,

            'rotation': obb.R.tolist()
        }