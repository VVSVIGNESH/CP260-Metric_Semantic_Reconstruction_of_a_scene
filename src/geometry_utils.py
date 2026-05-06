import numpy as np
import open3d as o3d

from sklearn.cluster import DBSCAN


class GeometryProcessor:

    def __init__(self):
        pass

    def cluster_surface(self,points):

        import open3d as o3d
        import numpy as np

        if len(points) < 10:
            return points

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            points
        )

        labels = np.array(

            pcd.cluster_dbscan(
                eps=0.02,
                min_points=15
            )
        )

        valid = labels >= 0

        if np.sum(valid) == 0:
            return points

        unique, counts = np.unique(
            labels[valid],
            return_counts=True
        )

        best_cluster = unique[
            np.argmax(counts)
        ]

        filtered = points[
            labels == best_cluster
        ]

        return filtered

    def estimate_obb(self, points):

        # Absolute fallback
        if len(points) == 0:

            return {

                'center': [0.0, 0.0, 0.0],

                'extent': [0.04, 0.015, 0.006],

                'rotation': np.eye(3).tolist()
            }

        # Few points fallback
        if len(points) < 4:

            center = np.mean(points, axis=0)

            return {

                'center': center.tolist(),

                'extent': [0.04, 0.015, 0.006],

                'rotation': np.eye(3).tolist()
            }

        try:

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(
                points
            )

            obb = pcd.get_minimal_oriented_bounding_box()

            extent = obb.extent.tolist()

            # enforce realistic connector depth
            extent[2] = 0.006

            return {

                'center': obb.center.tolist(),

                'extent': extent,

                'rotation': obb.R.tolist()
            }

        except:

            center = np.mean(points, axis=0)

            return {

                'center': center.tolist(),

                'extent': [0.04, 0.015, 0.006],

                'rotation': np.eye(3).tolist()
            }