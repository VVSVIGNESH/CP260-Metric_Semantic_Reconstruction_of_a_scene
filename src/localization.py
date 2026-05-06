import cv2
import numpy as np

from .geometry_utils import GeometryProcessor


class SemanticLocalizer:

    def __init__(
        self,
        point_cloud,
        images,
        poses,
        K
    ):

        self.point_cloud = point_cloud
        self.images = images
        self.poses = poses
        self.K = K

        self.geometry = GeometryProcessor()

        # stronger ORB settings for tiny connectors
        self.detector = cv2.ORB_create(
            nfeatures=8000,
            edgeThreshold=5,
            fastThreshold=5,
            patchSize=31
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        # physical size priors
        self.size_priors = {

            'power_socket': [0.04, 0.08],

            'ethernet_socket': [0.02, 0.05],

            'vga_socket': [0.03, 0.08],

            'hdmi_socket_left': [0.01, 0.03],

            'usb_socket_top_right': [0.01, 0.03]
        }

    def socket_annotations(self):

        return {

            "power_socket": {

                "365": [1464, 1115, 1601, 1296],
                "461": [1812, 815, 1936, 972],
                "468": [1501, 1007, 1707, 1109],
                "471": [1113, 994, 1376, 1113],
                "531": [987, 1058, 1312, 1329]
            },

            "ethernet_socket": {

                "365": [1517, 484, 1601, 597],
                "461": [1861, 464, 1954, 559],
                "468": [1576, 502, 1671, 643],
                "471": [1186, 491, 1281, 650]
            },

            "vga_socket": {

                "365": [1444, 188, 1561, 384],
                "461": [1843, 320, 1941, 422],
                "468": [1536, 312, 1662, 460],
                "471": [1119, 325, 1265, 457]
            },

            "hdmi_socket_left": {

                "365": [1435, 292, 1563, 482],
                "461": [1839, 385, 1920, 473],
                "468": [1521, 416, 1649, 521],
                "471": [1124, 422, 1248, 537]
            },

            "usb_socket_top_right": {

                "365": [1532, 82, 1638, 279],
                "461": [1898, 243, 1963, 382],
                "468": [1601, 210, 1693, 418],
                "471": [1172, 217, 1307, 407]
            }
        }

    def crop_roi(
        self,
        image,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        roi = image[y1:y2, x1:x2]

        return roi, x1, y1

    def extract_roi_features(
        self,
        image,
        bbox
    ):

        roi, offset_x, offset_y = self.crop_roi(
            image,
            bbox
        )

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # upscale tiny ROIs
        gray = cv2.resize(
            gray,
            None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_CUBIC
        )

        keypoints, descriptors = self.detector.detectAndCompute(
            gray,
            None
        )

        if keypoints is None or descriptors is None:
            return [], None

        # convert back to original image coordinates
        for kp in keypoints:

            kp.pt = (
                kp.pt[0] / 3.0 + offset_x,
                kp.pt[1] / 3.0 + offset_y
            )

        return keypoints, descriptors

    def triangulate_matches(
        self,
        kp1,
        kp2,
        matches,
        pose1,
        pose2
    ):

        pts1 = np.float32([
            kp1[m.queryIdx].pt
            for m in matches
        ])

        pts2 = np.float32([
            kp2[m.trainIdx].pt
            for m in matches
        ])

        P1 = self.K @ np.linalg.inv(
            pose1
        )[:3]

        P2 = self.K @ np.linalg.inv(
            pose2
        )[:3]

        points_4d = cv2.triangulatePoints(
            P1,
            P2,
            pts1.T,
            pts2.T
        )

        points_3d = (
            points_4d[:3] / points_4d[3]
        ).T

        valid = np.isfinite(
            points_3d
        ).all(axis=1)

        return points_3d[valid]

    def process_entity(
        self,
        entity_name,
        annotations
    ):

        frame_ids = sorted([

            int(fid)

            for fid in annotations.keys()

            if int(fid) in self.images
        ])

        all_points = []

        # multi-pair triangulation
        for i in range(len(frame_ids)):

            for j in range(i + 1, len(frame_ids)):

                id1 = frame_ids[i]
                id2 = frame_ids[j]

                image1 = self.images[id1]
                image2 = self.images[id2]

                bbox1 = annotations[str(id1)]
                bbox2 = annotations[str(id2)]

                kp1, desc1 = self.extract_roi_features(
                    image1,
                    bbox1
                )

                kp2, desc2 = self.extract_roi_features(
                    image2,
                    bbox2
                )

                print(
                    f'{entity_name} keypoints:',
                    len(kp1),
                    len(kp2)
                )

                if desc1 is None or desc2 is None:
                    continue

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

                    # relaxed ratio test
                    if m.distance < 0.92 * n.distance:
                        filtered.append(m)

                print(
                    f'{entity_name} | '
                    f'{id1}-{id2} | '
                    f'Matches:',
                    len(filtered)
                )

                if len(filtered) < 4:
                    continue

                triangulated = self.triangulate_matches(
                    kp1,
                    kp2,
                    filtered,
                    self.poses[id1],
                    self.poses[id2]
                )

                print(
                    f'{entity_name} triangulated:',
                    len(triangulated)
                )

                if len(triangulated) > 0:

                    all_points.append(
                        triangulated
                    )

        if len(all_points) == 0:

            print(
                f'WARNING: No valid geometry for {entity_name}'
            )

            return {

                'entity': entity_name,

                'obb': {

                    'center': [0, 0, 0],

                    'extent': [0.04, 0.015, 0.006],

                    'rotation': np.eye(3).tolist()
                }
            }

        all_points = np.concatenate(
            all_points,
            axis=0
        )

        clustered = self.geometry.cluster_surface(
            all_points
        )

        obb = self.geometry.estimate_obb(
            clustered
        )

        # apply physical size constraints
        min_size, max_size = self.size_priors[
            entity_name
        ]

        extent = np.array(
            obb['extent']
        )

        extent[0] = np.clip(
            extent[0],
            min_size,
            max_size
        )

        extent[1] = np.clip(
            extent[1],
            min_size,
            max_size
        )

        # fixed thickness
        extent[2] = 0.006

        obb['extent'] = extent.tolist()

        return {

            'entity': entity_name,

            'obb': obb
        }

    def localize_entities(self):

        annotations = self.socket_annotations()

        results = []

        for entity_name, entity_annotations in annotations.items():

            print(
                f'\nProcessing: {entity_name}'
            )

            result = self.process_entity(
                entity_name,
                entity_annotations
            )

            results.append(result)

        return results