import os
import json
import argparse
import numpy as np
import open3d as o3d

from .dataset import SceneDataset
from .feature_mapping import SparseMapper
from .localization import SemanticLocalizer
from .visualization import Visualizer
from .query_inference import QueryInference

from .settings import *


class ReconstructionPipeline:

    def __init__(self):

        self.dataset = SceneDataset()

    def export_transforms(self):

        frames = []

        for idx, pose in self.dataset.poses.items():

            frames.append({
                'file_path': f'images/frame_{idx:06d}.png',
                'transform_matrix': pose.tolist()
            })

        output = {
            'camera_model': 'OPENCV',
            'frames': frames
        }

        with open(
            os.path.join(
                OUTPUT_FOLDER,
                'transforms.json'
            ),
            'w'
        ) as f:

            json.dump(
                output,
                f,
                indent=2
            )

    def build_scene(self):

        mapper = SparseMapper(
            self.dataset.K
        )

        cloud = mapper.generate_sparse_cloud(
            self.dataset.images,
            self.dataset.poses
        )

        o3d.io.write_point_cloud(
            os.path.join(
                OUTPUT_FOLDER,
                'sparse_scene.ply'
            ),
            cloud
        )

        return cloud

    def estimate_semantics(
        self,
        cloud
    ):

        localizer = SemanticLocalizer(
        cloud,
        self.dataset.images,
        self.dataset.poses,
        self.dataset.K
    )

        results = localizer.localize_entities()

        with open(
            os.path.join(
                OUTPUT_FOLDER,
                'answers.json'
            ),
            'w'
        ) as f:

            json.dump(
                results,
                f,
                indent=2
            )

    def run(self):

        print('Exporting transforms...')

        self.export_transforms()

        print('Building sparse scene...')

        cloud = self.build_scene()

        print('Estimating semantic OBBs...')

        self.estimate_semantics(cloud)

        print('Generating visualization...')

        Visualizer().save_preview(cloud)

        print('Pipeline completed successfully.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--query',
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.query is not None:

        QueryInference().generate_predictions(
            args.query
        )

    else:

        pipeline = ReconstructionPipeline()

        pipeline.run()

