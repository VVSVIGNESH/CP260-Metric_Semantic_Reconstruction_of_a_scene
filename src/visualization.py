import os
import open3d as o3d

from .settings import *


class Visualizer:

    def save_preview(
        self,
        point_cloud
    ):

        vis = o3d.visualization.Visualizer()

        vis.create_window(
            visible=False
        )

        vis.add_geometry(point_cloud)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(
            os.path.join(
                VIS_FOLDER,
                'scene_preview.png'
            )
        )

        vis.destroy_window()
