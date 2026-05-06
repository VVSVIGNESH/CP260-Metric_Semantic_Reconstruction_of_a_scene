import os

ROOT_DIR = os.getcwd()

IMAGE_FOLDER = os.path.join(
    ROOT_DIR,
    'data/images'
)

POSE_FILE = os.path.join(
    ROOT_DIR,
    'data/images/poses.json'
)

OUTPUT_FOLDER = os.path.join(
    ROOT_DIR,
    'outputs'
)

VIS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    'visualizations'
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)

FX = 1477.010
FY = 1480.442
CX = 1298.250
CY = 686.820

DEPTH_PRIOR = 0.006