# Metric-Semantic Reconstruction of Desktop Connector Ports

## Overview

This repository implements a geometry-driven semantic reconstruction pipeline for estimating **3D Oriented Bounding Boxes (OBBs)** of desktop connector ports from posed RGB images.

The system reconstructs semantic entities such as:

- `power_socket`
- `ethernet_socket`
- `vga_socket`
- `hdmi_socket_left`
- `usb_socket_top_right`

using:
- multi-view feature matching,
- triangulation,
- geometric clustering,
- and constrained OBB estimation.

Unlike learning-heavy approaches, the pipeline relies mainly on:
- classical multi-view geometry,
- ROI-guided feature extraction,
- and semantic geometric reasoning.

---

# Project Objective

Given:
- posed RGB images,
- calibrated camera intrinsics,
- and camera poses,

the pipeline estimates:
- semantic 3D object locations,
- oriented bounding boxes,
- and metric scene geometry.

The final outputs include:

- `answers.json`
- `transforms.json`

compatible with:
- evaluation scripts,
- Gaussian Splatting,
- and NeRF-style rendering pipelines.

---

# Pipeline Summary

The reconstruction pipeline follows the workflow below:

```text
Input Images + Camera Poses
            ↓
Semantic ROI Annotation
            ↓
ORB Feature Extraction
            ↓
Cross-View Feature Matching
            ↓
Multi-Pair Triangulation
            ↓
DBSCAN Outlier Filtering
            ↓
Physically Constrained OBB Fitting
            ↓
answers.json Export
```

---

# Main Features

## Geometry-Based Reconstruction

The system avoids dependency on heavy object detection models and instead uses:
- feature geometry,
- triangulation,
- and clustering.

---

## ROI-Guided Matching

Features are extracted only inside semantic connector regions to improve:
- reconstruction stability,
- feature density,
- and connector localization.

---

## Multi-Pair Triangulation

Instead of triangulating only consecutive frames, the system uses:
- multiple valid frame combinations

for denser and more stable geometry.

---

## Physical OBB Constraints

Connector-specific size priors are used to prevent:
- oversized boxes,
- panel leakage,
- unstable extents.

---

## Gaussian Splatting Support

The pipeline exports:

```text
transforms.json
```

which can be directly used with:
- Gaussian Splatting,
- NeRFStudio,
- and neural rendering pipelines.

---

# Repository Layout

```text
project/
│
├── data/
│   ├── images/
│   └── poses.json
│
├── outputs/
│   ├── answers.json
│   └── transforms.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── reconstruction.py
│   ├── localization.py
│   ├── geometry_utils.py
│   ├── visualization.py
│   ├── export.py
│   ├── query_localization.py
│   └── main.py
│
├── report/
│   └── report.pdf
│
├── requirements.txt
└── README.md
```

---

# Dataset Setup

The dataset is intentionally not included in this repository.

Create the following structure manually:

```text
data/
├── images/
│   ├── frame_000365.png
│   ├── frame_000461.png
│   ├── ...
│
└── poses.json
```

The pipeline automatically reads:
- all images inside `data/images/`
- camera poses from `poses.json`

---

# Environment Setup

## Create Conda Environment

```bash
conda create -n cp260 python=3.10
conda activate cp260
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dependencies

Core libraries used:

- open3d==0.19.0
- numpy
- scipy
- opencv-python
- matplotlib
- scikit-learn
- tqdm

---

# Running the Reconstruction Pipeline

Run the complete reconstruction system:

```bash
python -m src.main
```

This generates:

```text
outputs/
├── answers.json
└── transforms.json
```

---

# Query-Time Localization

To localize semantic entities for a new image:

## Step 1

Place the query image inside the project root:

```text
query.jpg
```

---

## Step 2

Run:

```bash
python -m src.query_localization
```

The system generates:

```text
query_answers.json
```

using nearest-view semantic localization.

---

# Semantic Entities

The current implementation reconstructs:

| Entity |
|---|
| power_socket |
| ethernet_socket |
| vga_socket |
| hdmi_socket_left |
| usb_socket_top_right |

Additional entities can be added by:
- defining new ROIs,
- updating semantic annotations,
- rerunning reconstruction.

---

# Output Format

Example structure of `answers.json`:

```json
{
  "entity": "ethernet_socket",
  "obb": {
    "center": [cx, cy, cz],
    "extent": [ex, ey, ez],
    "rotation": [
      [r00, r01, r02],
      [r10, r11, r12],
      [r20, r21, r22]
    ]
  }
}
```

---

# Experimental Highlights

The proposed pipeline demonstrated:
- stable semantic reconstruction,
- realistic connector geometry,
- robust multi-view triangulation,
- and physically plausible OBB estimation.

Key improvements were achieved through:
- ROI upscaling,
- aggressive ORB tuning,
- DBSCAN filtering,
- and multi-pair triangulation.

---

# Current Limitations

Some limitations of the current implementation:

- manual ROI annotations,
- sensitivity to extreme viewpoints,
- sparse texture on metallic connectors,
- thin planar geometry instability.

---

# Future Improvements

Potential future extensions include:

- automatic semantic segmentation,
- learned feature matching,
- panel-plane alignment,
- dense reconstruction,
- and full SLAM-based localization.

---

# Author

### Veera Subrahmanya Vignesh Vemula

CP260 — Robotic Perception  
IIIT Trichy

---

# References

1. Hartley & Zisserman — Multiple View Geometry
2. OpenCV Documentation
3. Open3D Library
4. GroundingDINO
5. Segment Anything (SAM)
6. 3D Gaussian Splatting
