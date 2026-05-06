import os
import cv2
import json


IMAGE_FOLDER = 'data/images'

ANNOTATIONS = {}

DISPLAY_WIDTH = 1400


def annotate_image(image_path):

    image = cv2.imread(image_path)

    original_h, original_w = image.shape[:2]

    scale = DISPLAY_WIDTH / original_w

    display_h = int(original_h * scale)

    resized = cv2.resize(
        image,
        (DISPLAY_WIDTH, display_h)
    )

    print('\nAnnotating:', image_path)

    while True:

        entity = input(
            '\nEnter entity name '
            '(power_socket / ethernet_socket / '
            'vga_socket / hdmi_socket_left / '
            'usb_socket_top_right / done / skip): '
        )

        if entity == 'done':
            break

        if entity == 'skip':
            continue

        roi = cv2.selectROI(
            f'Annotate {entity}',
            resized,
            showCrosshair=True
        )

        x, y, w, h = roi

        # scale back to original image size
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + w) / scale)
        y2 = int((y + h) / scale)

        frame_name = os.path.basename(
            image_path
        )

        frame_id = int(
            frame_name.split('_')[1].split('.')[0]
        )

        if entity not in ANNOTATIONS:
            ANNOTATIONS[entity] = {}

        ANNOTATIONS[entity][frame_id] = [
            x1, y1, x2, y2
        ]

        print(
            f'Frame {frame_id} | {entity} →',
            [x1, y1, x2, y2]
        )

        cv2.destroyAllWindows()


image_files = sorted([
    file for file in os.listdir(IMAGE_FOLDER)
    if file.endswith('.png')
])


# Choose ONLY the frames you want
TARGET_FRAMES = [
    471,
    365,
    468,
    461,
    531
]

for file in image_files:

    frame_id = int(
        file.split('_')[1].split('.')[0]
    )

    if frame_id not in TARGET_FRAMES:
        continue

    path = os.path.join(
        IMAGE_FOLDER,
        file
    )

    annotate_image(path)


with open(
    'roi_annotations.json',
    'w'
) as f:

    json.dump(
        ANNOTATIONS,
        f,
        indent=2
    )

print('\nSaved ROI annotations.')