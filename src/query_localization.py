import cv2
import json
import numpy as np

from .dataset import SceneDataset


class QueryLocalizer:

    def __init__(self):

        self.dataset = SceneDataset()

        self.images = self.dataset.images

        self.detector = cv2.ORB_create(
            nfeatures=5000
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        with open(
            'outputs/answers.json',
            'r'
        ) as f:

            self.answers = json.load(f)

    def compute_features(
        self,
        image
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        kp, desc = self.detector.detectAndCompute(
            gray,
            None
        )

        return kp, desc

    def find_best_match(
        self,
        query_image
    ):

        qkp, qdesc = self.compute_features(
            query_image
        )

        best_score = -1
        best_frame = None

        for frame_id, image in self.images.items():

            kp, desc = self.compute_features(
                image
            )

            if desc is None or qdesc is None:
                continue

            matches = self.matcher.knnMatch(
                qdesc,
                desc,
                k=2
            )

            good = []

            for pair in matches:

                if len(pair) != 2:
                    continue

                m, n = pair

                if m.distance < 0.85 * n.distance:
                    good.append(m)

            score = len(good)

            print(
                f'Frame {frame_id}:',
                score
            )

            if score > best_score:

                best_score = score
                best_frame = frame_id

        return best_frame

    def localize(
        self,
        image_path
    ):

        query = cv2.imread(
            image_path
        )

        best_frame = self.find_best_match(
            query
        )

        print(
            '\nBest matching frame:',
            best_frame
        )

        return self.answers


if __name__ == '__main__':

    localizer = QueryLocalizer()

    results = localizer.localize(
        'query.jpg'
    )

    with open(
        'query_answers.json',
        'w'
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        '\nSaved query_answers.json'
    )