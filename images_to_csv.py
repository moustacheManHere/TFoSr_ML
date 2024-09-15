from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tqdm import tqdm
import cv2
import os

directory = "data/val"

sub_dirs = os.listdir(directory)

directorys = [
    os.path.join(directory, sub_dir)
    for sub_dir in sub_dirs
    if os.path.isdir(os.path.join(directory, sub_dir))
]

csv_output = "data/val.csv"

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

dataset = []

for directory in directorys:
    class_name = os.path.basename(directory)

    images = os.listdir(directory)
    images = [os.path.join(directory, image) for image in images]

    for image in tqdm(images, desc=f"Processing {directory}"):
        image_path = image
        image = cv2.imread(image)
        if image is None:
            print(f"Error: {image_path}")
            continue
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

        if detection_result.hand_landmarks == []:
            print(f"Image {image_path} has no hand landmarks, skipping")
            continue

        keypoints = {}

        for i in range(21):
            keypoints[f"kp_{i}_x"] = detection_result.hand_landmarks[0][i].x
            keypoints[f"kp_{i}_y"] = detection_result.hand_landmarks[0][i].y

        keypoints["class"] = class_name

        dataset.append(keypoints)

import pandas as pd

df = pd.DataFrame(dataset)

df.to_csv(csv_output, index=False)
print(f"Saved dataset to {csv_output}")
