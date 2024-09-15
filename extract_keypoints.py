from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tqdm import tqdm
import cv2
import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

directory = "data/val"

sub_dirs = os.listdir(directory)

directorys = [
    os.path.join(directory, sub_dir)
    for sub_dir in sub_dirs
    if os.path.isdir(os.path.join(directory, sub_dir))
]


# Utility function to draw a rectangle around a region of interest
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

for directory in directorys:
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
            print(f"Image {image_path} has no hand landmarks")
            os.remove(image_path)
            continue
    # STEP 5: Process the classification result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(
    #     image.numpy_view()[:, :, :3], detection_result
    # )
    # # replace the oiriginal path and add _keypoint to filename
    # if image_path.endswith(".jpg"):
    #     new_path = image_path.replace(".jpg", "_keypoint.jpg")
    #     cv2.imwrite(new_path, annotated_image)
    # elif image_path.endswith(".png"):
    #     new_path = image_path.replace(".png", "_keypoint.png")
    #     cv2.imwrite(new_path, annotated_image)
    # elif image_path.endswith(".jpeg"):
    #     new_path = image_path.replace(".jpeg", "_keypoint.jpeg")
    #     cv2.imwrite(new_path, annotated_image)
    # else:
    #     print(f"Error: {image_path}")
    #     break
