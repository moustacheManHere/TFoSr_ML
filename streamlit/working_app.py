import cv2
import streamlit as st
import numpy as np
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants for visualization
MARGIN = 10
FONT_SIZE = 5
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_COLOR = (255, 0, 0)  # Blue color
LETTER_POSITION = (100, 200)
BOX_COLOR = (255, 255, 255)  # White color

ML_THRESHOLDS = {
    "A": 10, "B": 15, "C": 20, "D": 0, "E": 5, "F": 10, "G": 15, "H": 10,
    "I": 0, "K": 10, "L": 5, "M": 5, "N": 5, "O": 0, "P": 10, "Q": 15, 
    "R": 0, "S": 0, "T": 5, "U": 0, "V": 5, "W": 10, "X": 10, "Y": 5, "Z": 0
}

class HandLandmarkClassifier:
    def __init__(self, landmark_model_path, classifier_model_path):
        """
        Initialize hand landmark detector and gesture classifier.
        
        Args:
            landmark_model_path (str): Path to MediaPipe hand landmark model
            classifier_model_path (str): Path to PyTorch hand gesture classification model
        """
        # Initialize hand landmark detector
        base_options = python.BaseOptions(model_asset_path=landmark_model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Initialize gesture classifier
        self.device = torch.device("cpu")
        self.model = torch.jit.load(classifier_model_path).to(self.device)
        self.model.eval()

    def _normalize_hand_landmarks(self, detection_result):
        """
        Normalize hand landmarks to a unit coordinate system.
        
        Args:
            detection_result: MediaPipe hand detection result
        
        Returns:
            list: Normalized coordinates of hand landmarks
        """
        landmarks = detection_result.hand_landmarks[0]
        min_x = min(landmark.x for landmark in landmarks)
        max_x = max(landmark.x for landmark in landmarks)
        min_y = min(landmark.y for landmark in landmarks)
        max_y = max(landmark.y for landmark in landmarks)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return [
            (
                (landmark.x - min_x) / width,
                (landmark.y - min_y) / height
            )
            for landmark in landmarks
        ]

    def draw_landmarks(self, rgb_image, detection_result):
        """
        Draw hand landmarks and handedness on the image.
        
        Args:
            rgb_image (np.ndarray): Input RGB image
            detection_result: MediaPipe hand detection result
        
        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = np.copy(rgb_image)
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        
        for idx, (hand_landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
            # Convert landmarks to proto format for drawing
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                 for landmark in hand_landmarks]
            )
            
            # Draw landmarks and connections
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )
        
        return annotated_image

    def classify_gesture(self, frame):
        """
        Detect hand landmarks and classify hand gesture.
        
        Args:
            frame (np.ndarray): Input video frame
        
        Returns:
            np.ndarray: Annotated image with gesture classification
        """
        # Convert frame to MediaPipe image
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hand landmarks
        detection_result = self.detector.detect(image)
        
        # Draw landmarks
        annotated_image = self.draw_landmarks(
            image.numpy_view()[:, :, :3], 
            detection_result
        )
        
        # If no hand detected, return annotated image
        if not detection_result.hand_landmarks:
            return annotated_image
        
        keypoints = {}

        normalised_coords = self._normalize_hand_landmarks(detection_result)

        for i, (x, y) in enumerate(normalised_coords):
            keypoints[f"kp_{i}_x"] = x
            keypoints[f"kp_{i}_y"] = y

        keypoints = (keypoints.items())
        key_values = [value for key, value in keypoints]

        input = np.array(key_values, dtype=np.float32)
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        print(input)
        with torch.no_grad():
            output = self.model(input)
        
        ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        predicted_class = ascii_uppercase[np.argmax(output.cpu().numpy())]
        
        if output[np.argmax(output.cpu().numpy())] < ML_THRESHOLDS[predicted_class]:
            predicted_class = "-"
        
        # Add white box behind the letter
        box_thickness = -1  # Filled rectangle
        box_size = (200, 200)
        box_start = (LETTER_POSITION[0] - 50, LETTER_POSITION[1] - 150)
        cv2.rectangle(
            annotated_image, 
            box_start, 
            (box_start[0] + box_size[0], box_start[1] + box_size[1]), 
            BOX_COLOR, 
            box_thickness
        )
        
        # Add predicted class to image with blue color and larger size
        cv2.putText(
            annotated_image,
            predicted_class,
            LETTER_POSITION,
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            LETTER_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        
        return annotated_image

def main():
    """
    Main Streamlit application entry point.
    """
    st.title("Hand Gesture Recognition")
    
    # Initialize classifier
    classifier = HandLandmarkClassifier(
        landmark_model_path="models/hand_landmarker.task",
        classifier_model_path="models/hand_keypoints_classifier_cpu_lr.pt"
    )
    
    # Webcam streaming
    run = st.checkbox('Start Camera')
    frame_window = st.image([])
    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        
        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = classifier.classify_gesture(frame)
        frame_window.image(processed_frame)
    
    # Stop camera when not running
    camera.release()

if __name__ == "__main__":
    main()