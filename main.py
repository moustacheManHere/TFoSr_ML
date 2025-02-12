from collections import Counter
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class HandLandmarkClassifier:
    def __init__(self, landmark_model_path, classifier_model_path):
        base_options = python.BaseOptions(model_asset_path=landmark_model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.device = torch.device("cpu")
        self.model = torch.jit.load(classifier_model_path).to(self.device)
        self.model.eval()

    def _normalize_hand_landmarks(self, detection_result):
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

    def classify_gesture(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            raise ValueError("No hand landmarks detected.")

        normalized_coords = self._normalize_hand_landmarks(detection_result)
        keypoints = []
        for x, y in normalized_coords:
            keypoints.extend([x, y])

        input_tensor = torch.tensor(keypoints, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)

        predicted_index = np.argmax(output.cpu().numpy())
        return ASCII_UPPERCASE[predicted_index], normalized_coords

# Initialize the classifier with your model paths
classifier = HandLandmarkClassifier(
    landmark_model_path="models/hand_landmarker.task",
    classifier_model_path="models/hand_keypoints_classifier_cpu_lr.pt"
)

@app.post("/predict")
async def predict_letters(files: List[UploadFile] = File(...)):
    try:
        predicted_letters = []

        for file in files:
            # Read and decode the image
            image_data = await file.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Classify the gesture
            predicted_letter, normalized_coords = classifier.classify_gesture(image_rgb)
            predicted_letters.append(predicted_letter)

        # Determine the majority letter
        if not predicted_letters:
            raise HTTPException(status_code=400, detail="No valid predictions made.")

        majority_letter = Counter(predicted_letters).most_common(1)[0][0]

        # return JSONResponse(content={"majority_letter": majority_letter, "all_predictions": predicted_letters})
        return JSONResponse(content={"majority_letter": majority_letter, "all_predictions": predicted_letters, "normalized_coords": normalized_coords})
    except ValueError as e:
        return JSONResponse(content={"majority_letter": None, "all_predictions": None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
