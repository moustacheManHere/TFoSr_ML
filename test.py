import requests
import cv2
import numpy as np
import time

# Path to the image file
image_path = "data/train/A/A1126.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert the image to a byte array
_, buffer = cv2.imencode(".jpg", image)
image_bytes = buffer.tobytes()

# Duplicate the image into an array of five images
files = [("files", (f"image_{i}.jpg", image_bytes, "image/jpeg")) for i in range(5)]

# Define the endpoint URL
# url = "http://0.0.0.0:8000/predict"
url = "https://tfosr-ml.onrender.com/predict"

# Number of times to post
num_requests = 10

# Measure response times
response_times = []
for _ in range(num_requests):
    start_time = time.time()
    response = requests.post(url, files=files)
    end_time = time.time()

    elapsed_time = end_time - start_time
    response_times.append(elapsed_time)

    if response.status_code == 200:
        print("Response received:", response.json())
    else:
        print(f"Failed to get a valid response: {response.status_code}", response.text)

# Print response time statistics
print("\nResponse Times (in seconds):")
print(response_times)
print(f"Average Response Time: {np.mean(response_times):.4f} seconds")