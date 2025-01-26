

# Hand Gesture Classification API  

This project provides a FastAPI-based API and a Streamlit app for classifying Singapore Sign Language gestures into letters using a combination of Mediapipe for hand landmark detection and a PyTorch model for keypoint classification.  

## Features  

- **Hand Gesture Detection:** Detects hand landmarks in images using Mediapipe.  
- **Keypoint Classification:** Classifies detected hand gestures into letters (A-Z) using a PyTorch model.  
- **Batch Predictions:** Supports batch image uploads and returns predictions for each image.  
- **Streamlit App:** Offers an interactive interface for using the model, including webcam-based gesture detection.  

---

## Setup and Usage  

### Prerequisites  

- Docker and Docker Compose installed.  
- Python 3.8 or later (optional, for local development).  

### Clone the Repository  

```bash  
git clone https://github.com/moustacheManHere/TFoSr_ML.git  
cd TFoSr_ML  
```  

### Running with Docker  

1. **Build and Start the Application**:  
   ```bash  
   docker compose up -d --build  
   ```  

2. **Access the API**:  
   The API will be available at [http://localhost:8000](http://localhost:8000).  

### Local Development  

1. **Install Dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

2. **Run the Application**:  
   ```bash  
   fastapi run main.py
   ```  

### API Endpoints  

- **`POST /predict`**  
  Upload one or more images for hand gesture classification.  
  - **Request**:  
    - Form-data with files: `files[]` (list of images).  
  - **Response**:  
    ```json  
    {  
      "majority_letter": "A",  
      "all_predictions": ["A", "A", "B", "A"]  
    }  
    ```  

An example of calling this API is shown in the provided `test.py` script.  

### Testing  

You can test the API using the provided `test.py` script:  

1. Edit the `image_path` in `test.py` to point to your image.  
2. Run the script:  
   ```bash  
   python test.py  
   ```  

The script will:  
- Send multiple requests to the `/predict` endpoint.  
- Print the server's responses and speed of prediction.  


---

### Training and Model Files  

The repository also includes scripts and notebooks for training and inference:  

#### Location of Files  
All training-related files are located in the `training` folder:  

```
├── training  
│   ├── extract_keypoints.py  
│   ├── hand_landmarker.ipynb  
│   ├── images_to_csv.py  
│   ├── inference_mediapipe.py  
│   ├── inference_pipeline.py  
│   └── training_notebook.ipynb  
```  

#### Purpose of Each Script/Notebook  

1. **`training_notebook.ipynb`**: A notebook for training the keypoint classifier (data provided in CSV format).  
2. **`extract_keypoints.py`**: Removes images from the dataset that do not have keypoints, since we can't use those to train classifier.  
3. **`images_to_csv.py`**: Runs mediapipe model on the images and saves their keypoints CSV format for training.  
4. **`inference_mediapipe.py`**: A script for running inference using the Mediapipe pretrained model only.  
5. **`inference_pipeline.py`**: Combines Mediapipe and the trained classifier for full pipeline inference.  
6. **`hand_landmarker.ipynb`**: Demonstrates how to use the Mediapipe hand landmark detector.  

#### Model Files  

The models used in this project are located in the `models` folder:  

```
├── models  
│   ├── hand_keypoints_classifier_new.pt  // Runs only on Macbook
│   ├── hand_keypoints_classifier_new_cpu.pt  // Runs on anything
│   ├── hand_landmarker.task  
│   └── mlp_hand_sign_classifier.pt  // Older model
```  

---

### Streamlit App  

In addition to the FastAPI backend, the model is hosted on Streamlit for an live user testing.  

The Streamlit-related scripts are in the `streamlit` folder:  

```
├── streamlit  
│   ├── deploy.py  
│   └── working_app.py  
```  

#### Streamlit Deployment  

The Streamlit app is deployed and accessible at:  
[https://tfosr-ml.streamlit.app/](https://tfosr-ml.streamlit.app/)  

You can run another script called `working_app.py` using the following command to test this locally.

```bash
streamlit run working_app.py
```

#### Difference Between the Streamlit Scripts

1. **Deployed Version (deploy.py)**:  
   - Users can take a photo using their webcam and have it classified by the model. 
   - Due to difficulties setting up streamlit_webrtc, cannot have a live stream. 

2. **Local Version (working_app.py)**:  
   - If users want live inference on their webcam, they can use the `working_app.py` script. 
   - Keypoints will be shown as you show your webcam 

