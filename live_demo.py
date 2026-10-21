from predictor import predict
import cv2
import pandas as pd
import joblib
import mediapipe as mp
from mediapipe.tasks.python import vision 


# Initialize task
model_path = "face_landmarker.task"

# Options for the task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
print("Loading options...")
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM, 
    num_faces=1,
    output_facial_transformation_matrixes=False, 
    output_face_blendshapes=True)

print("Loading task...")
detector = FaceLandmarker.create_from_options(options)
print("task loaded succesfully")
# test task
# Load model
model = joblib.load("model.joblib")
# test model
# Test webcam

cap = cv2.VideoCapture(1)
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not detect WebCam")
else:
    print("WebCam Detected")
 
# Get video properties (e.g., frame count and frame width)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second (FPS)
print(f"FPS: {fps}")
 
# Read and display each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error occurred.")
        break
 
    # Convert BGR frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(image)

    prediction = predict(result, "model.joblib")
    print(prediction)
 
# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
# Print for each frame