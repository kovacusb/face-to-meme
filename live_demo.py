from predictor import predict
import cv2
import pandas as pd
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 


# Callback function that receives async results from the FaceLandmarker

def on_result(result, timqestamp_ms, image):
    """Handle detection results.
    Args:
        result: FaceLandmarkerResult containing face landmarks and blendshapes.
        timestamp_ms: Timestamp (in ms) associated with the input frame.
    """
    try:
        # Predict emotion using the saved model
        prediction = predict(result, "model.joblib")
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error processing result at {timestamp_ms}: {e}")
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
    output_face_blendshapes=True,
    result_callback=on_result
)
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
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detector.detect_async(image, timestamp_ms)
 
# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
# Print for each frame