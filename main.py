import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path 
import helpers
model_path = str(Path(__file__).parent / "face_landmarker.task")

IMAGE_FILE = 'smile.jpg'
# Options for the task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
print("Loading options")
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, 
    num_faces=1,
    output_facial_transformation_matrixes=True, 
    output_face_blendshapes=True)

    #ERROR
print("Loading task")
detector = FaceLandmarker.create_from_options(options)
image = mp.Image.create_from_file(IMAGE_FILE)
detection_result = detector.detect(image)

annotated_image = helpers.draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imwrite("annotated.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))