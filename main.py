import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path 
import sys
# Local module
import helpers


model_path = str(Path(__file__).parent / "face_landmarker.task")

if len(sys.argv) != 2:
    raise Exception("Usage: python main.py image_path")
IMAGE_FILE = f"faces/{sys.argv[1]}"
# Options for the task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
print("Loading options...")
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, 
    num_faces=1,
    output_facial_transformation_matrixes=False, 
    output_face_blendshapes=True)

    #ERROR
print("Loading task...")
detector = FaceLandmarker.create_from_options(options)
image = mp.Image.create_from_file(IMAGE_FILE)
print(f"Parsing {sys.argv[1]}...")
detection_result = detector.detect(image)

for b in sorted(detection_result.face_blendshapes[0], key=lambda b: b.score, reverse=True):
    print(f"{b.category_name:<20} {b.score:.3f}")