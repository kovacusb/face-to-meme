import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path 

model_path = str(Path(__file__).parent / "face_landmarker.task")

# Options for the task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
print("Loading options")
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
print("Loading task")
# Actual Instance of the task
with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file("smile.jpg")
    face_landmarker_result = landmarker.detect(mp_image)
    print("Success?")
  # ...
    