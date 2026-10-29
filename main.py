import threading
import time
from collections import deque, Counter
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np

WINDOW = "Expression Matcher"

# This model expects a 2d array of 52 blendshape features provided by mediapipe.result.faceBlendshapes[0]
model = joblib.load("expression_model.joblib")


# Concurrency and softening flickering between images.
lock = threading.Lock()
recent = deque(maxlen=7)      


# Receives the detection result from mediapipe task. Its asyncroious since we initialize the faceLandmaker in LIVE_STREA mode. DO NOT EDIT PLEASE :(
def on_result(result, output_image, timestamp_ms):
    try:
        with lock:
            if result.face_blendshapes:
                scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
                recent.append(model.predict([scores])[0])
            else:
                recent.clear()          # no face: forget old predictions
    except Exception as e:
        print(f"Error processing result at {timestamp_ms} ms: {e}")


#Configurate and initialize task. 
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    output_face_blendshapes=True,
    result_callback=on_result,
)
detector = FaceLandmarker.create_from_options(options)
print("Detector ready")


#Capture input video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No webcam detected")
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam height: {H}px")

# Load and resize reference images once
def resize_to_height(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), height))


reference_dir = Path("memes")
reference_images = {}
for label in model.classes_:
    path = reference_dir / f"{label}.jpeg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Missing reference image {path}")
    reference_images[label] = resize_to_height(img, H)

placeholder = np.full_like(next(iter(reference_images.values())), 60)
cv2.putText(placeholder, "No Face", (20, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

try:
    while True:
        ok, frame = cap.read()                        # BGR
        if not ok:
            break

        frame = cv2.flip(frame, 1)                    # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB copy only for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Async call: returns immediately. The result arrives later in on_result.
        timestamp_ms = time.monotonic_ns() // 1_000_000
        detector.detect_async(mp_image, timestamp_ms)

        # Read the shared predictions safely (copy them while holding the lock)
        with lock:
            votes = list(recent)

        if votes:
            emotion = Counter(votes).most_common(1)[0][0]   # majority vote
            ref = reference_images[emotion]
        else:
            emotion = "no face"
            ref = placeholder

        cv2.putText(frame, emotion, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow(WINDOW, np.hstack((frame, ref)))   # same height H, so they stack

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()