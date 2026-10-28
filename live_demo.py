import threading
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

WINDOW = "Expression Matcher"

# ---------- 1. LOAD THE MODEL FIRST ----------
# The callback uses it, so it must exist before the detector starts.
model = joblib.load("expression_model.joblib")

# ---------- 2. SHARED STATE BETWEEN THE CALLBACK AND THE MAIN LOOP ----------
# The callback runs on ANOTHER thread, so we protect shared data with a lock.
lock = threading.Lock()
recent = deque(maxlen=7)       # last predictions, used for the majority vote

# ---------- 3. THE CALLBACK ----------
# Correct signature: (result, output_image, timestamp_ms).
# It only PREDICTS and stores the result. No cv2.imshow here.
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

# ---------- 4. CREATE THE DETECTOR IN LIVE_STREAM MODE ----------
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

# ---------- 5. OPEN THE WEBCAM ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No webcam detected")
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam height: {H}px")

# ---------- 6. LOAD THE REFERENCE IMAGES (once) ----------
def resize_to_height(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), height))

# model label -> filename in the memes/ folder.
# If your files are named exactly like the labels, you can delete this mapping.
LABEL_TO_FILE = {
    "happy": "happy",
    "angry": "angry",
    "sad": "sad",
    "surprised": "surprised",
    "neutral": "neutral",
    "afraid": "afraid",
    "disgusted": "disgusted",
}

reference_dir = Path("memes")
reference_images = {}
for label in model.classes_:
    path = reference_dir / f"{LABEL_TO_FILE.get(label, label)}.jpeg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Missing reference image {path}")
    reference_images[label] = resize_to_height(img, H)

placeholder = np.full_like(next(iter(reference_images.values())), 60)
cv2.putText(placeholder, "No Face", (20, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

# ---------- 7. MAIN LOOP ----------
try:
    while True:
        ok, frame = cap.read()                        # BGR
        if not ok:
            break

        frame = cv2.flip(frame, 1)                    # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB copy only for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Async call: returns immediately. The result arrives later in on_result.
        # The timestamp must ALWAYS increase, so we use a monotonic clock in ms.
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