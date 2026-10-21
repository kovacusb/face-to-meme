import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
import joblib


def result_to_dict(result: vision.FaceLandmarkerResult) -> dict:
    if not result.face_blendshapes:
        raise RuntimeError("No faces detected in the image")
    
    scores = result.face_blendshapes[0]
    return {c.category_name:c.score for c in scores}

def dict_to_df(scores: dict, model) -> pd.DataFrame:
    expected_cols = list(getattr(model, "feature_names_in_", []))

    if not expected_cols:
        expected_cols = list(scores.keys())
    row = {col:scores.get(col, 0.0) for col in expected_cols}
    return pd.DataFrame([row])

def predict(res: vision.FaceLandmarkerResult, m_model: str):
    dict_blendshapes  = result_to_dict(res)
    m_df = dict_to_df(dict_blendshapes, m_model)

    model = joblib.load(m_model)
    prediction = model.predict(m_df)

    return prediction[0]


