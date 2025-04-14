import joblib
from scipy.sparse import hstack
from sklearn.base import BaseEstimator
from typing import List, Dict

MODEL_PATH = "models/mood_classifier.pkl"
TFIDF_PATH = "vectorizers/tfidf.pkl"
GENRE_ENCODER_PATH = "vectorizers/genre_encoder.pkl"

model: BaseEstimator = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
genre_encoder = joblib.load(GENRE_ENCODER_PATH)


def predict_mood(playlist: Dict[str, List[str]]) -> str:
    """
    Predict the mood of a playlist.

    Args:
        playlist: Dict with keys:
            - 'playlist_title': str
            - 'track_names': List[str]
            - 'genres': List[str]

    Returns:
        Predicted mood label as a string.
    """
    title = playlist.get("playlist_title", "")
    tracks = playlist.get("track_names", [])
    genres = playlist.get("genres", [])

    combined_text = f"{title} {' '.join(tracks)}"
    X_text = tfidf.transform([combined_text])

    X_genres = genre_encoder.transform([genres])
    X = hstack([X_text, X_genres])

    y_pred = model.predict(X)
    return y_pred[0]
