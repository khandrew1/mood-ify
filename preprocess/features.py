import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
import joblib
import os

# Optional: Set output paths here
VECTORIZER_DIR = "vectorizers"
os.makedirs(VECTORIZER_DIR, exist_ok=True)


def combine_text_fields(row):
    title = row["playlist_title"]
    tracks = " ".join(ast.literal_eval(row["track_names"]))
    return f"{title} {tracks}"


def extract_features(df, save_vectorizers=True):
    # Target variable
    y = df["mood_label"]

    # Text preprocessing
    text_data = df.apply(combine_text_fields, axis=1)

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_text = tfidf.fit_transform(text_data)

    # Genre preprocessing
    genre_lists = df["genres"].apply(ast.literal_eval)
    genre_encoder = MultiLabelBinarizer()
    X_genres = genre_encoder.fit_transform(genre_lists)

    # Combine all features
    X = hstack([X_text, X_genres])

    # Optionally save vectorizers
    if save_vectorizers:
        joblib.dump(tfidf, os.path.join(VECTORIZER_DIR, "tfidf.pkl"))
        joblib.dump(genre_encoder, os.path.join(VECTORIZER_DIR, "genre_encoder.pkl"))

    return X, y
