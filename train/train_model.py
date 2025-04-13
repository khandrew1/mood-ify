import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from preprocess.features import extract_features

# === Load Data ===
DATA_PATH = "data/playlists_metadata_imputed.csv"
df = pd.read_csv(DATA_PATH)

# === Extract Features and Labels ===
X, y = extract_features(df)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train Model ===
# model = LogisticRegression(max_iter=1000, C=1.0)
model = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# === Evaluate on Test Set ===
y_pred = model.predict(X_test)

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# === Save Trained Model ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mood_classifier.pkl")
print("✅ Model saved to models/mood_classifier.pkl")

