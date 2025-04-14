import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("playlists_metadata.csv")

df["track_names"] = df["track_names"].apply(ast.literal_eval)
df["genres"] = df["genres"].apply(ast.literal_eval)

df_with_genres = df[df["genres"].apply(lambda g: len(g) > 0)].copy()
df_no_genres = df[df["genres"].apply(lambda g: len(g) == 0)].copy()

print(f"Playlists with genres: {len(df_with_genres)}")
print(f"Playlists missing genres: {len(df_no_genres)}")


def combine_text(row):
    return row["playlist_title"] + " " + " ".join(row["track_names"])


texts_with = df_with_genres.apply(combine_text, axis=1).tolist()
texts_no = df_no_genres.apply(combine_text, axis=1).tolist()

vectorizer = TfidfVectorizer(stop_words="english")
X_with = vectorizer.fit_transform(texts_with)
X_no = vectorizer.transform(texts_no)

nn = NearestNeighbors(n_neighbors=1, metric="cosine")
nn.fit(X_with)

distances, indices = nn.kneighbors(X_no)

imputed_genres = []
for idx in indices.flatten():
    imputed_genres.append(df_with_genres.iloc[idx]["genres"])


missing_indices = df[df["genres"].apply(lambda g: len(g) == 0)].index
for i, idx in enumerate(missing_indices):
    df.at[idx, "genres"] = imputed_genres[i]

df.to_csv("playlists_metadata_imputed.csv", index=False)
print("✅ Genres imputed and saved to playlists_metadata_imputed.csv")
