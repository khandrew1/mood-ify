import pandas as pd

df = pd.read_csv("playlists_metadata_imputed.csv")

missing = df["genres"].apply(lambda x: len(eval(x)) == 0).sum()
print(f"{missing} / {len(df)} playlists are missing genre info.")
