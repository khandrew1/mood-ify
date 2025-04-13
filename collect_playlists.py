import os
import json
import spotipy
import pandas as pd
from tqdm import tqdm
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load .env credentials
load_dotenv()

# Auth for Spotify
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    )
)

# Mood keywords to search for
MOOD_KEYWORDS = {
    "chill": ["chill", "relax", "calm"],
    "sad": ["sad", "cry", "heartbreak"],
    "hype": ["hype", "gym", "energy"],
    "romantic": ["romantic", "love", "date"],
    "study": ["study", "focus", "ambient"],
}

MAX_PLAYLISTS = 50  # per mood

# Load or init genre cache
GENRE_CACHE_PATH = "artist_genre_cache.json"


def load_cache(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


artist_genre_cache = load_cache(GENRE_CACHE_PATH)


def get_artist_genres(artist_id):
    if artist_id in artist_genre_cache:
        return artist_genre_cache[artist_id]
    try:
        artist = sp.artist(artist_id)
        genres = artist.get("genres", [])
        artist_genre_cache[artist_id] = genres
        return genres
    except:
        return []


# Main data collection
all_data = []

for mood, keywords in MOOD_KEYWORDS.items():
    for keyword in keywords:
        results = sp.search(q=keyword, type="playlist", limit=MAX_PLAYLISTS)
        playlists = results.get("playlists", {}).get("items", [])

        for playlist in tqdm(playlists, desc=f"Fetching '{keyword}'"):
            try:
                playlist_id = playlist.get("id")
                if not playlist_id:
                    continue

                playlist_data = sp.playlist_tracks(playlist_id)
                if not playlist_data or "items" not in playlist_data:
                    continue

                tracks = playlist_data["items"]
                track_names = []
                artist_names = []
                all_genres = []

                for item in tracks[:10]:  # limit to first 10 tracks
                    track = item.get("track")
                    if not track:
                        continue

                    track_names.append(track.get("name", ""))
                    for artist in track.get("artists", []):
                        artist_name = artist.get("name", "")
                        artist_id = artist.get("id")
                        artist_names.append(artist_name)
                        if artist_id:
                            all_genres.extend(get_artist_genres(artist_id))

                all_data.append(
                    {
                        "playlist_title": playlist.get("name", ""),
                        "track_names": track_names,
                        "artist_names": list(set(artist_names)),
                        "genres": list(set(all_genres)),
                        "mood_label": mood,
                    }
                )

            except Exception as e:
                print(f"Error fetching playlist: {e}")

# Save data and cache
df = pd.DataFrame(all_data)
df.to_csv("playlists_metadata.csv", index=False)
print("✅ Saved to playlists_metadata.csv")

save_cache(artist_genre_cache, GENRE_CACHE_PATH)
print("✅ Genre cache updated")

