import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from predict import predict_mood

load_dotenv()

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    )
)


def get_playlist_metadata(playlist_id: str):
    playlist = sp.playlist(playlist_id)
    title = playlist["name"]
    tracks = playlist["tracks"]["items"]

    track_names = []
    genres = set()
    for item in tracks[:10]:
    track = item.get("track")
        if not track:
            continue
        track_names.append(track["name"])
        for artist in track["artists"]:
            artist_info = sp.artist(artist["id"])
            genres.update(artist_info.get("genres", []))

    return {
        "playlist_title": title,
        "track_names": track_names,
        "genres": list(genres) if genres else ["unknown"],
    }


if __name__ == "__main__":
    playlist_url = input("🔗 Enter a Spotify playlist URL or ID: ").strip()
    playlist_id = playlist_url.split("/")[-1].split("?")[0]  # clean up URL
    playlist_data = get_playlist_metadata(playlist_id)

    print("\n🎧 Playlist Title:", playlist_data["playlist_title"])
    print("🎵 First Tracks:", playlist_data["track_names"])
    print("🎼 Genres:", playlist_data["genres"])

    predicted_mood = predict_mood(playlist_data)
    print("\n🧠 Predicted Mood:", predicted_mood)
