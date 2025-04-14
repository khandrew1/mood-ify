from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.predict import predict_mood
import os
import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Spotify setup ===
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    )
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Shared helper ===
def get_playlist_metadata(playlist_id: str):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spotify fetch failed: {e}")


# === New route: /analyze ===
@app.get("/analyze")
def analyze_playlist(playlist_id: str = Query(...)):
    metadata = get_playlist_metadata(playlist_id)
    mood = predict_mood(metadata)

    try:
        # === GPT prompt for mood description + image prompt ===
        gpt_prompt = (
            f"This Spotify playlist has the title: '{metadata['playlist_title']}'\n"
            f"The first few tracks are: {', '.join(metadata['track_names'])}.\n"
            f"The associated genres are: {', '.join(metadata['genres'])}.\n"
            f"The predicted mood is: '{mood}'.\n\n"
            "Based on this, write:\n"
            "A short visual prompt for generating cover art (like you'd use with DALL·E)\n\n"
        )

        gpt_response = openai.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": gpt_prompt}],
            temperature=0.7,
        )

        image_prompt = gpt_response.output_text

        image_response = openai.images.generate(
            model="dall-e-2",
            prompt=image_prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )

        image_base64 = image_response.data[0].b64_json

        return {
            "mood": mood,
            "image_prompt": image_prompt,
            "image_base64": image_base64,
            "metadata": metadata,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI pipeline failed: {e}")
