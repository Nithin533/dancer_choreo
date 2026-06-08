from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import json
import numpy as np
import pickle

from process_audio import process_audio
from generate_poses import generate_poses

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend",  StaticFiles(directory="frontend"),  name="frontend")
app.mount("/output",    StaticFiles(directory="output"),    name="output")
app.mount("/separated", StaticFiles(directory="separated"), name="separated")

os.makedirs("audio",    exist_ok=True)
os.makedirs("output",   exist_ok=True)
os.makedirs("frontend", exist_ok=True)


def npy_to_json(poses_path, out_path):
    poses = np.load(poses_path)
    with open(out_path, "w") as f:
        json.dump(poses.tolist(), f)
    return out_path


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):

    # Save uploaded file
    save_path = f"audio/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"\n📥 Received: {file.filename}")

    # Song name derived from uploaded filename
    song_name   = os.path.splitext(file.filename)[0]
    song_folder = f"output/{song_name}"

    # Step 1 — Audio pipeline → returns no_vocals_path string
    no_vocals_path = process_audio(save_path)
    if no_vocals_path is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "step": "audio", "message": "Audio processing failed"}
        )

    # Step 2 — Generate poses — song_folder auto-passed
    smooth_path = generate_poses(
        no_vocals_path  = no_vocals_path,
        checkpoint_path = "best_model.pt",
        song_folder     = song_folder,
    )

    if smooth_path is None or not os.path.exists(smooth_path):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "step": "poses", "message": "Pose generation failed"}
        )

    # Step 3 — Convert poses to JSON for Three.js
    poses_json_path = f"{song_folder}/poses.json"
    npy_to_json(smooth_path, poses_json_path)

    # Step 4 — Load metadata
    meta_path = f"{song_folder}/generation_meta.pkl"
    style = "hiphop"
    tempo = 120.0
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        style = meta.get("style", "hiphop")
        tempo = meta.get("tempo", 120.0)

    return JSONResponse(content={
        "status"      : "success",
        "song_name"   : song_name,
        "poses_url"   : f"/output/{song_name}/poses.json",
        "audio_url"   : f"/separated/{song_name}/htdemucs/normalized_audio/no_vocals.mp3",
        "style"       : style,
        "tempo"       : tempo,
        "total_frames": int(np.load(smooth_path).shape[0]),
    })


@app.get("/")
def root():
    return FileResponse("frontend/index.html")
