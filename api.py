import os
import sys
import pickle
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil

from process_audio import process_audio
from generate_poses import generate_poses

app = FastAPI(title="Dance Choreography AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("output",    exist_ok=True)
os.makedirs("separated", exist_ok=True)
os.makedirs("audio",     exist_ok=True)
os.makedirs("frontend",  exist_ok=True)

app.mount("/output",    StaticFiles(directory="output"),    name="output")
app.mount("/separated", StaticFiles(directory="separated"), name="separated")
app.mount("/frontend",  StaticFiles(directory="frontend"),  name="frontend")


def npy_to_json(npy_path, json_path):
    """Convert (T, 72) npy → list of 511 lists of 72 floats"""
    data = np.load(npy_path)          # shape (T, 72)
    # Ensure it's 2D
    if data.ndim == 1:
        data = data.reshape(-1, 72)
    with open(json_path, "w") as f:
        json.dump(data.tolist(), f)
    return True


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):

    print(f"\n📥 Received: {file.filename}")

    # Save uploaded file
    save_path = f"audio/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    song_name   = os.path.splitext(file.filename)[0]
    song_folder = f"output/{song_name}"

    # ── Step 1-6: Audio pipeline ──
    print("🚀 Running audio pipeline...")
    no_vocals_path = process_audio(save_path)

    if not no_vocals_path or not os.path.exists(no_vocals_path):
        raise HTTPException(status_code=500, detail="Audio processing failed")

    # ── Step 7: Generate poses ──
    print("💃 Generating poses...")
    smooth_path = generate_poses(
        no_vocals_path  = no_vocals_path,
        checkpoint_path = "best_model.pt",
        song_folder     = song_folder
    )

    if not smooth_path or not os.path.exists(smooth_path):
        raise HTTPException(status_code=500, detail="Pose generation failed")

    # ── Convert poses npy → JSON (2D array guaranteed) ──
    json_pose_path = f"{song_folder}/poses.json"
    npy_to_json(smooth_path, json_pose_path)

    # Verify JSON is correct shape
    poses_check = np.load(smooth_path)
    print(f"✅ Poses shape: {poses_check.shape}")  # should be (T, 72)

    # ── Load metadata ──
    meta_path = f"{song_folder}/generation_meta.pkl"
    style = "freestyle"
    tempo = 120.0
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        style = meta.get("style", "freestyle")
        tempo = meta.get("tempo", 120.0)

    # ── Load beat timestamps ──
    beat_times = []
    beat_path  = f"{song_folder}/beat_times.npy"
    if os.path.exists(beat_path):
        beat_times = np.load(beat_path).tolist()
        print(f"✅ Beat timestamps: {len(beat_times)} beats loaded")
    else:
        print("⚠️ beat_times.npy not found")

    return JSONResponse(content={
        "status"      : "success",
        "song_name"   : song_name,
        "poses_url"   : f"/output/{song_name}/poses.json",
        "audio_url"   : f"/separated/{song_name}/htdemucs/normalized_audio/no_vocals.mp3",
        "style"       : style,
        "tempo"       : float(tempo),
        "total_frames": int(poses_check.shape[0]),
        "beat_times"  : beat_times,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
