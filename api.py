from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
import os

from process_audio import process_audio

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):

    # Save uploaded file to audio/
    save_path = f"audio/{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📥 Received: {file.filename}")

    # Run full pipeline
    result = process_audio(save_path)

    return JSONResponse(content={
    "status"   : "success",
    "no_vocals": result["no_vocals_path"],
    "beat_map" : result["beat_map"],
     })

    if result is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Pipeline failed — check terminal for details"}
        )

    return JSONResponse(content={
        "status"       : "success",
        "message"      : "Pipeline complete",
        "no_vocals"    : result,
        "beat_map": f"output/{song_name}/beat_map.png"
    })


@app.get("/")
def root():
    return {"message": "Dance Choreography API is running"}
