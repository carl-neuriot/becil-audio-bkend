from fastapi import APIRouter, HTTPException, Query
from s3 import stream_audio_from_s3
from database import SessionLocal

router = APIRouter(prefix="/audio", tags=["audio"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{type}/{filename}")
def get_audio_file(filename: str, type: str):
    """
    Fetch from audio/ads/{filename} for ads
    Fetch from audio/broadcasts/{filename} for broadcasts
    """
    try:
        folder = ""
        if type == "ads":
            folder = "ad_masters"
        elif type == "broadcasts":
            folder = "broadcasts"
        elif type == "songs":
            folder = "song_masters"
        else:
            raise HTTPException(status_code=404, detail="Invalid audio type")
        return stream_audio_from_s3(filename, folder)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found")
