from fastapi import APIRouter, Depends, UploadFile, File, Body
from sqlalchemy.orm import Session
from database import SessionLocal
from pathlib import Path
import crud
import schemas
import s3

router = APIRouter(prefix="/songs", tags=["songs"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", response_model=schemas.SongOut)
def create_song(song: schemas.SongCreate, db: Session = Depends(get_db)):
    return crud.create_song(db, song)


@router.get("", response_model=list[schemas.SongOut])
def list_songs(db: Session = Depends(get_db)):
    return crud.get_songs(db)


@router.post("/upload-audio")
def upload_audio(file: UploadFile = File(...)):
    local_folder = Path("song_masters")
    local_folder.mkdir(parents=True, exist_ok=True)

    local_path = local_folder / file.filename

    with open(local_path, "wb") as buffer:
        buffer.write(file.file.read())
    file.file.seek(0)

    # Upload to S3
    s3_url = s3.upload_file_to_s3(file, file.filename, 'song_masters')
    return {"url": s3_url}


@router.put("/status")
def update_status(db: Session = Depends(get_db), status=Body(...), id=Body(...)):
    return crud.set_song_status(db, id, status)
