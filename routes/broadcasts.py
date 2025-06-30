from fastapi import APIRouter, Depends, UploadFile, File, Body, HTTPException
import shutil
from threading import Thread
from sqlalchemy.orm import Session
from database import SessionLocal
from final import process_single_radio_clip, fetch_excel_report, extract_clip
import boto3
import os
from dotenv import load_dotenv

import crud
import schemas
import s3

router = APIRouter(prefix="/broadcasts", tags=["broadcasts"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=schemas.BroadcastOut)
def create_broadcast(bc: schemas.BroadcastCreate, db: Session = Depends(get_db)):
    return crud.create_broadcast(db, bc)


@router.get("/", response_model=list[schemas.BroadcastOut])
def list_broadcasts(db: Session = Depends(get_db)):
    return crud.get_broadcasts(db)


@router.post("/upload-audio")  # Removed trailing slash
def upload_audio(file: UploadFile = File(...)):
    return {"url": s3.upload_file_to_s3(file, file.filename, 'broadcasts')}


def download_from_s3(filename, folder="radio_recording"):
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        s3.download_file_from_s3(filename, folder)
        print("Download complete.")
        return
    except Exception as e:
        print(e)
        return {"message": "An error has occured", "error": e}


@router.post("/start-processing")
def start_processing(
    file_name: str = Body(...),
    broadcast_id: int = Body(...),
    db: Session = Depends(get_db)
):
    try:
        download_from_s3(file_name)
        thread = Thread(target=process_single_radio_clip, args=(broadcast_id,))
        thread.start()
        crud.set_broadcast_status(db, broadcast_id, "Processing")

        return {"message": "Processing has started"}
    except Exception as e:
        print(e)
        return {"message": "An error has occured", "error": e}


@router.get("/{broadcast_id}/detections")
def list_detections(broadcast_id: int, db: Session = Depends(get_db)):
    return crud.get_ad_detections_by_broadcast(db, broadcast_id)


@router.get("/{broadcast_id}/report")
def get_excel_report(broadcast_id: int, db:  Session = Depends(get_db)):
    return fetch_excel_report(broadcast_id)


@router.post("/{broadcast_id}/designate_clip")
def designate_clip(
    broadcast_id: int,
    clip_type: str = Body(...),
    brand_artist: str = Body(...),
    advertisement_name: str = Body(...),
    start_time: int = Body(...),
    end_time: int = Body(...),
    db: Session = Depends(get_db)
):
    try:
        print(broadcast_id, clip_type, brand_artist, advertisement_name, start_time)
        broadcast = crud.get_broadcast(db, broadcast_id)
        if not broadcast:
            raise HTTPException(status_code=404, detail="Broadcast not found")
        download_from_s3(broadcast.filename)
        thread = Thread(target=extract_clip, args=(broadcast_id, brand_artist, advertisement_name, clip_type, start_time, end_time))
        thread.start()

        return {"message": "Clip designation underway"}

    except Exception as e:
        print(e)
        return {"message": "An error has occured", "error": e}

