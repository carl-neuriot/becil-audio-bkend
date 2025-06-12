from fastapi import APIRouter, Depends, UploadFile, File, Body
import shutil
from threading import Thread
from sqlalchemy.orm import Session
from database import SessionLocal
from final import process_single_radio_clip
import boto3
import os
from dotenv import load_dotenv

import crud, schemas, s3

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
    return {"url": s3.upload_file_to_s3(file, file.filename)}

@router.post("/start-processing")
def start_processing(
    file_name: str = Body(...),
    broadcast_id: int = Body(...),
    db: Session = Depends(get_db)
):
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    s3_key = f"broadcasts/{file_name}"
    local_dir = "radio_recording"
    local_path = os.path.join(local_dir, file_name)

    # Step 2: Clear the radio_recording folder
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir)

    # Step 3: Download the file into the folder
    s3.download_file_from_s3(file_name, "radio_recording")
    print("Download complete.")

    # Step 4: Call the processing function in a separate thread
    thread = Thread(target=process_single_radio_clip, args=(broadcast_id,))
    thread.start()
    crud.set_broadcast_status(db, broadcast_id, "Processing")

    # Step 5: Return immediately
    return {"Message": "Processing has started"}

@router.get("/{broadcast_id}/detections")
def list_detections(broadcast_id: int, db: Session = Depends(get_db)):
    print('id get', broadcast_id)
    return crud.get_ad_detections_by_broadcast(db, broadcast_id)
