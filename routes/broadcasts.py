from fastapi import APIRouter, Depends, UploadFile, File, Body, HTTPException
import shutil
from threading import Thread
from sqlalchemy.orm import Session
from database import SessionLocal
from processing import (
    process_single_radio_clip,
    fetch_excel_report,
    extract_clip,
    reprocess_broadcast,
)

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
def list_broadcasts(
    radio_station: str = None,
    broadcast_recording: str = None,
    db: Session = Depends(get_db),
):
    return crud.get_broadcasts(
        db, radio_station=radio_station, broadcast_recording=broadcast_recording
    )


@router.post("/upload-audio")  # Removed trailing slash
def upload_audio(file: UploadFile = File(...)):
    return {"url": s3.upload_file_to_s3(file, file.filename, "broadcasts")}


@router.post("/start-processing")
def start_processing(
    broadcast_id: int = Body(...),
    db: Session = Depends(get_db),
):
    try:
        broadcast = crud.get_broadcast(db, broadcast_id)
        if not broadcast:
            raise HTTPException(status_code=404, detail="Broadcast not found")

        if broadcast.status == "Processed":
            thread = Thread(target=reprocess_broadcast, args=(broadcast_id,))
            thread.start()
            crud.set_broadcast_status(db, broadcast_id, "Processing")
            return {"message": "Reprocessing has started"}
        elif broadcast.status == "Pending":
            thread = Thread(target=process_single_radio_clip, args=(broadcast_id, broadcast.filename))
            thread.start()
            crud.set_broadcast_status(db, broadcast_id, "Processing")
            return {"message": "Processing has started"}
        else:
            return {"message": "Broadcast is already processing"}

    except Exception as e:
        print(e)
        return {"message": "An error has occured", "error": e}


@router.get("/{broadcast_id}/detections")
def list_detections(broadcast_id: int, db: Session = Depends(get_db)):
    return crud.get_ad_detections_by_broadcast(db, broadcast_id)


@router.get("/{broadcast_id}/report")
def get_excel_report(broadcast_id: int, db: Session = Depends(get_db)):
    return fetch_excel_report(broadcast_id)


@router.post("/{broadcast_id}/designate_clip")
def designate_clip(
    broadcast_id: int,
    clip_type: str = Body(...),
    brand_artist: str = Body(...),
    advertisement_name: str = Body(...),
    start_time: int = Body(...),
    end_time: int = Body(...),
    db: Session = Depends(get_db),
):
    try:
        broadcast = crud.get_broadcast(db, broadcast_id)
        if not broadcast:
            raise HTTPException(status_code=404, detail="Broadcast not found")
        thread = Thread(
            target=extract_clip,
            args=(
                broadcast_id,
                brand_artist,
                advertisement_name,
                clip_type,
                start_time,
                end_time,
                broadcast.filename,
            ),
        )
        thread.start()

        return {"message": "Clip designation underway"}

    except Exception as e:
        print(e)
        return {"message": "An error has occured", "error": e}
