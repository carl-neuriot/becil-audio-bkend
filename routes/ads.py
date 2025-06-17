from fastapi import APIRouter, Depends, UploadFile, File, Body
from sqlalchemy.orm import Session
from database import SessionLocal
import crud
import schemas
import s3

router = APIRouter(prefix="/ads", tags=["ads"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", response_model=schemas.AdOut)
def create_ad(ad: schemas.AdCreate, db: Session = Depends(get_db)):
    return crud.create_ad(db, ad)


@router.get("", response_model=list[schemas.AdOut])
def list_ads(db: Session = Depends(get_db)):
    return crud.get_ads(db)


@router.post("/upload-audio")
def upload_audio(file: UploadFile = File(...)):
    return {"url": s3.upload_file_to_s3(file, file.filename, 'ad_masters')}

@router.put("/status")
def update_status(db: Session = Depends(get_db), status=Body(...), id=Body(...)):
    return crud.set_ad_status(db, id, status)
