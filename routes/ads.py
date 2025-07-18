from fastapi import APIRouter, Depends, UploadFile, File, Body
from sqlalchemy.orm import Session
from database import SessionLocal
from pathlib import Path
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
def list_ads(brand: str = None, advertisement: str = None, db: Session = Depends(get_db)):
    return crud.get_ads(db, brand=brand, advertisement=advertisement)


@router.post("/upload-audio")
def upload_audio(file: UploadFile = File(...)):
    local_folder = Path("ad_masters")
    local_folder.mkdir(parents=True, exist_ok=True)

    local_path = local_folder / file.filename

    with open(local_path, "wb") as buffer:
        buffer.write(file.file.read())
    file.file.seek(0)

    # Upload to S3
    s3_url = s3.upload_file_to_s3(file, file.filename, 'ad_masters')
    return {"url": s3_url}


@router.put("/status")
def update_status(db: Session = Depends(get_db), status=Body(...), id=Body(...)):
    return crud.set_ad_status(db, id, status)
