from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from database import SessionLocal
import crud, schemas, s3

router = APIRouter(prefix="/ads", tags=["ads"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=schemas.AdOut)  # Removed trailing slash
def create_ad(ad: schemas.AdCreate, db: Session = Depends(get_db)):
    return crud.create_ad(db, ad)

@router.get("", response_model=list[schemas.AdOut])  # Removed trailing slash
def list_ads(db: Session = Depends(get_db)):
    return crud.get_ads(db)
