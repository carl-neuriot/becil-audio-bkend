from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
import crud, schemas

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
