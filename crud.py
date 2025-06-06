
from sqlalchemy.orm import Session
from models import Ad, Broadcast
import schemas

# Ads
def create_ad(db: Session, ad: schemas.AdCreate):
    db_ad = Ad(**ad.dict())
    db.add(db_ad)
    db.commit()
    db.refresh(db_ad)
    return db_ad

def get_ads(db: Session):
    return db.query(Ad).all()

def get_ad(db: Session, ad_id: int):
    return db.query(Ad).filter(Ad.id == ad_id).first()

def delete_ad(db: Session, ad_id: int):
    db_ad = get_ad(db, ad_id)
    if db_ad:
        db.delete(db_ad)
        db.commit()
    return db_ad

# Broadcasts
def create_broadcast(db: Session, broadcast: schemas.BroadcastCreate):
    db_bc = Broadcast(**broadcast.dict())
    db.add(db_bc)
    db.commit()
    db.refresh(db_bc)
    return db_bc

def get_broadcasts(db: Session):
    return db.query(Broadcast).all()

def get_broadcast(db: Session, bc_id: int):
    return db.query(Broadcast).filter(Broadcast.id == bc_id).first()

def delete_broadcast(db: Session, bc_id: int):
    db_bc = get_broadcast(db, bc_id)
    if db_bc:
        db.delete(db_bc)
        db.commit()
    return db_bc
