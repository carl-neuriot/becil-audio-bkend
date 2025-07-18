from sqlalchemy.orm import Session
from models import AdDetectionResult  # add this import if not already present
from models import Ad, Broadcast, Song
import schemas

# Ads


def create_ad(db: Session, ad: schemas.AdCreate):
    # This is fine because Pydantic Enums cast to strings
    db_ad = Ad(**ad.dict())
    db.add(db_ad)
    db.commit()
    db.refresh(db_ad)
    return db_ad


def get_ads(db: Session, brand: str = None, advertisement: str = None):
    query = db.query(Ad)
    if brand:
        query = query.filter(Ad.brand.ilike(f"%{brand}%"))
    if advertisement:
        query = query.filter(Ad.advertisement.ilike(f"%{advertisement}%"))
    return query.all()


def get_ad(db: Session, ad_id: int):
    return db.query(Ad).filter(Ad.id == ad_id).first()


def delete_ad(db: Session, ad_id: int):
    db_ad = get_ad(db, ad_id)
    if db_ad:
        db.delete(db_ad)
        db.commit()
    return db_ad


def set_ad_status(db: Session, ad_id: int, status: str):
    ad = get_ad(db, ad_id)
    if ad:
        ad.status = status
        db.commit()
        db.refresh(ad)
    return ad

# Broadcasts


def create_broadcast(db: Session, broadcast: schemas.BroadcastCreate):
    db_bc = Broadcast(**broadcast.dict())  # Same here: enums become strings
    db.add(db_bc)
    db.commit()
    db.refresh(db_bc)
    return db_bc


def get_broadcasts(db: Session, radio_station: str = None, broadcast_recording: str = None):
    query = db.query(Broadcast)
    if radio_station:
        query = query.filter(Broadcast.radio_station.ilike(f"%{radio_station}%"))
    if broadcast_recording:
        query = query.filter(Broadcast.broadcast_recording.ilike(f"%{broadcast_recording}%"))
    return query.all()


def get_broadcast(db: Session, bc_id: int):
    return db.query(Broadcast).filter(Broadcast.id == bc_id).first()


def delete_broadcast(db: Session, bc_id: int):
    db_bc = get_broadcast(db, bc_id)
    if db_bc:
        db.delete(db_bc)
        db.commit()
    return db_bc


def get_ad_detections_by_broadcast(db: Session, broadcast_id: int):
    return db.query(AdDetectionResult).filter(AdDetectionResult.broadcast_id == broadcast_id).all()


def set_broadcast_status(db: Session, broadcast_id: int, status: str):
    broadcast = db.query(Broadcast).filter(
        Broadcast.id == broadcast_id).first()
    if not broadcast:
        return None
    broadcast.status = status
    db.commit()
    db.refresh(broadcast)
    return broadcast


def create_song(db: Session, song: schemas.SongCreate):
    # This is fine because Pydantic Enums cast to strings
    db_song = Song(**song.dict())
    db.add(db_song)
    db.commit()
    db.refresh(db_song)
    return db_song


def get_songs(db: Session):
    return db.query(Song).all()


def get_song(db: Session, song_id: int):
    return db.query(Song).filter(Song.id == song_id).first()


def set_song_status(db: Session, song_id: int, status: str):
    song = get_song(db, song_id)
    if song:
        song.status = status
        db.commit()
        db.refresh(song)
    return song
