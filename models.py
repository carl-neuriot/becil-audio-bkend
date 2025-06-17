from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime


class Ad(Base):
    __tablename__ = "ads"
    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(255), nullable=False)
    advertisement = Column(String(255), nullable=False)
    duration = Column(Integer)
    upload_date = Column(DateTime, default=datetime.utcnow)
    duration = Column(Integer)
    filename = Column(String(255))
    status = Column(String(8))  # Stored as VARCHAR(8) in SQLite


class Broadcast(Base):
    __tablename__ = "broadcasts"
    id = Column(Integer, primary_key=True, index=True)
    radio_station = Column(String(255), nullable=False)
    broadcast_recording = Column(String(255), nullable=False)
    duration = Column(Integer)
    filename = Column(String(255))
    broadcast_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(9))  # Stored as VARCHAR(9) in SQLite


class AdDetectionResult(Base):
    __tablename__ = "ad_detection_results"

    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(100), nullable=False)
    description = Column(String(255), nullable=False)
    start_time_seconds = Column(Integer, nullable=False)
    end_time_seconds = Column(Integer, nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    correlation_score = Column(Integer)
    raw_correlation = Column(Integer)
    mfcc_correlation = Column(Integer)
    overlap_duration = Column(Integer)
    detection_timestamp = Column(DateTime)
    processing_status = Column(String(50))
    total_matches_found = Column(Integer)
    ad_id = Column(Integer, nullable=False)
    broadcast_id = Column(Integer, nullable=False)
