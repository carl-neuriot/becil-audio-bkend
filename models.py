from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class Ad(Base):
    __tablename__ = "ads"
    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(255), nullable=False)
    advertisement = Column(String(255), nullable=False)
    duration = Column(Integer)
    upload_date = Column(DateTime, default=datetime.now)
    filename = Column(String(255))
    status = Column(String(8))


class Song(Base):
    __tablename__ = 'songs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    artist = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    duration = Column(Integer, nullable=True)
    upload_date = Column(DateTime, default=datetime.now)
    status = Column(String(8), default='Active')


class Broadcast(Base):
    __tablename__ = "broadcasts"
    id = Column(Integer, primary_key=True, index=True)
    radio_station = Column(String(255), nullable=False)
    broadcast_recording = Column(String(255), nullable=False)
    duration = Column(Integer)
    filename = Column(String(255))
    broadcast_date = Column(DateTime, default=datetime.now)
    processing_time = Column(DateTime, nullable=True)
    status = Column(String(9))


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
    clip_type = Column(String(24), nullable=True)


class ExcelReports(Base):
    __tablename__ = 'excel_reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    broadcast_id = Column(Integer, ForeignKey(
        'broadcasts.id'), nullable=False)
    excel_data = Column(LargeBinary, nullable=False)
    excel_filename = Column(String(255), nullable=False)
    created_timestamp = Column(DateTime, default=datetime.now)
    total_ads_detected = Column(Integer, default=0)
    file_size_bytes = Column(Integer, default=0)
