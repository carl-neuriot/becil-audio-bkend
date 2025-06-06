from sqlalchemy import Column, Integer, String, DateTime, Enum
from database import Base
import enum
from datetime import datetime


class StatusEnum(str, enum.Enum):
    active = "active"
    inactive = "inactive"


class BroadcastStatusEnum(str, enum.Enum):
    Pending = "Pending"
    Processed = "Processed"


class Ad(Base):
    __tablename__ = "ads"
    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(255), nullable=False)
    advertisement = Column(String(255), nullable=False)
    duration = Column(Integer)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(StatusEnum))


class Broadcast(Base):
    __tablename__ = "broadcasts"
    id = Column(Integer, primary_key=True, index=True)
    radio_station = Column(String(255), nullable=False)
    broadcast_recording = Column(String(255), nullable=False)
    duration = Column(Integer)
    broadcast_date = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(BroadcastStatusEnum))
