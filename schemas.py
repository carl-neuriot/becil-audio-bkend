from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class AdBase(BaseModel):
    brand: str
    advertisement: str
    duration: int
    filename: Optional[str]
    status: str


class AdCreate(AdBase):
    pass


class AdOut(AdBase):
    id: int
    upload_date: datetime

    model_config = ConfigDict(from_attributes=True)


class BroadcastBase(BaseModel):
    radio_station: str
    broadcast_recording: str
    duration: int
    filename: str
    status: str


class BroadcastCreate(BroadcastBase):
    pass


class BroadcastOut(BroadcastBase):
    id: int
    broadcast_date: datetime
    processing_time: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class SongBase(BaseModel):
    artist: str
    name: str
    filename: str
    duration: int
    status: str


class SongCreate(SongBase):
    pass


class SongOut(SongBase):
    id: int
    upload_date: datetime

    model_config = ConfigDict(from_attributes=True)
