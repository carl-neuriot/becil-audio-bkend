from pydantic import BaseModel
from datetime import datetime


class AdBase(BaseModel):
    brand: str
    advertisement: str
    filename: str
    duration: int
    status: str


class AdCreate(AdBase):
    pass


class AdOut(AdBase):
    id: int
    upload_date: datetime

    class Config:
        orm_mode = True


class BroadcastBase(BaseModel):
    radio_station: str
    broadcast_recording: str
    filename: str
    duration: int
    status: str


class BroadcastCreate(BroadcastBase):
    pass


class BroadcastOut(BroadcastBase):
    id: int
    broadcast_date: datetime

    class Config:
        orm_mode = True
