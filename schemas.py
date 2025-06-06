from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from models import StatusEnum, BroadcastStatusEnum

class AdBase(BaseModel):
    brand: str
    advertisement: str
    duration: int 
    status: StatusEnum

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
    duration: int
    status: BroadcastStatusEnum

class BroadcastCreate(BroadcastBase):
    pass

class BroadcastOut(BroadcastBase):
    id: int
    broadcast_date: datetime
    class Config:
        orm_mode = True
