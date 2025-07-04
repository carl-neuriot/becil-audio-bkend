from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import ads, broadcasts, audio, songs
from database import Base, engine
import logging

Base.metadata.create_all(bind=engine)

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ads.router, prefix="/api")
app.include_router(songs.router, prefix="/api")
app.include_router(broadcasts.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
