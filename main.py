from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import ads, broadcasts
from database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Or specify ["GET", "POST", ...]
    allow_headers=["*"],
)

app.include_router(ads.router, prefix="/api")
app.include_router(broadcasts.router, prefix="/api")
