import logging

from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from api.query.handler import router as query_router

load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)

app = FastAPI()


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROUTE_BASE = "/api/v1"
app.include_router(query_router, prefix=f"{ROUTE_BASE}/query")
