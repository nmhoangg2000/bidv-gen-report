"""
BIDV Report AI Agent — FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.templates import router as templates_router
from api.pipeline  import router as pipeline_router
from api.sources   import router as sources_router
from utils.database import engine, Base

app = FastAPI(
    title="BIDV Report AI Agent",
    description="LangGraph-powered agentic pipeline for filling BIDV report templates",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(templates_router, prefix="/api/templates", tags=["Templates"])
app.include_router(pipeline_router,  prefix="/api/pipeline",  tags=["Pipeline"])
app.include_router(sources_router,   prefix="/api/sources",   tags=["Sources"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "BIDV Agent"}


@app.on_event("startup")
async def startup():
    # Tables are created via schema.sql in Docker init
    pass
