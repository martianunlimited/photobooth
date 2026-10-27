"""FastAPI backend server for high-quality AI Zombie Photobooth.

Exposes:
- GET /health: Health status and VRAM telemetry.
- POST /zombify: Multipart image upload endpoint returning transformed JPEG with X-Gen-Time header.
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from pipeline import pipeline_service
from schemas import HealthResponse

# Configure server logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models, warm up kernels, and prepare GPU on startup."""
    logger.info("Initializing high-quality GPU photobooth models...")
    pipeline_service.initialize()
    logger.info("Server is fully warmed up and ready for attendee photos.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="AI Zombie Photobooth Inference Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin local Node proxy or browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return GPU health and VRAM allocation."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_mb = pipeline_service.get_vram_allocated_mb()
    return HealthResponse(
        status="ok",
        gpu=device_name,
        vram_allocated_mb=round(vram_mb, 2),
    )


@app.post("/zombify")
async def zombify_endpoint(file: UploadFile = File(...)) -> Response:
    """Accept multipart image upload, run photobooth pipeline, return transformed JPEG."""
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        t0 = time.perf_counter()
        # Offload heavy synchronous GPU inference to threadpool
        output_bytes, _, _ = await asyncio.to_thread(
            pipeline_service.process_image_bytes, raw_bytes
        )
        gen_time_ms = (time.perf_counter() - t0) * 1000.0

        return Response(
            content=output_bytes,
            media_type="image/jpeg",
            headers={"X-Gen-Time": f"{gen_time_ms:.2f}"},
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error processing zombify request")
        raise HTTPException(status_code=500, detail="Internal inference failure")


def main():
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
