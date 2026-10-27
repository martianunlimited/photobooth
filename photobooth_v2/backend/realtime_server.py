"""FastAPI + WebSocket server for remote GPU inference photobooth.

Exposes:
- GET /health: Health status and VRAM telemetry.
- WS /ws: Real-time binary WebSocket frame inference.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
logger = logging.getLogger("realtime_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models into GPU memory before serving requests."""
    logger.info("Initializing GPU pipeline components on startup...")
    pipeline_service.initialize()
    logger.info("GPU pipeline ready to accept WebSocket connections.")
    yield
    logger.info("Shutting down GPU server.")


app = FastAPI(
    title="AI Zombie Mirror Inference Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system readiness, active GPU name, and current allocated VRAM in MB."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_mb = pipeline_service.get_vram_allocated_mb()
    return HealthResponse(
        status="ok",
        gpu=device_name,
        vram_allocated_mb=round(vram_mb, 2),
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept binary JPEG frame bytes, run pipeline inference, and return transformed JPEG bytes."""
    await websocket.accept()
    logger.info(
        f"WebSocket client connected from {websocket.client.host}:{websocket.client.port}"
    )
    try:
        while True:
            # Receive raw binary JPEG bytes
            raw_bytes = await websocket.receive_bytes()
            if not raw_bytes:
                continue

            # Process frame through domain pipeline off the async event loop
            output_bytes = await asyncio.to_thread(
                pipeline_service.process_frame_bytes, raw_bytes
            )

            # Send back transformed JPEG bytes
            await websocket.send_bytes(output_bytes)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally.")
    except Exception:
        logger.exception("WebSocket processing error")
        try:
            await websocket.close()
        except (RuntimeError, OSError) as close_exc:
            logger.debug(f"Error closing websocket: {close_exc}")


def main():
    """Entrypoint for running the server directly."""
    uvicorn.run(
        "realtime_server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
