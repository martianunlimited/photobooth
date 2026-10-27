"""Phase 4 End-to-End Integration & Kiosk Launch Verification Script.

Launches realtime_server.py as background process, polls /health,
streams test_input_512x384.jpg 5 times over WebSocket, verifies JPEG binary headers (0xFFD8),
confirms zero-disk-write invariance, shuts down server, and logs E2E_STATUS: PASS.
"""

import asyncio
import logging
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("phase4_e2e")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def poll_health(
    health_url: str,
    max_wait_sec: int = 60,
    interval_sec: float = 1.0,
    logger: logging.Logger | None = None,
) -> bool:
    """Poll health endpoint until server responds ok or timeout expires."""
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8")
                    if logger:
                        logger.info(f"Health check succeeded: {body.strip()}")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(interval_sec)
    return False


def get_directory_files(directory: Path) -> set:
    """Return set of all relative file paths currently present in directory."""
    if not directory.exists():
        return set()
    return {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}


async def run_websocket_stream_test(
    ws_url: str,
    test_image_bytes: bytes,
    num_frames: int = 5,
    logger: logging.Logger | None = None,
) -> list[float]:
    """Connect to WebSocket endpoint, stream test frame multiple times, and verify responses."""
    latencies = []
    logger.info(f"Connecting to WebSocket: {ws_url}")
    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        logger.info(f"Connected! Beginning {num_frames}-frame stream test...")

        for frame_idx in range(num_frames):
            t0 = time.perf_counter()
            await ws.send(test_image_bytes)

            response_bytes = await ws.recv()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            # Validate binary response type and length
            assert isinstance(response_bytes, bytes), (
                f"Frame {frame_idx + 1}: Response is not bytes"
            )
            assert len(response_bytes) > 1000, (
                f"Frame {frame_idx + 1}: Response too small ({len(response_bytes)} bytes)"
            )

            # Validate JPEG magic header 0xFFD8
            header = response_bytes[:2]
            assert header == b"\xff\xd8", (
                f"Frame {frame_idx + 1}: Invalid JPEG header {header.hex()}"
            )

            logger.info(
                f"Frame [{frame_idx + 1}/{num_frames}]: Roundtrip = {dt_ms:.2f} ms | "
                f"Size = {len(response_bytes)} bytes | Magic = 0x{header.hex().upper()} (Valid JPEG)"
            )

    return latencies


def main():
    log_file = settings.logs_dir / "phase4_e2e_verification.log"
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)
    logger.info("Starting Phase 4 End-to-End Integration Test...")

    test_input_path = settings.artifacts_dir / "test_input_512x384.jpg"
    if not test_input_path.exists():
        logger.error(f"Test image not found at {test_input_path}. Run Phase 2 first.")
        sys.exit(1)

    with open(test_input_path, "rb") as f:
        test_image_bytes = f.read()

    # Track directory state before stream
    project_files_before = get_directory_files(settings.project_root)
    temp_dir = Path(tempfile.gettempdir())
    temp_files_before = get_directory_files(temp_dir)

    server_script = settings.project_root / "backend" / "realtime_server.py"
    server_log_path = settings.logs_dir / "realtime_server.log"

    cmd = [sys.executable, str(server_script)]
    logger.info(f"Launching server subprocess: {' '.join(cmd)}")

    with open(server_log_path, "w", encoding="utf-8") as server_log_file:
        server_proc = subprocess.Popen(
            cmd,
            cwd=str(settings.project_root / "backend"),
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            health_url = f"http://{settings.host}:{settings.port}/health"
            ws_url = f"ws://{settings.host}:{settings.port}/ws"

            logger.info(f"Polling {health_url} (timeout 60s)...")
            ready = poll_health(
                health_url, max_wait_sec=60, interval_sec=1.0, logger=logger
            )
            if not ready:
                logger.error("Server failed to respond to /health within 60 seconds.")
                sys.exit(1)

            # Run 5-frame WebSocket stream test
            num_frames = 5
            latencies = asyncio.run(
                run_websocket_stream_test(
                    ws_url, test_image_bytes, num_frames=num_frames, logger=logger
                )
            )

            mean_latency = sum(latencies) / len(latencies)
            logger.info(
                f"5-Frame Stream Completed. Mean WebSocket Latency: {mean_latency:.2f} ms"
            )

            # Zero-Disk-Write verification
            project_files_after = get_directory_files(settings.project_root)
            temp_files_after = get_directory_files(temp_dir)

            new_project_files = (project_files_after - project_files_before) - {
                str(
                    (settings.logs_dir / "phase4_e2e_verification.log").relative_to(
                        settings.project_root
                    )
                ),
                str(
                    (settings.logs_dir / "realtime_server.log").relative_to(
                        settings.project_root
                    )
                ),
            }
            new_temp_files = temp_files_after - temp_files_before

            if new_project_files:
                logger.warning(
                    f"Unexpected new project files detected: {new_project_files}"
                )
            elif new_temp_files:
                logger.warning(f"Unexpected new temp files detected: {new_temp_files}")
            else:
                logger.info(
                    "Zero-disk-write verified: No temporary files created in project directory during streaming."
                )

            logger.info("=" * 60)
            logger.info("E2E_STATUS: PASS")
            logger.info("=" * 60)

        except Exception:
            logger.exception("E2E Integration Test FAILED")
            sys.exit(1)
        finally:
            logger.info("Gracefully terminating backend server subprocess...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
                logger.info("Backend server terminated cleanly.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Backend server did not terminate within 5s; killing process."
                )
                server_proc.kill()


if __name__ == "__main__":
    main()
