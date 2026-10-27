"""Standalone benchmark script for measuring latency, throughput, and VRAM consumption.

Generates a synthetic portrait test image, executes 3 warmup passes,
followed by 10 timed inference passes, persists output and telemetry metrics.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from pipeline import pipeline_service
from schemas import LatencyMetrics


def setup_benchmark_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("benchmark")
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


def generate_synthetic_portrait(width: int = 512, height: int = 384) -> Image.Image:
    """Synthesize a clean frontal portrait image with facial features."""
    img = Image.new("RGB", (width, height), color=(220, 225, 230))
    draw = ImageDraw.Draw(img)

    # Gradient-like background backdrop
    for y in range(height):
        ratio = y / height
        r = int(210 * (1 - ratio * 0.3))
        g = int(220 * (1 - ratio * 0.25))
        b = int(235 * (1 - ratio * 0.2))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Shoulders / Body
    center_x = width // 2
    draw.ellipse(
        [center_x - 140, height - 120, center_x + 140, height + 80], fill=(45, 55, 70)
    )

    # Neck
    draw.rectangle(
        [center_x - 30, height // 2 + 30, center_x + 30, height - 60],
        fill=(225, 185, 160),
    )

    # Head / Face oval
    face_top = height // 2 - 110
    face_bottom = height // 2 + 65
    face_left = center_x - 70
    face_right = center_x + 70
    draw.ellipse([face_left, face_top, face_right, face_bottom], fill=(240, 195, 170))

    # Hair
    draw.chord(
        [face_left - 8, face_top - 20, face_right + 8, face_top + 70],
        180,
        360,
        fill=(35, 25, 20),
    )

    # Eyebrows
    draw.line(
        [center_x - 45, face_top + 65, center_x - 15, face_top + 65],
        fill=(30, 20, 15),
        width=3,
    )
    draw.line(
        [center_x + 15, face_top + 65, center_x + 45, face_top + 65],
        fill=(30, 20, 15),
        width=3,
    )

    # Eyes
    draw.ellipse(
        [center_x - 40, face_top + 72, center_x - 20, face_top + 84],
        fill=(255, 255, 255),
    )
    draw.ellipse(
        [center_x - 33, face_top + 74, center_x - 27, face_top + 82], fill=(40, 60, 80)
    )
    draw.ellipse(
        [center_x + 20, face_top + 72, center_x + 40, face_top + 84],
        fill=(255, 255, 255),
    )
    draw.ellipse(
        [center_x + 27, face_top + 74, center_x + 33, face_top + 82], fill=(40, 60, 80)
    )

    # Nose
    draw.polygon(
        [
            (center_x, face_top + 90),
            (center_x - 8, face_top + 115),
            (center_x + 8, face_top + 115),
        ],
        fill=(220, 175, 150),
    )

    # Mouth / Lips
    draw.ellipse(
        [center_x - 25, face_top + 128, center_x + 25, face_top + 142],
        fill=(195, 110, 110),
    )
    draw.line(
        [center_x - 20, face_top + 135, center_x + 20, face_top + 135],
        fill=(150, 70, 70),
        width=2,
    )

    return img


def run_benchmark():
    log_file = settings.logs_dir / "phase2_backend_bench.log"
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_benchmark_logger(log_file)
    logger.info("Initializing Phase 2 Pipeline Benchmark...")

    # 1. Generate & persist synthetic input portrait
    input_img = generate_synthetic_portrait(
        settings.resolution_width, settings.resolution_height
    )
    input_path = settings.artifacts_dir / "test_input_512x384.jpg"
    input_img.save(input_path, format="JPEG", quality=95)
    logger.info(
        f"Generated synthetic test input image at {input_path} (size: {os.path.getsize(input_path)} bytes)"
    )

    # Read bytes for benchmark testing
    with open(input_path, "rb") as f:
        raw_test_bytes = f.read()

    # 2. Initialize pipeline service
    pipeline_service.initialize()
    pipeline_service.reset_peak_memory_stats()

    # 3. Warmup passes (3 runs)
    num_warmups = 3
    logger.info(f"Executing {num_warmups} warmup passes...")
    for i in range(num_warmups):
        t0 = time.perf_counter()
        _ = pipeline_service.process_frame_bytes(raw_test_bytes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Warmup [{i + 1}/{num_warmups}]: {dt_ms:.2f} ms")

    # Reset peak memory after warmup
    pipeline_service.reset_peak_memory_stats()

    # 4. Timed benchmark passes (10 runs)
    num_runs = 10
    latencies_ms = []
    last_output_bytes = None
    logger.info(f"Executing {num_runs} timed benchmark passes...")

    for i in range(num_runs):
        t0 = time.perf_counter()
        last_output_bytes = pipeline_service.process_frame_bytes(raw_test_bytes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        logger.info(f"Run [{i + 1}/{num_runs}]: {dt_ms:.2f} ms")

    # 5. Persist generated test output image
    output_path = settings.artifacts_dir / "test_output_zombie.jpg"
    with open(output_path, "wb") as f:
        f.write(last_output_bytes)
    logger.info(
        f"Saved generated output image to {output_path} (size: {os.path.getsize(output_path)} bytes)"
    )

    # 6. Compute metrics
    mean_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    fps_equivalent = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0
    peak_vram_mb = round(pipeline_service.get_peak_vram_mb(), 2)

    metrics = LatencyMetrics(
        warmup_runs=num_warmups,
        benchmark_runs=num_runs,
        mean_latency_ms=round(mean_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        min_latency_ms=round(min_latency, 2),
        max_latency_ms=round(max_latency, 2),
        fps_equivalent=round(fps_equivalent, 2),
        peak_vram_mb=peak_vram_mb,
    )

    metrics_path = settings.artifacts_dir / "latency_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)

    logger.info(
        f"Benchmark results successfully written to {metrics_path}:\n{json.dumps(metrics.model_dump(), indent=2)}"
    )
    logger.info(
        f"Summary: Mean Latency = {mean_latency:.2f} ms | P95 = {p95_latency:.2f} ms | "
        f"FPS = {fps_equivalent:.2f} | Peak VRAM = {peak_vram_mb:.2f} MB"
    )
    logger.info("Phase 2 benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
