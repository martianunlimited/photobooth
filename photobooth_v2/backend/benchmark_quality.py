"""Phase 2 benchmark script: quality validation, artifact persistence, and latency measurement.

Generates artifacts/test_input_640x480.jpg,
persists intermediate noisy ControlNet image artifacts/test_noisy_control_682x512.png,
persists final zombie portrait artifacts/test_output_zombie.jpg,
and logs metrics to artifacts/quality_and_latency_metrics.json and logs/phase2_backend_bench.log.
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
from schemas import QualityMetrics


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("benchmark_quality")
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


def generate_test_portrait(width: int = 640, height: int = 480) -> Image.Image:
    """Generate a clean synthetic frontal portrait test image at 640x480 resolution."""
    img = Image.new("RGB", (width, height), color=(225, 230, 235))
    draw = ImageDraw.Draw(img)

    # Gradient studio backdrop
    for y in range(height):
        factor = y / height
        r = int(215 - factor * 35)
        g = int(225 - factor * 30)
        b = int(240 - factor * 25)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    cx = width // 2
    # Shoulders
    draw.ellipse([cx - 175, height - 150, cx + 175, height + 100], fill=(40, 50, 65))
    # Neck
    draw.rectangle(
        [cx - 38, height // 2 + 40, cx + 38, height - 80], fill=(230, 190, 165)
    )
    # Head oval
    face_top = height // 2 - 140
    face_bottom = height // 2 + 80
    face_left = cx - 88
    face_right = cx + 88
    draw.ellipse([face_left, face_top, face_right, face_bottom], fill=(245, 200, 175))

    # Dark Hair
    draw.chord(
        [face_left - 10, face_top - 25, face_right + 10, face_top + 90],
        180,
        360,
        fill=(30, 20, 15),
    )

    # Eyes & Eyebrows
    draw.line(
        [cx - 55, face_top + 80, cx - 18, face_top + 80], fill=(25, 18, 12), width=4
    )
    draw.line(
        [cx + 18, face_top + 80, cx + 55, face_top + 80], fill=(25, 18, 12), width=4
    )
    draw.ellipse(
        [cx - 50, face_top + 90, cx - 25, face_top + 105], fill=(255, 255, 255)
    )
    draw.ellipse([cx - 41, face_top + 92, cx - 34, face_top + 103], fill=(35, 55, 75))
    draw.ellipse(
        [cx + 25, face_top + 90, cx + 50, face_top + 105], fill=(255, 255, 255)
    )
    draw.ellipse([cx + 34, face_top + 92, cx + 41, face_top + 103], fill=(35, 55, 75))

    # Nose
    draw.polygon(
        [(cx, face_top + 110), (cx - 10, face_top + 140), (cx + 10, face_top + 140)],
        fill=(225, 180, 155),
    )

    # Mouth
    draw.ellipse(
        [cx - 30, face_top + 155, cx + 30, face_top + 172], fill=(200, 115, 115)
    )
    draw.line(
        [cx - 25, face_top + 163, cx + 25, face_top + 163], fill=(150, 75, 75), width=2
    )

    return img


def run_benchmark():
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_file = settings.logs_dir / "phase2_backend_bench.log"
    logger = setup_logger(log_file)
    logger.info("Initializing Phase 2 Quality & Latency Benchmark...")

    # 1. Synthesize input portrait at 640x480
    input_img = generate_test_portrait(640, 480)
    input_path = settings.artifacts_dir / "test_input_640x480.jpg"
    input_img.save(input_path, format="JPEG", quality=95)
    logger.info(
        f"Saved synthetic portrait to {input_path} (size: {os.path.getsize(input_path)} bytes)"
    )

    with open(input_path, "rb") as f:
        test_bytes = f.read()

    # 2. Initialize pipeline service
    pipeline_service.initialize()
    pipeline_service.reset_peak_memory_stats()

    # Preprocess and persist intermediate noisy ControlNet condition image
    clean_img, noisy_img = pipeline_service.preprocess_image(input_img)
    noisy_path = settings.artifacts_dir / "test_noisy_control_682x512.png"
    noisy_img.save(noisy_path, format="PNG")
    logger.info(
        f"Saved intermediate noisy ControlNet condition image to {noisy_path} "
        f"(dimensions: {noisy_img.size}, size: {os.path.getsize(noisy_path)} bytes)"
    )

    # 3. Timed benchmark passes
    num_runs = 3
    latencies = []
    final_output_bytes = None
    logger.info(f"Executing {num_runs} timed high-quality inference passes...")

    for i in range(num_runs):
        t0 = time.perf_counter()
        final_output_bytes, _, _ = pipeline_service.process_image_bytes(test_bytes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        logger.info(f"Pass [{i + 1}/{num_runs}]: Latency = {dt_ms:.2f} ms")

    # 4. Save final output image
    output_path = settings.artifacts_dir / "test_output_zombie.jpg"
    with open(output_path, "wb") as f:
        f.write(final_output_bytes)
    logger.info(
        f"Saved generated high-quality zombie output to {output_path} (size: {os.path.getsize(output_path)} bytes)"
    )

    # 5. Compute metrics
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    actual_unet_steps = int(round(settings.steps * settings.strength))
    peak_vram_mb = round(pipeline_service.get_peak_vram_mb(), 2)

    metrics = QualityMetrics(
        mean_latency_ms=round(mean_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        actual_unet_steps=actual_unet_steps,
        guidance_scale=settings.guidance_scale,
        peak_vram_mb=peak_vram_mb,
    )

    metrics_path = settings.artifacts_dir / "quality_and_latency_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)

    logger.info(
        f"Persisted {metrics_path}:\n{json.dumps(metrics.model_dump(), indent=2)}"
    )
    logger.info(
        f"Mean Latency: {mean_latency:.2f} ms | P95: {p95_latency:.2f} ms | Peak VRAM: {peak_vram_mb:.2f} MB"
    )
    logger.info("Phase 2 benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
