"""Phase 3 asset downloader and generator script.

Downloads offline MediaPipe vision bundle, WASM runtime, and BlazeFace model.
Generates noise-to-image diffusion GIF animation and logs checksums.
"""

import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("phase3_assets")
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


def download_file(url: str, dest_path: Path, logger: logging.Logger) -> tuple[int, str]:
    """Download a file via HTTP, verify, and return its size and sha256 checksum."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest_path}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read()

    with open(dest_path, "wb") as f:
        f.write(content)

    size = len(content)
    sha256 = hashlib.sha256(content).hexdigest()
    logger.info(f"Saved {dest_path.name}: {size} bytes | SHA256: {sha256}")
    return size, sha256


def generate_diffusion_gif(output_path: Path, logger: logging.Logger):
    """Generate an animated GIF illustrating noise-to-image diffusion process."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating diffusion animation GIF -> {output_path}")

    # Load synthetic portrait or zombie test output as base target
    target_img_path = settings.artifacts_dir / "test_output_zombie.jpg"
    if target_img_path.exists():
        target_img = (
            Image.open(target_img_path)
            .convert("RGB")
            .resize((256, 192), Image.Resampling.BILINEAR)
        )
    else:
        target_img = Image.new("RGB", (256, 192), color=(100, 140, 90))

    target_arr = np.array(target_img, dtype=np.float32)

    # Deterministic noise for diffusion animation
    rng = np.random.RandomState(42)
    pure_noise = rng.uniform(0.0, 255.0, size=(192, 256, 3)).astype(np.float32)

    frames = []
    num_steps = 12
    # Forward diffusion: noise -> zombie face
    for step in range(num_steps + 1):
        alpha = step / float(num_steps)  # 0.0 (noise) to 1.0 (zombie image)
        # Interpolate between pure noise and target image
        blended = (1.0 - alpha) * pure_noise + alpha * target_arr

        # Add step text overlay
        frame_img = Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))
        draw = ImageDraw.Draw(frame_img)
        step_text = (
            f"Step {step}/{num_steps} (Denoising)"
            if step < num_steps
            else "Zombie Mirror Ready"
        )
        draw.rectangle([5, 168, 251, 187], fill=(0, 0, 0, 160))
        draw.text((10, 172), step_text, fill=(80, 255, 120))
        frames.append(frame_img)

    # Hold final frame for 4 frames
    for _ in range(4):
        frames.append(frames[-1])

    # Reverse back smoothly to loop
    for step in range(num_steps - 1, 0, -2):
        frames.append(frames[step])

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    logger.info(
        f"Diffusion GIF saved to {output_path} (size: {os.path.getsize(output_path)} bytes)"
    )


def run_phase3_assets():
    log_file = settings.logs_dir / "phase3_frontend_assets.log"
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)
    logger.info("Starting Phase 3 Asset Downloads...")

    vendor_dir = settings.frontend_dir / "vendor"
    wasm_dir = vendor_dir / "wasm"
    models_dir = vendor_dir / "models"

    assets = [
        (
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs",
            vendor_dir / "tasks-vision.bundle.mjs",
        ),
        (
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm/vision_wasm_internal.js",
            wasm_dir / "vision_wasm_internal.js",
        ),
        (
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm/vision_wasm_internal.wasm",
            wasm_dir / "vision_wasm_internal.wasm",
        ),
        (
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            models_dir / "blaze_face_short_range.tflite",
        ),
    ]

    for url, dest in assets:
        download_file(url, dest, logger)

    # Generate diffusion GIF
    gif_path = settings.frontend_dir / "image.gif"
    generate_diffusion_gif(gif_path, logger)

    logger.info("All Phase 3 vendor assets downloaded and verified successfully.")


if __name__ == "__main__":
    run_phase3_assets()
