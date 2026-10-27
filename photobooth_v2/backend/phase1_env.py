"""Phase 1: Environment Initialization & System Manifest Persistence.

Detects Python, Node.js, PyTorch, CUDA, and GPU hardware configurations.
Writes configs/system_manifest.json and logs to logs/phase1_env.log.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("phase1_env")
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


def get_node_version() -> str:
    try:
        out = subprocess.check_output(["node", "-v"], text=True).strip()
        return out
    except Exception as e:
        return f"Unknown ({e})"


def get_npm_version() -> str:
    try:
        out = subprocess.check_output(["npm", "-v"], shell=True, text=True).strip()
        return out
    except Exception as e:
        return f"Unknown ({e})"


def run_phase1():
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.configs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_file = settings.logs_dir / "phase1_env.log"
    logger = setup_logger(log_file)
    logger.info("Initializing Phase 1 Environment Initialization...")

    # Hardware & runtime checks
    node_ver = get_node_version()
    npm_ver = get_npm_version()
    py_ver = sys.version.replace("\n", " ")
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

    logger.info(f"Python Runtime: {py_ver}")
    logger.info(f"Node.js Runtime: {node_ver} (npm: {npm_ver})")
    logger.info(f"PyTorch Version: {torch_ver} (CUDA: {cuda_ver})")
    logger.info(f"GPU Hardware: {gpu_name}")

    # Verify node_modules/express exists
    express_path = settings.frontend_node_dir / "node_modules" / "express"
    if not express_path.exists():
        logger.error(f"Express module not found at {express_path}")
        sys.exit(1)
    logger.info(f"Verified Express installed at: {express_path}")

    # Build system manifest
    manifest_data = {
        "python_version": py_ver,
        "node_version": node_ver,
        "torch_version": torch_ver,
        "cuda_version": cuda_ver,
        "gpu_name": gpu_name,
        "hyperparameters": {
            "steps": settings.steps,
            "strength": settings.strength,
            "guidance_scale": settings.guidance_scale,
            "clip_skip": settings.clip_skip,
            "noise_sigma": settings.noise_sigma,
        },
    }

    manifest_path = settings.configs_dir / "system_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Persisted configs/system_manifest.json:\n{json.dumps(manifest_data, indent=2)}"
    )
    logger.info("Phase 1 initialization completed successfully.")


if __name__ == "__main__":
    run_phase1()
