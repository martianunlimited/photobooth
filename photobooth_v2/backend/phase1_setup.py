"""Phase 1: Environment Scaffolding & Deterministic Tensor Persistence.

Generates deterministic Gaussian noise array and precomputes CLIP text embeddings.
Writes configs/system_manifest.json and logs to logs/phase1_env.log.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("phase1")
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


def run_phase1():
    log_file = settings.logs_dir / "phase1_env.log"
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.configs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file)
    logger.info("Starting Phase 1 Setup...")

    # Enforce deterministic random seed
    logger.info(f"Setting random seed to {settings.seed}")
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    rng = np.random.RandomState(settings.seed)

    # 1. Generate Static Gaussian Noise (384, 512, 3)
    noise_shape = (
        settings.resolution_height,
        settings.resolution_width,
        3,
    )  # (384, 512, 3)
    logger.info(
        f"Generating static Gaussian noise array with shape {noise_shape}, "
        f"mean={settings.gaussian_mean}, sigma={settings.gaussian_sigma}"
    )
    noise = rng.normal(
        loc=settings.gaussian_mean,
        scale=settings.gaussian_sigma,
        size=noise_shape,
    ).astype(np.float32)

    noise_path = settings.artifacts_dir / "static_noise_512x384.npy"
    np.save(noise_path, noise)
    logger.info(
        f"Saved deterministic noise to {noise_path} (size: {os.path.getsize(noise_path)} bytes)"
    )

    # 2. Extract CLIP Text Embeddings
    checkpoint_source = settings.sd_checkpoint_path or settings.base_model_id
    logger.info(f"Loading CLIP text encoder from source: {checkpoint_source}")

    if settings.sd_checkpoint_path and os.path.exists(settings.sd_checkpoint_path):
        logger.info(f"Loading pipeline from single file: {settings.sd_checkpoint_path}")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            settings.sd_checkpoint_path,
            load_safety_checker=True,
            torch_dtype=torch.float16,
        )
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
    else:
        logger.info(
            f"Loading tokenizer & text_encoder from pretrained: {settings.base_model_id}"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            settings.base_model_id, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            settings.base_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)

    logger.info(f"Tokenizing locked zombie prompt: '{settings.zombie_prompt}'")
    text_inputs = tokenizer(
        settings.zombie_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,  # 77
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]

    # Shape: [1, 77, 768], dtype: torch.float16
    # Save as CPU tensor so it can be verified/loaded by any python process
    prompt_embeds_cpu = prompt_embeds.to(dtype=torch.float16, device="cpu")
    logger.info(
        f"Generated prompt_embeds with shape {list(prompt_embeds_cpu.shape)} "
        f"and dtype {prompt_embeds_cpu.dtype}"
    )

    embeds_path = settings.artifacts_dir / "precomputed_prompt_embeds.pt"
    torch.save(prompt_embeds_cpu, embeds_path)
    logger.info(
        f"Saved precomputed prompt embeddings to {embeds_path} (size: {os.path.getsize(embeds_path)} bytes)"
    )

    # 3. System Manifest
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "None"

    manifest_data = {
        "seed": settings.seed,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "checkpoint_source": checkpoint_source,
        "resolution": [settings.resolution_width, settings.resolution_height],
        "gaussian_sigma": settings.gaussian_sigma,
    }

    manifest_path = settings.configs_dir / "system_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"System manifest successfully written to {manifest_path}:\n{json.dumps(manifest_data, indent=2)}"
    )
    logger.info("Phase 1 completed successfully.")


if __name__ == "__main__":
    run_phase1()
