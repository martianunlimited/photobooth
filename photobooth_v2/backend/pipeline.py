"""High-quality IP2P + ControlNet photobooth pipeline service.

Replicates the photobooth_api.py reference quality:
- 15 steps with DPMSolverMultistepScheduler (use_karras_sigmas=True)
- strength=0.60 (9 actual UNet steps)
- guidance_scale=6.0 (full CFG)
- Standard full-precision VAE
- Aspect-ratio preserving resize to shortest side 512 (multiple of 8)
- Additive Gaussian noise (sigma=15) for ControlNet condition
- NSFW safety retry (up to 3 seeds)
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

logger = logging.getLogger("high_quality_pipeline")


class HighQualityZombiePipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe: StableDiffusionControlNetImg2ImgPipeline = None

    def initialize(self):
        """Initialize pipeline with ControlNet IP2P and DPMSolverMultistepScheduler."""
        logger.info(
            f"Initializing HighQualityZombiePipeline on {self.device} ({self.dtype})..."
        )

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 1. Load ControlNet IP2P
        logger.info(f"Loading ControlNet: {settings.controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            settings.controlnet_id,
            torch_dtype=self.dtype,
        )

        # 2. Load Base Checkpoint
        if settings.sd_checkpoint_path and os.path.exists(settings.sd_checkpoint_path):
            logger.info(
                f"Loading single-file checkpoint: {settings.sd_checkpoint_path}"
            )
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                settings.sd_checkpoint_path,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                load_safety_checker=True,
            )
        else:
            logger.info(f"Loading base model: {settings.base_model_id}")
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                settings.base_model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
            )

        # 3. Configure DPMSolverMultistepScheduler
        logger.info("Configuring DPMSolverMultistepScheduler with Karras sigmas...")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )

        # 4. Hardware layout optimizations
        if self.device == "cuda":
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.controlnet.to(memory_format=torch.channels_last)
            pipe.to(self.device)

        self.pipe = pipe
        logger.info("Pipeline loaded successfully.")

        # 5. Pre-compile CUDA kernels with 2 warmup passes
        self.warmup(num_passes=2)

    def warmup(self, num_passes: int = 2):
        """Execute warmup passes to pre-compile CUDA kernels before handling user requests."""
        logger.info(f"Executing {num_passes} warmup passes...")
        dummy_img = Image.new("RGB", (680, 512), color=(120, 120, 120))
        for i in range(num_passes):
            with torch.inference_mode():
                _ = self.pipe(
                    prompt=settings.zombie_prompt,
                    negative_prompt=settings.negative_prompt,
                    image=dummy_img,
                    control_image=dummy_img,
                    num_inference_steps=settings.steps,
                    strength=settings.strength,
                    guidance_scale=settings.guidance_scale,
                    controlnet_conditioning_scale=settings.controlnet_conditioning_scale,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            logger.info(f"Warmup pass [{i + 1}/{num_passes}] complete.")

    def preprocess_image(
        self, orig_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Aspect-ratio preserving resize to shortest side 512 (multiple of 8), with noise condition."""
        w, h = orig_img.size
        if w < h:
            new_w = 512
            new_h = int(round((h * 512.0 / w) / 8.0) * 8)
        else:
            new_h = 512
            new_w = int(round((w * 512.0 / h) / 8.0) * 8)

        clean_img = orig_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Add Gaussian noise (mean=0, sigma=15) clipped to [0, 255]
        clean_arr = np.array(clean_img, dtype=np.float32)
        noise = np.random.normal(
            loc=0.0, scale=settings.noise_sigma, size=clean_arr.shape
        )
        noisy_arr = np.clip(clean_arr + noise, 0.0, 255.0).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_arr)

        return clean_img, noisy_img

    def zombify(self, clean_img: Image.Image, noisy_img: Image.Image) -> Image.Image:
        """Run inference with up to 3 seed retries if NSFW safety checker triggers."""
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        base_seed = settings.seed
        for attempt in range(3):
            seed = base_seed + attempt
            generator = torch.Generator(device=self.device).manual_seed(seed)
            with torch.inference_mode():
                output = self.pipe(
                    prompt=settings.zombie_prompt,
                    negative_prompt=settings.negative_prompt,
                    image=clean_img,
                    control_image=noisy_img,
                    num_inference_steps=settings.steps,
                    strength=settings.strength,
                    guidance_scale=settings.guidance_scale,
                    controlnet_conditioning_scale=settings.controlnet_conditioning_scale,
                    generator=generator,
                )

            is_nsfw = (
                hasattr(output, "nsfw_content_detected")
                and output.nsfw_content_detected
                and output.nsfw_content_detected[0]
            )

            if not is_nsfw:
                return output.images[0]

            logger.warning(
                f"NSFW safety checker triggered on attempt {attempt + 1}/3 with seed {seed}. Retrying..."
            )

        logger.warning(
            "All 3 seeds triggered NSFW checker. Returning clean input as fallback."
        )
        return clean_img

    def process_image_bytes(
        self, image_bytes: bytes
    ) -> Tuple[bytes, Image.Image, Image.Image]:
        """Convert bytes -> clean & noisy images -> run zombify -> return JPEG bytes."""
        orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        clean_img, noisy_img = self.preprocess_image(orig_img)
        output_img = self.zombify(clean_img, noisy_img)

        buffer = io.BytesIO()
        output_img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), clean_img, noisy_img

    def get_vram_allocated_mb(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
        return 0.0

    def get_peak_vram_mb(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        return 0.0

    def reset_peak_memory_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


pipeline_service = HighQualityZombiePipeline()
