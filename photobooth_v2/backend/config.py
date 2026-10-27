"""Configuration and operational settings using pydantic-settings."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Determinism
    seed: int = 42

    # Generation Hyperparameters
    steps: int = 15
    strength: float = 0.60
    guidance_scale: float = 6.0
    clip_skip: int = 1
    noise_sigma: float = 15.0
    controlnet_conditioning_scale: float = 1.0

    # Prompts
    zombie_prompt: str = (
        "Imagine yourself as a scary undead creature, a zombie, with tattered clothes."
    )
    negative_prompt: str = "nsfw, nudity, female_nudity, topless"

    # Model IDs
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: str = "lllyasviel/control_v11e_sd15_ip2p"
    sd_checkpoint_path: Optional[str] = None

    # Directories
    project_root: Path = Path(__file__).resolve().parent.parent
    configs_dir: Path = project_root / "configs"
    logs_dir: Path = project_root / "logs"
    artifacts_dir: Path = project_root / "artifacts"
    frontend_node_dir: Path = project_root / "frontend_node"

    # Network
    host: str = "0.0.0.0"
    port: int = 8000
    remote_gpu_url: str = "http://127.0.0.1:8000/zombify"


settings = Settings()
