"""Automated test suite for the decoupled AI Zombie Photobooth system."""

import json

from photobooth_v2.backend.config import settings
from photobooth_v2.backend.schemas import QualityMetrics


def test_settings_integrity():
    """Verify settings defaults and types."""
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.steps == 15
    assert settings.strength == 0.60
    assert settings.guidance_scale == 6.0
    assert settings.noise_sigma == 15.0
    assert "zombie" in settings.zombie_prompt.lower()
    assert settings.artifacts_dir.exists()
    assert settings.configs_dir.exists()
    assert settings.logs_dir.exists()


def test_manifest_artifact():
    """Verify system_manifest.json exists and adheres to schema."""
    manifest_path = settings.configs_dir / "system_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "python_version" in data
    assert "node_version" in data
    assert "torch_version" in data
    assert "cuda_version" in data
    assert "gpu_name" in data
    hp = data["hyperparameters"]
    assert hp["steps"] == 15
    assert hp["strength"] == 0.60
    assert hp["guidance_scale"] == 6.0
    assert hp["clip_skip"] == 1
    assert hp["noise_sigma"] == 15.0


def test_quality_and_latency_metrics_artifact():
    """Verify quality_and_latency_metrics.json schema and operational thresholds."""
    metrics_path = settings.artifacts_dir / "quality_and_latency_metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = QualityMetrics(**data)
    assert metrics.actual_unet_steps == 9
    assert metrics.guidance_scale == 6.0
    assert metrics.mean_latency_ms < 3500.0  # Must be strictly under 3500ms target
    assert metrics.peak_vram_mb < 8000.0  # Must be under RTX 3070 8GB limit


def test_artifacts_exist_and_non_empty():
    """Verify all intermediate and final test images exist and are non-empty."""
    input_img = settings.artifacts_dir / "test_input_640x480.jpg"
    noisy_img = settings.artifacts_dir / "test_noisy_control_682x512.png"
    output_img = settings.artifacts_dir / "test_output_zombie.jpg"

    assert input_img.exists() and input_img.stat().st_size > 5000
    assert noisy_img.exists() and noisy_img.stat().st_size > 5000
    assert output_img.exists() and output_img.stat().st_size > 5000


def test_log_files_exist():
    """Verify all 4 phase execution logs exist and are non-empty."""
    for phase_name in [
        "phase1_env.log",
        "phase2_backend_bench.log",
        "phase3_nodejs_frontend.log",
        "phase4_e2e_verification.log",
    ]:
        log_path = settings.logs_dir / phase_name
        assert log_path.exists(), f"Missing log: {phase_name}"
        assert log_path.stat().st_size > 0, f"Empty log: {phase_name}"


def test_e2e_proxy_status_pass():
    """Assert that Phase 4 integration log recorded E2E_PROXY_STATUS: PASS."""
    log_path = settings.logs_dir / "phase4_e2e_verification.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "E2E_PROXY_STATUS: PASS" in content
