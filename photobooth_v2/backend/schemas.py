"""Pydantic schemas for request/response contracts and quality metrics."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    gpu: str = Field(..., description="Active GPU name")
    vram_allocated_mb: float = Field(..., description="Allocated VRAM in MB")


class QualityMetrics(BaseModel):
    mean_latency_ms: float = Field(
        ..., description="Mean inference latency in milliseconds"
    )
    p95_latency_ms: float = Field(
        ..., description="95th percentile latency in milliseconds"
    )
    actual_unet_steps: int = Field(
        ..., description="Actual number of executed UNet denoising steps"
    )
    guidance_scale: float = Field(..., description="Classifier-free guidance scale")
    peak_vram_mb: float = Field(..., description="Peak allocated VRAM in megabytes")
