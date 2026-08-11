"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── dataset schemas ────────────────────────────────────────────────────────

class DatasetResponse(BaseModel):
    id: int
    name: str
    path: str
    num_classes: int
    is_upload: bool
    is_ready: bool
    description: str = ""
    created_at: str = ""

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]


# ── device config schemas ──────────────────────────────────────────────────

class DeviceConfigResponse(BaseModel):
    id: int
    name: str
    lookup_csv: Optional[str] = None
    constraint_column: Optional[str] = None
    display_name: str

    class Config:
        from_attributes = True


# ── search schemas ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    dataset_id: int
    device_type: str = "jetson_gpu"
    budget: float                          # latency ms or FLOPs


class SearchResult(BaseModel):
    structure_str: str
    score: float
    constraint: float
    flops: float
    params: float


class SearchResponse(BaseModel):
    result: SearchResult
    device_type: str
    budget: float
    num_classes: int


# ── job schemas ────────────────────────────────────────────────────────────

class JobCreateRequest(BaseModel):
    dataset_id: int
    device_type: str = "jetson_gpu"
    budget: float
    epochs: int = 100
    warmup: int = 5
    batch_size: int = 64
    workers: int = 4
    lr: float = 0.1
    gpu: int = 0
    max_iters: Optional[int] = None        # demo mode: cap total training iterations
    structure_str: Optional[str] = None    # if None, search will be run


class JobResponse(BaseModel):
    id: int
    dataset_id: int
    dataset_name: Optional[str] = None
    device_type: str
    budget: float
    metric: str
    aggregate: str
    epochs: int
    warmup: int
    batch_size: int
    workers: int
    lr: float
    gpu: int
    max_iters: Optional[int] = None
    structure_str: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    best_val_acc: Optional[float] = None
    output_path: Optional[str] = None
    save_dir: Optional[str] = None
    search_score: Optional[float] = None
    search_constraint: Optional[float] = None
    search_flops: Optional[float] = None
    search_params: Optional[float] = None
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class JobStatusResponse(BaseModel):
    id: int
    status: str
    best_val_acc: Optional[float] = None
    error_message: Optional[str] = None


# ── generic ────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
