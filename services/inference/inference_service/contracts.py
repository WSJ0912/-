from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SetupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=12, max_length=256)
    role: Literal["doctor"] = "doctor"


class StageImportRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=500)


class Rectangle(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)


class CommitImportRequest(BaseModel):
    adultConfirmed: bool
    chestConfirmed: bool = False
    viewConfirmed: bool = False
    viewPosition: Literal["AP", "PA"] | None = None
    burnedInReviewed: bool
    rectangles: list[Rectangle] = Field(default_factory=list)


class InstallModelRequest(BaseModel):
    packagePath: str


class ReviewRequest(BaseModel):
    studyId: str
    predictionId: str
    decisions: dict[str, Literal["confirmed", "denied", "uncertain"]]
    notes: str = Field(default="", max_length=10000)


class ReportDraftRequest(BaseModel):
    reportId: str | None = None
    studyId: str
    body: str = Field(max_length=50000)
    reviewId: str | None = None


class ConfirmReportRequest(BaseModel):
    reportId: str
    revision: int = Field(ge=1)


class AssistantReportRequest(BaseModel):
    observations: list[dict[str, Any]]
    review: dict[str, Any]
    clinicianText: str = Field(default="", max_length=20000)


class AssistantExperimentRequest(BaseModel):
    aggregateMetrics: dict[str, float | None]
    perClassMetrics: dict[str, dict[str, float | None]]
    experimentNotes: str = Field(default="", max_length=20000)


class ExportReportRequest(BaseModel):
    reportId: str
    revision: int = Field(ge=1)
    outputPath: str
