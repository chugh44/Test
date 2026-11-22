from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class HorizonOut(BaseModel):
    label: str
    days: int
    target_date: str
    base: float
    bull: float
    bear: float
    lo80: float
    hi80: float
    lo95: float
    hi95: float
    expected_return_pct: float
    bull_upside_pct: float
    bear_downside_pct: float
    prob_up: float
    calibration: float
    model_dispersion: float
    best_skill: float | None
    weights: dict[str, float]
    model_point: dict[str, float] = Field(default_factory=dict, description="each model's median price")


class ModelMetrics(BaseModel):
    name: str
    family: str
    status: str
    by_horizon: dict[str, dict] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class LevelOut(BaseModel):
    price: float
    kind: str
    touches: int
    strength: float
    last_touch: str


class ForecastData(BaseModel):
    last_price: float
    last_date: str
    horizons: list[HorizonOut]
    models: list[ModelMetrics]
    skipped_models: dict[str, str]
    path: list[dict]
    levels: list[LevelOut]
    method: str


class ForecastFinding(AgentFinding):
    data: ForecastData | None = None
