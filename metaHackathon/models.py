"""
Pydantic models for the Vaccine Cold Chain OpenEnv environment.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─── Action ───────────────────────────────────────────────────────────────────

class VaccineAction(BaseModel):
    action_type: str = Field(
        ...,
        description=(
            "One of: transfer_stock | request_fuel | cancel_outreach | "
            "check_truck_status | do_nothing"
        ),
    )
    source_node: Optional[str] = Field(
        None,
        description="Required for transfer_stock. warehouse | center_a | center_b",
    )
    target_node: Optional[str] = Field(
        None,
        description="Required for transfer_stock and cancel_outreach.",
    )
    node: Optional[str] = Field(
        None,
        description="Required for request_fuel.",
    )
    vial_count: Optional[int] = Field(
        None,
        description="Required for transfer_stock. Number of vials to move.",
        ge=1,
    )


# ─── Observation ──────────────────────────────────────────────────────────────

class NodeObservation(BaseModel):
    name: str
    vial_count: int
    temperature_c: float
    temperature_alarm: bool
    generator_fuel_pct: float = Field(..., ge=0.0, le=1.0)
    hours_until_expiry: float  # 0 = expired/no stock; 999 = never


class VaccineObservation(BaseModel):
    # Per-node state
    nodes: List[NodeObservation]

    # Global fields
    hour: int
    road_to_center_a_open: bool
    road_to_center_b_open: bool
    truck_eta_known: bool
    truck_arriving_in_hours: Optional[float]
    outreach_center_a_in_hours: Optional[float]
    outreach_center_b_in_hours: Optional[float]
    outreach_vials_needed: int

    # Feedback
    last_action_analysis: str
    task_description: str


# ─── State (internal ground truth) ────────────────────────────────────────────

class TemperatureRecord(BaseModel):
    hour: int
    node: str
    real_temp_c: float
    reported_temp_c: float
    sensor_lying: bool


class ActionRecord(BaseModel):
    hour: int
    action: VaccineAction
    result: str  # human-readable outcome


class VaccineState(BaseModel):
    episode_id: str
    task: str       # easy | medium | hard
    seed: int
    hour: int

    # Per-node real values (not noisy)
    real_temperatures: Dict[str, float]
    sensor_is_lying: Dict[str, bool]

    # Running totals
    vials_delivered: int
    vials_spoiled: int
    vials_available_at_start: int

    # Session tracking
    outreach_sessions_completed: int
    outreach_sessions_total: int

    # Histories (for judge graphing / replay)
    temperature_history: List[TemperatureRecord]
    action_history: List[ActionRecord]

    # End-of-episode scores (None until done=True)
    coverage_score: Optional[float]
    waste_penalty: Optional[float]
    missed_penalty: Optional[float]
    final_reward: Optional[float]
