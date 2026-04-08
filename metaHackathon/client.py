"""
VaccineColdChainClient — thin HTTP wrapper so remote callers can drive
the environment exactly like they would locally.

Usage:
    from client import VaccineColdChainClient
    from models import VaccineAction

    env = VaccineColdChainClient(base_url="http://localhost:8000")
    obs = env.reset(task="easy")

    action = VaccineAction(
        action_type="transfer_stock",
        source_node="warehouse",
        target_node="center_a",
        vial_count=80,
    )
    obs, reward, done = env.step(action)
    state = env.state()
"""
from __future__ import annotations
import requests
from models import VaccineAction, VaccineObservation, VaccineState


class VaccineColdChainClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def reset(self, task: str = "easy") -> VaccineObservation:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=30,
        )
        r.raise_for_status()
        return VaccineObservation.model_validate(r.json())

    def step(self, action: VaccineAction) -> tuple[VaccineObservation, float, bool]:
        r = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        obs = VaccineObservation.model_validate(data["observation"])
        return obs, data["reward"], data["done"]

    def state(self) -> VaccineState:
        r = requests.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return VaccineState.model_validate(r.json())
