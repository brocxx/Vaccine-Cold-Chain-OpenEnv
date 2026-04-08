"""
VaccineColdChainEnv — core simulation engine.

All physics happen in this exact order every step (before the action):
  1. hour++
  2. generator failure trigger
  3. temperature update
  4. spoilage check (temperature)
  5. spoilage check (calendar expiry)
  6. flood trigger (Hard only)
  7. truck arrival
  8. sensor noise roll (Hard only)
  9. outreach auto-fire

Then the submitted action is executed.
"""
from __future__ import annotations

import random
import uuid
from copy import deepcopy
from typing import Dict, Optional, Tuple

from models import (
    ActionRecord,
    NodeObservation,
    TemperatureRecord,
    VaccineAction,
    VaccineObservation,
    VaccineState,
)


# ─── Task configurations ───────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "seed": 0,
        "max_steps": 48,
        "description": (
            "One obvious move, no math, no pressure. "
            "Transfer vials to center_a before its outreach at hour 24."
        ),
        "nodes": {
            "warehouse": {"vials": 200, "expiry_hour": 999, "generator_on": True, "fuel_pct": 1.0},
            "center_a":  {"vials": 0,   "expiry_hour": 999, "generator_on": True, "fuel_pct": 1.0},
            "center_b":  {"vials": 50,  "expiry_hour": 999, "generator_on": True, "fuel_pct": 1.0},
        },
        "outreach": {
            "center_a": {"hour": 24, "vials_needed": 80},
            "center_b": None,
        },
        "generator_failure": {},   # node -> failure_hour
        "expiry_hours": {},        # node -> expiry_hour (overrides default)
        "flood_hour": None,
        "truck_eta": None,
        "sensor_noise": False,
    },
    "medium": {
        "seed": 1,
        "max_steps": 48,
        "description": (
            "Generator crisis + timing constraint. Sequence matters. "
            "center_a generator fails at hour 6; outreach at center_b needs 80 vials by hour 24."
        ),
        "nodes": {
            "warehouse": {"vials": 100, "expiry_hour": 999, "generator_on": True, "fuel_pct": 1.0},
            "center_a":  {"vials": 150, "expiry_hour": 10,  "generator_on": True, "fuel_pct": 1.0},
            "center_b":  {"vials": 40,  "expiry_hour": 999, "generator_on": True, "fuel_pct": 1.0},
        },
        "outreach": {
            "center_a": None,
            "center_b": {"hour": 24, "vials_needed": 80},
        },
        "generator_failure": {"center_a": 6},
        "expiry_hours": {},
        "flood_hour": None,
        "truck_eta": None,
        "sensor_noise": False,
    },
    "hard": {
        "seed": 2,
        "max_steps": 72,
        "description": (
            "Flood + lying sensor + unknown truck + two simultaneous outreaches. "
            "Information seeking required. Road to center_a closes at hour 4."
        ),
        "nodes": {
            "warehouse": {"vials": 150, "expiry_hour": 60, "generator_on": True, "fuel_pct": 1.0},
            "center_a":  {"vials": 80,  "expiry_hour": 24, "generator_on": True, "fuel_pct": 1.0},
            "center_b":  {"vials": 80,  "expiry_hour": 48, "generator_on": True, "fuel_pct": 1.0},
        },
        "outreach": {
            "center_a": {"hour": 18, "vials_needed": 100},
            "center_b": {"hour": 18, "vials_needed": 100},
        },
        "generator_failure": {},
        "expiry_hours": {},
        "flood_hour": 4,
        "truck_eta": None,          # determined at reset() via rng
        "sensor_noise": True,
    },
}

NODE_NAMES = ["warehouse", "center_a", "center_b"]


class VaccineColdChainEnv:
    def __init__(self, task: str = "easy"):
        assert task in TASK_CONFIGS, f"task must be one of {list(TASK_CONFIGS)}"
        self._task = task
        self._cfg = TASK_CONFIGS[task]

        # All mutable state lives in these dicts (populated by reset())
        self._hour: int = 0
        self._vials: Dict[str, int] = {}
        self._generator_on: Dict[str, bool] = {}
        self._fuel_pct: Dict[str, float] = {}
        self._temperature: Dict[str, float] = {}        # real temperature
        self._reported_temp: Dict[str, float] = {}      # what AI sees
        self._sensor_lying: Dict[str, bool] = {}
        self._road_center_a_open: bool = True
        self._road_center_b_open: bool = True
        self._truck_eta: Optional[int] = None
        self._truck_arrived: bool = False
        self._truck_eta_known: bool = False
        self._truck_arriving_in: Optional[float] = None
        self._outreach: Dict[str, Optional[dict]] = {}  # node -> {hour, vials_needed, done, cancelled}
        self._done: bool = False
        self._last_action_analysis: str = "Episode just started."

        # Tracking
        self._episode_id: str = ""
        self._vials_at_start: int = 0
        self._vials_delivered: int = 0
        self._vials_spoiled: int = 0
        self._sessions_completed: int = 0
        self._sessions_total: int = 0
        self._temp_history: list = []
        self._action_history: list = []
        self._step_reward_acc: float = 0.0

        # Pending fuel requests on Hard (3hr delay)
        self._fuel_requests: Dict[str, int] = {}   # node -> hour_when_ready

        self._rng = random.Random()

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> VaccineObservation:
        cfg = self._cfg
        self._rng.seed(cfg["seed"])

        self._hour = 0
        self._done = False
        self._last_action_analysis = "Episode started. Observe and plan."
        self._step_reward_acc = 0.0
        self._episode_id = str(uuid.uuid4())
        self._temp_history = []
        self._action_history = []
        self._vials_delivered = 0
        self._vials_spoiled = 0
        self._sessions_completed = 0
        self._fuel_requests = {}
        self._truck_arrived = False
        self._truck_eta_known = False
        self._truck_arriving_in = None
        self._sensor_lying = {n: False for n in NODE_NAMES}

        # Node state
        self._vials = {n: cfg["nodes"][n]["vials"] for n in NODE_NAMES}
        self._generator_on = {n: cfg["nodes"][n]["generator_on"] for n in NODE_NAMES}
        self._fuel_pct = {n: float(cfg["nodes"][n]["fuel_pct"]) for n in NODE_NAMES}
        self._temperature = {n: 4.0 for n in NODE_NAMES}
        self._reported_temp = deepcopy(self._temperature)

        # Road state
        self._road_center_a_open = True
        self._road_center_b_open = True

        # Truck ETA (Hard only, hidden between 20–45)
        if self._task == "hard":
            self._truck_eta = self._rng.randint(20, 45)
        else:
            self._truck_eta = None

        # Outreach sessions
        self._outreach = {}
        self._sessions_total = 0
        for node, info in cfg["outreach"].items():
            if info is not None:
                self._outreach[node] = {
                    "hour": info["hour"],
                    "vials_needed": info["vials_needed"],
                    "done": False,
                    "cancelled": False,
                }
                self._sessions_total += 1
            else:
                self._outreach[node] = None

        # Expiry hours per node (from config or node default)
        self._expiry_hour: Dict[str, int] = {}
        for n in NODE_NAMES:
            self._expiry_hour[n] = cfg["nodes"][n]["expiry_hour"]

        # Vials at start for coverage calculation
        self._vials_at_start = sum(self._vials.values())

        return self._build_observation()

    def step(self, action: VaccineAction) -> Tuple[VaccineObservation, float, bool]:
        if self._done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        step_reward = 0.0

        # ── Physics (in order, before action) ────────────────────────────────

        # 1. Hour counter
        self._hour += 1

        # 2. Generator failure trigger
        for node, fail_hour in self._cfg["generator_failure"].items():
            if self._hour == fail_hour:
                self._generator_on[node] = False
                self._fuel_pct[node] = 0.0

        # Process pending fuel deliveries (Hard mode 3hr delay)
        for node in list(self._fuel_requests):
            if self._hour >= self._fuel_requests[node]:
                self._generator_on[node] = True
                self._fuel_pct[node] = 1.0
                del self._fuel_requests[node]

        # 3. Temperature update
        for node in NODE_NAMES:
            if self._generator_on[node]:
                self._temperature[node] = 4.0
            else:
                self._temperature[node] = min(self._temperature[node] + 2.0, 40.0)

        # 4. Spoilage check — temperature
        spoilage_events = 0
        for node in NODE_NAMES:
            if self._temperature[node] > 8.0 and self._vials[node] > 0:
                self._vials_spoiled += self._vials[node]
                self._vials[node] = 0
                spoilage_events += 1

        # 5. Spoilage check — calendar expiry
        for node in NODE_NAMES:
            if self._hour >= self._expiry_hour[node] and self._vials[node] > 0:
                self._vials_spoiled += self._vials[node]
                self._vials[node] = 0
                spoilage_events += 1

        # 6. Flood trigger (Hard only)
        if self._cfg["flood_hour"] is not None and self._hour == self._cfg["flood_hour"]:
            self._road_center_a_open = False

        # 7. Truck arrival
        if (
            self._truck_eta is not None
            and not self._truck_arrived
            and self._hour == self._truck_eta
        ):
            self._vials["warehouse"] += 300
            self._truck_arrived = True
            # Update known ETA to reflect arrival
            if self._truck_eta_known:
                self._truck_arriving_in = 0.0

        # Update truck_arriving_in if known
        if self._truck_eta_known and not self._truck_arrived:
            self._truck_arriving_in = max(0.0, float(self._truck_eta - self._hour))

        # 8. Sensor noise (Hard only, center_b, 30% per hour)
        if self._cfg["sensor_noise"]:
            lying = self._rng.random() < 0.30
            self._sensor_lying["center_b"] = lying
            if lying:
                # Report a falsely low temp
                self._reported_temp["center_b"] = round(self._rng.uniform(2.0, 6.0), 1)
            else:
                self._reported_temp["center_b"] = self._temperature["center_b"]
        for node in NODE_NAMES:
            if node != "center_b" or not self._cfg["sensor_noise"]:
                self._reported_temp[node] = self._temperature[node]

        # 9. Outreach auto-fire
        for node, session in self._outreach.items():
            if session is None or session["done"] or session["cancelled"]:
                continue
            if self._hour == session["hour"]:
                if self._vials[node] >= session["vials_needed"]:
                    self._vials[node] -= session["vials_needed"]
                    self._vials_delivered += session["vials_needed"]
                    session["done"] = True
                    self._sessions_completed += 1
                # If not enough vials, outreach is missed (session stays not done)

        # ── Step rewards ──────────────────────────────────────────────────────

        for node in NODE_NAMES:
            t = self._temperature[node]
            if 2.0 <= t <= 8.0:
                step_reward += 0.01
            if self._fuel_pct[node] == 0.0:
                step_reward -= 0.1

        step_reward -= 0.2 * spoilage_events

        # ── Record temperature snapshot ───────────────────────────────────────
        for node in NODE_NAMES:
            self._temp_history.append(
                TemperatureRecord(
                    hour=self._hour,
                    node=node,
                    real_temp_c=self._temperature[node],
                    reported_temp_c=self._reported_temp[node],
                    sensor_lying=self._sensor_lying[node],
                )
            )

        # ── Execute action ────────────────────────────────────────────────────
        action_result = self._execute_action(action)
        self._action_history.append(
            ActionRecord(hour=self._hour, action=action, result=action_result)
        )
        self._last_action_analysis = action_result

        # ── Check episode done ────────────────────────────────────────────────
        all_sessions_resolved = all(
            s is None or s["done"] or s["cancelled"]
            for s in self._outreach.values()
        )
        max_steps_hit = self._hour >= self._cfg["max_steps"]
        self._done = max_steps_hit or all_sessions_resolved

        terminal_reward = 0.0
        if self._done:
            terminal_reward = self._compute_terminal_reward()

        total_reward = step_reward + terminal_reward
        self._step_reward_acc += total_reward

        return self._build_observation(), total_reward, self._done

    def state(self) -> VaccineState:
        """Internal ground truth — for judges, not the agent."""
        sessions_missed = sum(
            1
            for s in self._outreach.values()
            if s is not None and not s["done"] and not s["cancelled"] and self._done
        )
        missed_penalty: Optional[float] = None
        waste_penalty: Optional[float] = None
        coverage_score: Optional[float] = None
        final_reward: Optional[float] = None
        if self._done:
            coverage_score = self._vials_delivered / max(1, self._vials_at_start)
            waste_penalty = 0.3 * (self._vials_spoiled / max(1, self._vials_at_start))
            missed_penalty = 0.5 * (sessions_missed / max(1, self._sessions_total))
            score = coverage_score - waste_penalty - missed_penalty
            final_reward = max(0.001, min(0.999, score))

        return VaccineState(
            episode_id=self._episode_id,
            task=self._task,
            seed=self._cfg["seed"],
            hour=self._hour,
            real_temperatures=dict(self._temperature),
            sensor_is_lying=dict(self._sensor_lying),
            vials_delivered=self._vials_delivered,
            vials_spoiled=self._vials_spoiled,
            vials_available_at_start=self._vials_at_start,
            outreach_sessions_completed=self._sessions_completed,
            outreach_sessions_total=self._sessions_total,
            temperature_history=list(self._temp_history),
            action_history=list(self._action_history),
            coverage_score=coverage_score,
            waste_penalty=waste_penalty,
            missed_penalty=missed_penalty,
            final_reward=final_reward,
        )

    # ─── Private helpers ───────────────────────────────────────────────────────

    def _execute_action(self, action: VaccineAction) -> str:
        t = action.action_type

        if t == "do_nothing":
            return "Agent chose to wait and observe. No world change."

        if t == "transfer_stock":
            src = action.source_node
            tgt = action.target_node
            count = action.vial_count
            if src is None or tgt is None or count is None:
                return "Action Failed: transfer_stock requires source_node, target_node, and vial_count."
            if src not in NODE_NAMES or tgt not in NODE_NAMES:
                return f"Action Failed: Unknown node name. Valid nodes: {NODE_NAMES}."
            if src == tgt:
                return "Action Failed: source_node and target_node cannot be the same."
            # Road checks
            if (src == "center_a" or tgt == "center_a") and not self._road_center_a_open:
                return (
                    "Action Failed: Road to center_a has been closed since hour 4 due to flooding. "
                    "No stock was moved."
                )
            if (src == "center_b" or tgt == "center_b") and not self._road_center_b_open:
                return "Action Failed: Road to center_b is closed. No stock was moved."
            actual_count = min(count, self._vials[src])
            self._vials[src] -= actual_count
            self._vials[tgt] += actual_count
            if actual_count < count:
                return (
                    f"Warning: {src} only had {actual_count} vials. "
                    f"Partial transfer completed. "
                    f"{src} now has {self._vials[src]} vials, {tgt} now has {self._vials[tgt]} vials."
                )
            return (
                f"Transferred {count} vials from {src} to {tgt}. "
                f"{src} now has {self._vials[src]} vials, {tgt} now has {self._vials[tgt]} vials."
            )

        if t == "request_fuel":
            node = action.node
            if node is None or node not in NODE_NAMES:
                return f"Action Failed: request_fuel requires a valid node. Valid nodes: {NODE_NAMES}."
            if self._task == "hard":
                ready_hour = self._hour + 3
                self._fuel_requests[node] = ready_hour
                return (
                    f"Fuel request submitted for {node}. "
                    f"Fuel will arrive in 3 hours (hour {ready_hour}). Generator still off until then."
                )
            else:
                # Instant refuel on Easy/Medium
                self._generator_on[node] = True
                self._fuel_pct[node] = 1.0
                return (
                    f"Generator at {node} refuelled. "
                    "Temperature will stabilise at 4°C next step."
                )

        if t == "cancel_outreach":
            node = action.target_node or action.node
            if node is None or node not in self._outreach:
                return f"Action Failed: cancel_outreach requires a valid node with an outreach session."
            session = self._outreach.get(node)
            if session is None:
                return f"Action Failed: No outreach session scheduled for {node}."
            if session["done"]:
                return f"Action Failed: Outreach at {node} has already fired. Cannot cancel."
            if session["cancelled"]:
                return f"Action Failed: Outreach at {node} is already cancelled."
            session["cancelled"] = True
            return f"Outreach session at {node} cancelled. No vials will be delivered there."

        if t == "check_truck_status":
            if self._task != "hard":
                return "No truck incoming on this task."
            if self._truck_arrived:
                return "Truck has already arrived and unloaded 300 vials at the warehouse."
            self._truck_eta_known = True
            hours_remaining = self._truck_eta - self._hour
            self._truck_arriving_in = float(hours_remaining)
            return (
                f"Truck ETA confirmed: arriving at warehouse in {hours_remaining} hours "
                f"(hour {self._truck_eta}). It carries 300 vials."
            )

        return f"Action Failed: Unknown action_type '{t}'."

    def _compute_terminal_reward(self) -> float:
        sessions_missed = sum(
            1
            for s in self._outreach.values()
            if s is not None and not s["done"] and not s["cancelled"]
        )
        coverage = self._vials_delivered / max(1, self._vials_at_start)
        waste = 0.3 * (self._vials_spoiled / max(1, self._vials_at_start))
        missed = 0.5 * (sessions_missed / max(1, self._sessions_total))
        score = coverage - waste - missed
        return max(0.001, min(0.999, score))

    def _get_outreach_vials_needed(self) -> int:
        """Return the vials_needed for the active outreach session(s)."""
        for node in NODE_NAMES:
            s = self._outreach.get(node)
            if s is not None and not s["done"] and not s["cancelled"]:
                return s["vials_needed"]
        return 0

    def _build_observation(self) -> VaccineObservation:
        nodes_obs = []
        for name in NODE_NAMES:
            # temperature alarm: real temp > 8 OR reported temp > 8
            alarm = self._reported_temp[name] > 8.0 or self._temperature[name] > 8.0
            # hours until expiry
            if self._vials[name] == 0:
                hue = 0.0
            else:
                hue = max(0.0, float(self._expiry_hour[name] - self._hour))

            nodes_obs.append(
                NodeObservation(
                    name=name,
                    vial_count=self._vials[name],
                    temperature_c=self._reported_temp[name],
                    temperature_alarm=alarm,
                    generator_fuel_pct=self._fuel_pct[name],
                    hours_until_expiry=hue,
                )
            )

        # Outreach countdowns
        def _outreach_hours(node: str) -> Optional[float]:
            s = self._outreach.get(node)
            if s is None or s["done"] or s["cancelled"]:
                return None
            return max(0.0, float(s["hour"] - self._hour))

        return VaccineObservation(
            nodes=nodes_obs,
            hour=self._hour,
            road_to_center_a_open=self._road_center_a_open,
            road_to_center_b_open=self._road_center_b_open,
            truck_eta_known=self._truck_eta_known,
            truck_arriving_in_hours=self._truck_arriving_in,
            outreach_center_a_in_hours=_outreach_hours("center_a"),
            outreach_center_b_in_hours=_outreach_hours("center_b"),
            outreach_vials_needed=self._get_outreach_vials_needed(),
            last_action_analysis=self._last_action_analysis,
            task_description=self._cfg["description"],
        )
