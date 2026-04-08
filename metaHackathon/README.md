---
title: Vaccine Cold Chain OpenEnv
emoji: 💉
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
---

# 💉 Vaccine Cold Chain — OpenEnv Environment

An AI agent manages a **3-node vaccine cold chain** over up to 72 simulated hours.
Each hour the agent takes actions to ensure vaccines reach outreach sessions without spoiling.

This is a real-world planning environment with **time pressure**, **uncertainty**, and **multi-constraint decisions** — wrapped in an OpenEnv-compliant server.

---

## Motivation

Vaccine cold chain failures cause millions of preventable deaths annually. An RL-capable simulation of cold chain logistics can train agents to make better decisions under sensor uncertainty, supply chain surprises, and infrastructure failures — all conditions that occur in real deployments.

---

## Observation Space

Each step the agent receives a `VaccineObservation` with:

**Per node** (3 nodes: `warehouse`, `center_a`, `center_b`):
| Field | Type | Notes |
|---|---|---|
| `name` | str | Node identifier |
| `vial_count` | int | Vaccines currently at this node |
| `temperature_c` | float | **May be a lie** on Hard/center_b |
| `temperature_alarm` | bool | True if sensor flagged — could be real OR fake |
| `generator_fuel_pct` | float | 0.0–1.0, **always truthful** |
| `hours_until_expiry` | float | Hours until soonest batch expires |

**Global fields**:
| Field | Notes |
|---|---|
| `hour` | Current simulation hour |
| `road_to_center_a_open` | False from hour 4 on Hard (flood) |
| `truck_eta_known` | True only after `check_truck_status` action |
| `outreach_center_a_in_hours` | Hours until center_a outreach fires |
| `outreach_center_b_in_hours` | Hours until center_b outreach fires |
| `last_action_analysis` | Human-readable feedback on last action |

---

## Action Space

| action_type | Required Fields | Effect |
|---|---|---|
| `transfer_stock` | source_node, target_node, vial_count | Move vials between nodes |
| `request_fuel` | node | Refuel generator (instant Easy/Medium; 3hr delay Hard) |
| `cancel_outreach` | target_node | Cancel upcoming outreach session |
| `check_truck_status` | *(none)* | Hard only — reveals exact truck ETA |
| `do_nothing` | *(none)* | Wait and observe |

---

## Tasks

### Easy (seed=0, max 48 hours)
**One obvious move, no math, no pressure.**
- Warehouse: 200 vials | center_a: 0 vials | center_b: 50 vials
- Outreach at center_a, hour 24, needs 80 vials
- Correct play: transfer 80+ vials from warehouse to center_a
- Expected scores: Random ~15% | Any LLM ~90%+

### Medium (seed=1, max 48 hours)
**Generator crisis + timing constraint. Sequence matters.**
- center_a has 150 vials but generator FAILS at hour 6 → spoilage at hour 8
- Outreach at center_b needs 80 vials by hour 24
- Correct play: transfer ~80 to center_b AND request_fuel at center_a
- Expected scores: Random ~5% | GPT-4o-mini ~40% | GPT-4o ~65%

### Hard (seed=2, max 72 hours)
**Flood + lying sensor + unknown truck + two simultaneous outreaches.**
- Road to center_a CLOSES at hour 4 (only 3 hours to act)
- center_b sensor lies 30% of hours
- 300-vial truck arrives at warehouse (hidden ETA 20–45)
- Both centers need 100 vials each at hour 18
- Expected scores: Random ~2% | GPT-4o-mini ~15% | GPT-4o ~35–40%

---

## Reward Function

**Step rewards** (every hour):
- +0.01 per node staying 2–8°C
- −0.1 per node with empty generator
- −0.2 per spoilage event

**Terminal reward** (end of episode):
```
coverage = vials_delivered / vials_available_at_start
waste    = 0.3 × (vials_spoiled / vials_available_at_start)
missed   = 0.5 × (sessions_missed / total_sessions)
terminal = max(0.0, coverage - waste - missed)
```

---

## Setup & Usage

### Local (without Docker)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t vaccine-cold-chain .
docker run -p 8000:8000 vaccine-cold-chain
```

### Run baseline

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key_here
export ENV_BASE_URL=http://localhost:8000

python baseline/inference.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/reset` | Start new episode: `{"task": "easy"}` |
| POST | `/step` | Submit action: VaccineAction JSON |
| GET | `/state` | Internal ground truth (for judges) |
| GET | `/openenv.yaml` | Environment metadata |

---

## Baseline Scores (GPT-4o-mini)

| Task | final_reward | coverage | Notes |
|---|---|---|---|
| Easy | ≥ 0.70 | ≥ 0.80 | Single transfer, should be near-perfect |
| Medium | ≥ 0.30 | ≥ 0.40 | Requires generator reasoning |
| Hard | ≥ 0.10 | ≥ 0.20 | Requires proactive info-seeking |
