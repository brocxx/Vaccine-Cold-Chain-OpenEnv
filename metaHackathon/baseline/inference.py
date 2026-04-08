"""
Baseline inference script for Vaccine Cold Chain OpenEnv.

Uses the OpenAI client (pointed at API_BASE_URL) to run a language model
against all three tasks and prints reproducible scores.

Required environment variables:
  API_BASE_URL   — The API endpoint for the LLM
  MODEL_NAME     — Model identifier (e.g. gpt-4o, claude-sonnet-4-20250514)
  HF_TOKEN       — Your Hugging Face / API key

Usage:
  python baseline/inference.py

Output format (stdout) — strictly follows [START], [STEP], [END] protocol:
  [START] {"task": "easy", "episode_id": "..."}
  [STEP]  {"hour": 1, "action": {...}, "reward": 0.03, "done": false}
  ...
  [END]   {"task": "easy", "final_reward": 0.85, "coverage": 0.90, ...}
"""

import json
import os
import sys
import requests
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# Optional – if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",
)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """
You are an expert vaccine cold chain manager. You must make decisions each hour
to ensure vaccines reach outreach sessions without spoiling.

You will receive the current environment state as JSON and must respond with
a single action in JSON format. No other text — just the JSON action.

Available action types:
- {"action_type": "transfer_stock", "source_node": "<node>", "target_node": "<node>", "vial_count": <int>}
- {"action_type": "request_fuel", "node": "<node>"}
- {"action_type": "cancel_outreach", "target_node": "<node>"}
- {"action_type": "check_truck_status"}
- {"action_type": "do_nothing"}

Valid node names: warehouse, center_a, center_b

KEY RULES:
1. Transfer vials to centers before their outreach sessions.
2. If a generator is off (fuel_pct = 0), vaccines will spoil in 2 hours. Refuel ASAP.
3. Vaccines expire at calendar hours shown in hours_until_expiry.
4. If temperature_alarm is True but generator_fuel_pct is 1.0 — it's likely a sensor lie. Don't panic.
5. On Hard tasks: road to center_a closes at hour 4. Move stock BEFORE then.
6. On Hard tasks: call check_truck_status early to plan around the incoming 300-vial truck.
7. Respond with ONLY the JSON action object. No explanation.
""".strip()


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_BASE_URL}{endpoint}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_action(obs: dict, task: str, history: list) -> dict:
    """Ask the LLM for the next action given the current observation."""
    user_content = f"Task: {task}\n\nCurrent state:\n{json.dumps(obs, indent=2)}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include last 3 turns for context (keep context window manageable)
    for turn in history[-3:]:
        messages.append({"role": "user", "content": turn["obs_str"]})
        messages.append({"role": "assistant", "content": turn["action_str"]})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        action_dict = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: do nothing
        action_dict = {"action_type": "do_nothing"}

    return action_dict, user_content, raw


def run_task(task: str) -> dict:
    # Reset env
    reset_resp = call_env("POST", "/reset", {"task": task})
    episode_id = "unknown"
    # Try to get episode_id from state
    try:
        state = call_env("GET", "/state")
        episode_id = state.get("episode_id", "unknown")
    except Exception:
        pass

    print(json.dumps({"_log": "START", "task": task, "episode_id": episode_id}))
    sys.stdout.write(f"[START] {{\"task\": \"{task}\", \"episode_id\": \"{episode_id}\"}}\n")
    sys.stdout.flush()

    obs = reset_resp
    history = []
    total_reward = 0.0
    step_num = 0

    while True:
        action_dict, obs_str, action_str = get_action(obs, task, history)
        history.append({"obs_str": obs_str, "action_str": action_str})

        step_resp = call_env("POST", "/step", action_dict)
        reward = step_resp["reward"]
        done = step_resp["done"]
        obs = step_resp["observation"]
        total_reward += reward
        step_num += 1

        log_line = {
            "hour": obs["hour"],
            "action": action_dict,
            "reward": round(reward, 4),
            "done": done,
        }
        sys.stdout.write(f"[STEP]  {json.dumps(log_line)}\n")
        sys.stdout.flush()

        if done:
            break

    # Get final state
    try:
        final_state = call_env("GET", "/state")
        coverage = final_state.get("coverage_score", 0.0)
        waste = final_state.get("waste_penalty", 0.0)
        missed = final_state.get("missed_penalty", 0.0)
        final_reward = final_state.get("final_reward", 0.0)
        delivered = final_state.get("vials_delivered", 0)
        spoiled = final_state.get("vials_spoiled", 0)
    except Exception:
        coverage = waste = missed = final_reward = 0.0
        delivered = spoiled = 0

    end_line = {
        "task": task,
        "total_steps": step_num,
        "total_reward": round(total_reward, 4),
        "final_reward": round(final_reward, 4),
        "coverage": round(coverage, 4),
        "waste_penalty": round(waste, 4),
        "missed_penalty": round(missed, 4),
        "vials_delivered": delivered,
        "vials_spoiled": spoiled,
    }
    sys.stdout.write(f"[END]   {json.dumps(end_line)}\n")
    sys.stdout.flush()

    return end_line


def main():
    print("=" * 60)
    print("Vaccine Cold Chain — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"ENV:   {ENV_BASE_URL}")
    print("=" * 60)

    results = {}
    for task in TASKS:
        print(f"\n{'─' * 40}")
        print(f"Running task: {task.upper()}")
        print(f"{'─' * 40}")
        result = run_task(task)
        results[task] = result

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for task, r in results.items():
        status = ""
        fr = r["final_reward"]
        if task == "easy":
            status = "✓ PASS" if fr >= 0.5 else "✗ FAIL (need ≥0.5)"
        elif task == "medium":
            status = "✓ PASS" if fr >= 0.2 else "✗ FAIL (need ≥0.2)"
        elif task == "hard":
            status = "✓ PASS" if fr >= 0.1 else "✗ FAIL (need ≥0.1)"
        print(f"  {task:8s}: final_reward={fr:.4f}  coverage={r['coverage']:.4f}  {status}")

    print("=" * 60)


if __name__ == "__main__":
    main()
