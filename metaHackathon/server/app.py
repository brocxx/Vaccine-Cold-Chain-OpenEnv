"""
FastAPI server wrapping VaccineColdChainEnv.

Endpoints:
  GET  /health
  GET  /openenv.yaml
  POST /reset          body: {"task": "easy"|"medium"|"hard"}
  POST /step           body: VaccineAction JSON
  GET  /state
  POST /validate       (lightweight self-check for openenv validate)
"""
import sys
import os

# Allow imports from project root (where models.py lives)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from models import VaccineAction, VaccineObservation, VaccineState
from server.coldchain_env import VaccineColdChainEnv


app = FastAPI(title="Vaccine Cold Chain OpenEnv", version="1.0.0")

# Single shared env instance (stateful per session)
_env: VaccineColdChainEnv = None


class ResetRequest(BaseModel):
    task: str = "easy"


@app.get("/health")
def health():
    return {"status": "healthy", "env": "vaccine_cold_chain", "version": "1.0.0"}


@app.post("/reset", response_model=VaccineObservation)
def reset(req: ResetRequest):
    global _env
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"task must be one of: easy, medium, hard")
    _env = VaccineColdChainEnv(task=req.task)
    obs = _env.reset()
    return obs


@app.post("/step")
def step(action: VaccineAction):
    if _env is None:
        raise HTTPException(400, "Call /reset first.")
    try:
        obs, reward, done = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": {}}


@app.get("/state", response_model=VaccineState)
def state():
    if _env is None:
        raise HTTPException(400, "Call /reset first.")
    return _env.state()


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openenv.yaml")
    with open(yaml_path) as f:
        return f.read()


def create_app() -> FastAPI:
    """Entry point used by uvicorn."""
    return app
