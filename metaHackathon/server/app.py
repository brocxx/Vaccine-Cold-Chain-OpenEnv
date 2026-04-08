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
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel

from models import VaccineAction, VaccineObservation, VaccineState
from server.coldchain_env import VaccineColdChainEnv


app = FastAPI(title="Vaccine Cold Chain OpenEnv", version="1.0.0")

# Single shared env instance (stateful per session)
_env: VaccineColdChainEnv = None


class ResetRequest(BaseModel):
    task: str = "easy"


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Vaccine Cold Chain OpenEnv</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
                body { 
                    font-family: 'Inter', sans-serif; 
                    line-height: 1.6; 
                    max-width: 900px; 
                    margin: 0 auto; 
                    padding: 40px 20px; 
                    background: #0f172a; 
                    color: #f8fafc; 
                }
                .card { 
                    background: #1e293b; 
                    padding: 40px; 
                    border-radius: 24px; 
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                    border: 1px solid #334155;
                }
                h1 { color: #38bdf8; margin-top: 0; font-size: 2.5rem; letter-spacing: -0.025em; }
                p { font-size: 1.125rem; color: #94a3b8; }
                .status-badge { 
                    display: inline-flex; 
                    align-items: center; 
                    padding: 6px 16px; 
                    border-radius: 9999px; 
                    background: #064e3b; 
                    color: #34d399; 
                    font-weight: 600; 
                    font-size: 0.875rem;
                    margin-bottom: 24px;
                }
                .pulse {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: #34d399;
                    border-radius: 50%;
                    margin-right: 8px;
                    box-shadow: 0 0 0 rgba(52, 211, 153, 0.4);
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
                    70% { box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
                }
                .endpoints { margin-top: 40px; }
                .endpoint-item {
                    background: #0f172a;
                    padding: 16px;
                    border-radius: 12px;
                    margin-bottom: 12px;
                    border: 1px solid #334155;
                    display: flex;
                    align-items: center;
                }
                .method {
                    font-weight: 700;
                    font-size: 0.75rem;
                    padding: 4px 8px;
                    border-radius: 6px;
                    margin-right: 16px;
                    min-width: 60px;
                    text-align: center;
                }
                .get { background: #0c4a6e; color: #38bdf8; }
                .post { background: #5b21b6; color: #a78bfa; }
                .path { font-family: monospace; font-size: 1rem; color: #e2e8f0; }
                .desc { margin-left: auto; color: #64748b; font-size: 0.875rem; }
                .footer { margin-top: 40px; text-align: center; color: #475569; font-size: 0.875rem; }
                a { color: #38bdf8; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="card">
                <div class="status-badge"><span class="pulse"></span> OpenEnv Instance Active</div>
                <h1>💉 Vaccine Cold Chain</h1>
                <p>A high-fidelity simulation engine for evaluating AI agents on vaccine logistics under uncertainty, infrastructure failure, and time constraints.</p>
                
                <div class="endpoints">
                    <div class="endpoint-item">
                        <span class="method get">GET</span>
                        <span class="path">/health</span>
                        <span class="desc">Liveness & system check</span>
                    </div>
                    <div class="endpoint-item">
                        <span class="method get">GET</span>
                        <span class="path">/openenv.yaml</span>
                        <span class="desc">Environment metadata spec</span>
                    </div>
                    <div class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="path">/reset</span>
                        <span class="desc">Initialize new episode</span>
                    </div>
                    <div class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="path">/step</span>
                        <span class="desc">Submit agent action</span>
                    </div>
                </div>
                
                <div class="footer">
                    Explore API documentation at <a href="/docs">/docs</a> &bull; Powered by FastAPI
                </div>
            </div>
        </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "healthy", "env": "vaccine_cold_chain", "version": "1.0.0"}


@app.post("/reset", response_model=VaccineObservation)
def reset(req: ResetRequest | None = None):
    global _env
    if req is None:
        req = ResetRequest(task="easy")
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
