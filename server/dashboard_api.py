"""Minimal dashboard API - TRUTHFUL representation of StratArena mechanics."""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import time
from typing import Any

from server.stratarena_environment import StratArenaEnvironment
from models import StratArenaAction
from inference import heuristic_allocation, AdaptiveStrategyController, StepTrace

app = FastAPI(title="StratArena Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI files
UI_PATH = Path(__file__).parent.parent / "arena_ui"
if UI_PATH.exists():
    app.mount("/static", StaticFiles(directory=UI_PATH), name="static")


# ─────────────────────────────────────────────────────────────
# Session state (one per episode)
# ─────────────────────────────────────────────────────────────

class EpisodeSession:
    """Tracks one running episode."""
    
    def __init__(self, episode_id: str, task: str, seed: int | None = None):
        self.episode_id = episode_id
        self.task = task
        self.seed = seed
        self.env = StratArenaEnvironment()
        self.obs = self.env.reset(task=task, seed=seed)
        self.controller = AdaptiveStrategyController(task)
        
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self.prev_bids = {"aggressive": 0.0, "conservative": 0.0, "me": 0.0}
        self.is_done = False
        self.last_reward = 0.0
        
    def step_one(self) -> dict[str, Any]:
        """Execute one step and return UI-ready data."""
        if self.is_done:
            raise ValueError("Episode already done")
        
        # 1. Update strategy controller with last reward (bandit learning)
        current_strategy = self.controller.update(self.obs, self.history, self.step_count, self.last_reward)
        
        # 2. Get smart agent action (heuristic + adaptive strategy)
        allocation, reason = heuristic_allocation(self.obs, self.history, current_strategy)
        
        # 3. Step environment
        self.obs = self.env.step(StratArenaAction(allocation=allocation))
        self.step_count += 1
        
        # 4. Extract reward for next iteration
        self.last_reward = float(self.env.last_reward.reward) if self.env.last_reward else 0.0
        
        # 5. Extract raw step info (TRUTHFUL - no fake data)
        last_info = self.env.last_info
        winner = last_info.winner if last_info else "none"
        
        # 6. Build ToM summary (what agent believes about opponents)
        tom_summary = self.env.tom_tracker.summary()
        tom_features = self.env.tom_tracker.get_features()
        
        # 7. Compute deltas for animation
        deltas = {
            "aggressive": last_info.aggressive_bid - self.prev_bids["aggressive"] if last_info else 0,
            "conservative": last_info.conservative_bid - self.prev_bids["conservative"] if last_info else 0,
            "me": last_info.my_bid - self.prev_bids["me"] if last_info else 0,
        }
        
        # 8. Get opponent signals (actual observed behavior)
        opp_signals = self.env.get_opponent_signals()
        
        # 9. Summary metrics (what matters for grading)
        metrics = self.env.summary_metrics()
        
        # 10. Current phase of budget
        budget_ratio = self.obs.my_budget_ratio
        phase = "late" if budget_ratio < 0.28 else "mid" if budget_ratio < 0.68 else "early"
        
        # 11. Regime shift active?
        regime_shift_step = self.env.task.regime_shift_step
        regime_active = regime_shift_step is not None and self.step_count >= regime_shift_step
        
        # 12. Build UI round data - ONLY TRUE DATA
        round_data = {
            "step": self.step_count,
            "max_steps": self.env.max_steps,
            "done": self.is_done,
            
            # Auction results (ACTUAL)
            "winner": winner,
            "agg_bid": last_info.aggressive_bid if last_info else 0,
            "con_bid": last_info.conservative_bid if last_info else 0,
            "sm_bid": last_info.my_bid if last_info else 0,
            "resource_value": last_info.resource_value if last_info else 0,
            "market_pressure": last_info.market_pressure if last_info else 0,
            "scarcity": last_info.scarcity if last_info else 0,
            "exploit_opportunity": last_info.exploit_opportunity if last_info else False,
            
            # Agent states
            "agg_budget_remaining": self.env.aggressive.budget,
            "con_budget_remaining": self.env.conservative.budget,
            "sm_budget_remaining": self.obs.my_budget,
            
            "agg_wins": self.env.aggressive.wins,
            "con_wins": self.env.conservative.wins,
            "sm_wins": self.env.my_wins,
            
            # ToM beliefs (what smart agent thinks about opponents)
            "tom_beliefs": tom_summary,
            "tom_features": tom_features,
            "exploit_signal": self.env.tom_tracker.get_exploit_signal(),
            "uncertainty_signal": self.env.tom_tracker.get_uncertainty_signal(),
            
            # Opponent signals (what we observed from their behavior)
            "opponent_signals": opp_signals,
            
            # Reward breakdown (training signal - key for RL)
            "reward": self.env.last_reward.reward if self.env.last_reward else 0,
            "reward_breakdown": {
                "value": self.env.last_reward.value_component,
                "efficiency": self.env.last_reward.efficiency_component,
                "strategy": self.env.last_reward.strategy_component,
                "penalty": self.env.last_reward.penalty_component,
            },
            
            # Strategy & adaptation (core RL mechanism)
            "strategy": current_strategy,
            "strategy_transitions": [
                {
                    "step": t.step,
                    "from": t.from_mode,
                    "to": t.to_mode,
                    "trigger": t.trigger,
                }
                for t in self.controller.transitions[-5:]
            ],
            "bandit_q": self.controller.bandit.values(),
            
            # Deltas for animation
            "deltas": deltas,
            
            # Task info
            "phase": phase,
            "regime_active": regime_active,
            "regime_shift_step": regime_shift_step,
            
            # Running metrics
            "metrics": metrics,
        }
        
        # Update state
        self.prev_bids = {
            "aggressive": last_info.aggressive_bid if last_info else 0,
            "conservative": last_info.conservative_bid if last_info else 0,
            "me": last_info.my_bid if last_info else 0,
        }
        self.history.append(round_data)
        self.is_done = self.obs.done
        
        return round_data
    
    def get_final_score(self) -> float:
        """Grade the episode (task-specific)."""
        return self.env.grade()


# Global sessions
_sessions: dict[str, EpisodeSession] = {}


@app.get("/")
async def serve_ui():
    """Serve main dashboard HTML."""
    ui_file = UI_PATH / "index.html"
    if ui_file.exists():
        return FileResponse(ui_file)
    return {"error": "UI not found"}


@app.post("/api/episode/start")
async def start_episode(task: str = "medium", seed: int | None = None):
    """Start a new episode."""
    episode_id = f"{task}_{int(time.time()*1000)}"
    try:
        session = EpisodeSession(episode_id, task, seed)
        _sessions[episode_id] = session
        
        return {
            "success": True,
            "episode_id": episode_id,
            "task": task,
            "max_steps": session.env.max_steps,
            "initial_budget": session.env.initial_budget,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/episode/{episode_id}/step")
async def step_episode(episode_id: str):
    """Execute one step."""
    if episode_id not in _sessions:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    session = _sessions[episode_id]
    if session.is_done:
        raise HTTPException(status_code=400, detail="Episode already done")
    
    try:
        round_data = session.step_one()
        return {
            "success": True,
            "round": round_data,
            "done": session.is_done,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/episode/{episode_id}/reset")
async def reset_episode(episode_id: str):
    """Reset episode."""
    if episode_id in _sessions:
        del _sessions[episode_id]
    return {"status": "reset"}


@app.get("/api/episode/{episode_id}/summary")
async def get_episode_summary(episode_id: str):
    """Get final episode summary."""
    if episode_id not in _sessions:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    session = _sessions[episode_id]
    if not session.is_done:
        raise HTTPException(status_code=400, detail="Episode not done")
    
    score = session.get_final_score()
    return {
        "episode_id": episode_id,
        "task": session.task,
        "score": score,
        "total_steps": session.step_count,
        "history_length": len(session.history),
        "metrics": session.env.summary_metrics(),
    }


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("DASHBOARD_PORT", "8001"))
    print(f"[Dashboard] Starting on http://localhost:{port}")
    uvicorn.run("server.dashboard_api:app", host="0.0.0.0", port=port, reload=False)
