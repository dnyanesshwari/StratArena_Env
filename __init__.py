"""StratArena — OpenEnv-compatible multi-agent RL environment.

Quick start::

    from client import StratArenaEnv           # HTTP client
    from models import StratArenaAction, StratArenaObservation
    from inference import heuristic_allocation, run_episode
"""
from client import StratArenaEnv
from models import StratArenaAction, StratArenaObservation, StratArenaState

__all__ = ["StratArenaEnv", "StratArenaAction", "StratArenaObservation", "StratArenaState"]

