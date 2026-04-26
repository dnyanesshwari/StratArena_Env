from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from models import StratArenaAction, StratArenaObservation
from openenv.core.env_server import create_app
from server.dashboard_api import include_dashboard_routes
from server.stratarena_environment import StratArenaEnvironment


app = create_app(
    StratArenaEnvironment,
    StratArenaAction,
    StratArenaObservation,
    env_name="stratarena",
)

include_dashboard_routes(app)

UI_PATH = Path(__file__).resolve().parent.parent / "arena_ui"


@app.get("/")
def root() -> FileResponse:
    index_file = UI_PATH / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs", status_code=307)


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
