from __future__ import annotations

import os

import uvicorn
from fastapi.responses import RedirectResponse

from models import StratArenaAction, StratArenaObservation
from openenv.core.env_server import create_app
from server.stratarena_environment import StratArenaEnvironment


app = create_app(
    StratArenaEnvironment,
    StratArenaAction,
    StratArenaObservation,
    env_name="stratarena",
)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
