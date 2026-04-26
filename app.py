"""Main StratArena launcher for the Arena UI dashboard."""

from __future__ import annotations

import os

from server.dashboard_api import app


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8001")))
    print(f"[StratArena] Arena UI ready at {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
