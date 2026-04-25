"""Root-level convenience launcher — delegates to ``server.app``.

Run the server directly::

    python app.py

or via uvicorn::

    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
