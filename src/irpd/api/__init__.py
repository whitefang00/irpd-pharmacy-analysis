"""Read-only HTTP API for the web app. The contract is in `docs/api.md`.

- `seen`: what the agent saw, through `holdout.agent_view()` (data to the cutoff).
- `sealed`: H2 2025 actuals and the replay, for display only. The only module
  here that opens the sealed holdout.
- `chat`: the `/chat` stub and the transcript shape.
- `app`: the FastAPI routes. Run with `uv run uvicorn irpd.api.app:app`.
"""
