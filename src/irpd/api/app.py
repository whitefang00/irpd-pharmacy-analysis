"""FastAPI routes. See `docs/api.md` for the contract."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from irpd import config, holdout
from irpd.api import chat, sealed, seen

CONTEST = Path(os.environ.get("IRPD_CONTEST", config.ROOT / "data/contest"))


class Turn(BaseModel):
    question: str
    answer: str


class ChatIn(BaseModel):
    question: str = chat.DEMO_QUESTION
    history: list[Turn] = []


def create_app(agent: Path = holdout.AGENT, sealed_file: Path = holdout.SEALED,
               chat_client: Any = None, contest_dir: Path = CONTEST) -> FastAPI:
    """The app. `chat_client` stands in for the OpenAI client in tests."""
    app = FastAPI(title="IRPD pharmacy analyst", version="0.1")

    def agent_con():
        # The agent's own read path: the agent file as of the cutoff, locked.
        return holdout.agent_view(agent=agent)

    @lru_cache(maxsize=1)
    def overview_cached() -> dict:
        with agent_con() as con:
            return seen.overview(con)

    @app.get("/overview")
    def overview() -> dict:
        return overview_cached()

    @app.get("/drugs")
    def drugs(q: str = "", limit: int = 20) -> list[dict]:
        with agent_con() as con:
            return seen.search(con, q.strip(), min(max(limit, 1), 100))

    @app.get("/drugs/{code}")
    def drug(code: str, holdout: bool = False) -> dict:
        with agent_con() as con:
            page = seen.drug(con, code)
        if page is None:
            raise HTTPException(404, f"no drug {code!r}")
        # Display only: the agent never receives this part.
        page["holdout"] = sealed.drug_actuals(code, sealed_file) if holdout else None
        if holdout:
            page["holdout"]["h2"] = sealed.h2_scores(agent, sealed_file).get(code)
        return page

    @lru_cache(maxsize=1)
    def forecasts_cached(stamp: int) -> list[dict]:
        with agent_con() as con:
            return seen.forecast_list(con)

    @app.get("/forecasts")
    def forecasts(holdout: bool = False) -> dict:
        """Every drug with both forecasts; with `holdout`, each one's H2 2025 score."""
        from irpd.ml import serve

        f = serve.FORECAST_FILE
        rows = forecasts_cached(f.stat().st_mtime_ns if f.exists() else 0)
        if holdout:  # display only: the agent never receives this part
            scores = sealed.h2_scores(agent, sealed_file)
            rows = [{**r, "h2": scores.get(r["code"])} for r in rows]
        return {"as_of": config.CUTOFF.isoformat(), "drugs": rows}

    @app.get("/forecasts/charts")
    def forecast_charts(codes: str, holdout: bool = False) -> list[dict]:
        """Small-chart data for up to 20 drugs, in the order given."""
        out = []
        for code in [c for c in codes.split(",") if c][:20]:
            with agent_con() as con:
                c = seen.chart(con, code)
            if c is None:
                continue
            if holdout:
                h = sealed.drug_actuals(code, sealed_file)
                c["holdout"] = {k: h[k] for k in ("sales", "stockouts", "rolling")}
            else:
                c["holdout"] = None
            out.append(c)
        return out

    @app.get("/replay")
    def replay() -> dict:
        return sealed.replay_report(agent, sealed_file)

    @lru_cache(maxsize=1)
    def contest_cached(stamp: tuple) -> dict | None:
        return sealed.contest_report(contest_dir)

    @app.get("/contest")
    def contest() -> dict | None:
        # Re-read when a run is scored or re-frozen.
        stamp = tuple(sorted((p.parent.name, p.stat().st_mtime)
                             for p in contest_dir.glob("final-*/*.*")))
        return contest_cached(stamp)

    @app.post("/chat")
    def ask(request: Request, body: ChatIn | None = None, demo: bool = False):
        body = body or ChatIn()
        if demo:
            question, events = chat.DEMO_QUESTION, chat.demo()
        else:
            history = [(t.question, t.answer) for t in body.history]
            question = body.question
            events = chat.live(question, history, agent=agent, client=chat_client)
        if "text/event-stream" in request.headers.get("accept", ""):
            return StreamingResponse(chat.sse(events), media_type="text/event-stream")
        if demo:
            return chat.transcript()
        t = chat.fold(events, question)
        if "error" in t:
            raise HTTPException(429 if t["error"]["code"] == "busy" else 503,
                                t["error"]["message"])
        return t

    return app


app = create_app()
