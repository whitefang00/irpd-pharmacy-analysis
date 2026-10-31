# Web app

React + Vite + TypeScript, no other runtime dependencies. Charts are
hand-rolled SVG. It reads the API in `src/irpd/api` (contract:
[docs/api.md](../docs/api.md)).

It has four pages.

- Overview: headline numbers up to the cutoff.
- Drug page: the stock line, stock-out and shelf-out spells, deliveries,
  sales, alternatives and prices.
- Agent vs pharmacist: the frozen contest runs from `/contest`, with the
  headline, the tradeoff chart, the six-policy table, the agent's rules and
  reasons, in-sample vs holdout, and the caveats. Without a scored run it
  falls back to the reference replay.
- Chat: the live agent. Tool calls stream in as cards you can expand. Without
  a key, an example transcript plays instead.

## Run it

Build the warehouse first (`uv run irpd build`), which also seals the H2 2025
holdout next to it. Then, from the repo root:

```sh
uv run uvicorn irpd.api.app:app --port 8000      # the API
cd web && npm install && npm run dev            # the app, on http://localhost:5173
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`. If the API runs elsewhere,
set `IRPD_API`, for example `IRPD_API=http://127.0.0.1:8077 npm run dev`. Set
`IRPD_DB` for the build and the API if the warehouse isn't at
`data/warehouse/irpd.duckdb` (the sealed files sit next to it), and
`IRPD_CONTEST` for the API if the contest runs aren't in `data/contest/`.

The chat calls the model when `OPENAI_API_KEY` is set (`IRPD_CHAT_MODEL`,
default `gpt-6-luna`; spend goes to `IRPD_LEDGER`, default `data/spend.jsonl`).

`npm run build` type-checks and writes `web/dist/` (git-ignored).
