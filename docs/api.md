# Web API

Read-only JSON over HTTP, served by `irpd.api.app` (FastAPI). Start it with
`uv run uvicorn irpd.api.app:app --port 8000`. Interactive docs are at
`/docs`.

The code keeps two sides apart. What the agent saw (`irpd.api.seen`) goes
through `holdout.agent_view()` on every query, so it shows data up to the
cutoff (30 Jun 2025) and nothing after. The holdout (`irpd.api.sealed`) is
the H2 2025 actuals and the replay, read from the sealed file. It is for
people looking at the app, never for the agent, and it appears only in
`/replay`, `/contest` and the `holdout` field of a drug page.

Dates are `YYYY-MM-DD`. Money is rupiah at selling price (there are no costs).
Stock and units are in the ledger's base unit (tablets, bottles, ...).

## `GET /overview`

Headline numbers up to the cutoff.

```
as_of, period {first, last}
revenue_total, revenue_by_year [{year, revenue, drugs_sold}]
drugs {listed, sold, fast_or_regular}
tiers [{tier, drugs, drug_days, stockout_days, shelfout_days,
        stockout_share, shelfout_share}]      # inside each drug's selling window, from Oct 2023
revenue_share_mapped                          # share of revenue with an ingredient mapping
```

## `GET /drugs?q=&limit=20`

Drugs whose name contains `q` (or whose code starts with it), by revenue.

```
[{code, name, category, tier, revenue, ingredients, strength, form, kind, mapped}]
```

`tier` is fast / regular / slow / rare at the cutoff, or `none`.
`ingredients` is a ` + `-joined list, or null when unmapped.

## `GET /drugs/{code}?holdout=false`

The drug page. 404 if the code is unknown.

```
drug          the search row above
as_of         the cutoff
stock_start   stock before this day is unreliable (opening errors)
stock         [{date, floor, store, total}]       end of day
sales         [{date, units, revenue}]            days with sales only
stockouts     [{start, stop, days, ended_by}]     total <= 0
shelfouts     [{start, stop, days}]               floor <= 0 while the store has stock
purchases     [{date, qty, supplier, location}]
prices        [{unit, price, valid_from, valid_to}]
alternatives  [{code, name, tier, revenue, strength, form,
                same_strength_form, stock_at_cutoff}]   same ingredient set
holdout       null, or with holdout=true:
              {stock, sales, purchases, stockouts, shelfouts} for 1 Jul - 31 Dec 2025
```

## `GET /replay`

The H2 2025 replay (`irpd replay`): the pharmacist's real ledger, the mimic
and textbook reference policies, and each scaled to the pharmacist's average
stock value. Computed once (about 2 s), then cached.

```
period {first, last}, scope, fair_contest (one-paragraph framing)
matched_factor {policy: k or null}           # the s and S multiplier at matched stock
summary [{policy, kind: ledger|reference|matched, drugs, days,
          stockout_days, stockout_days_long, shelfout_days, stockout_share, shelfout_share,
          lost_units_stockout, lost_units_shelfout,
          lost_revenue_stockout, lost_revenue_shelfout, lost_revenue,
          avg_stock_value, units_sold}]
tradeoff {policy: [{factor, avg_stock_value, stockout_days, shelfout_days, lost_revenue}]}
per_drug [{code, name, "<policy>": {stockout_days, shelfout_days, lost_revenue,
                                    avg_stock_value}}]
          # for the pharmacist and each matched policy
```

## `GET /contest`

The agent vs pharmacist result: the agent's frozen policies (`irpd contest plan`) and
their holdout scores (`irpd contest score`), read from the run folders in
`IRPD_CONTEST` (default `data/contest/`). A run counts once it is named
`final-*` and has a `holdout_score.csv`. Returns `null` when there is none.
Nothing is recomputed: the scores are the ones logged when each run was scored.

```
drugs                                  # drugs scored
holdout_runs                           # lines in holdout_runs.jsonl (every scoring)
table {uncensored|recorded: [{policy, stockout_days, shelfout_days, lost_revenue,
        avg_stock_value, stockout_days_long, scale}]}
      # pharmacist (ledger), mimic, textbook, plain tools, then "agent (<model>)" per run,
      # all at the pharmacist's average stock value; scale is the s and S multiplier
mimic_tradeoff {uncensored|recorded: [{factor, avg_stock_value, stockout_days,
                                       shelfout_days, lost_revenue}]}
runs [{run_id, model, as_of, frozen_at, usd, rationale,
       rules [{name, when, sets, drugs}]   # the knobs in plain words, first match wins,
                                           # drugs = candidates on the policy day
       score {uncensored|recorded: rows as in table, policy "agent" for this run}
       tradeoff {uncensored|recorded: [curve points as above]}
       in_sample [{window, rows [{policy, stockout_days, shelfout_days, lost_revenue,
                                  avg_stock_value}]}]}]   # the backtests the agent saw
     # best first, by lost revenue under uncensored demand
```

## `POST /chat`

Runs the agent loop (`irpd.agent.loop.run`) on one question at the cutoff.
Body (optional): `{question, history?: [{question, answer}]}`. The last three
history pairs are folded into the input as plain text.

The model is `IRPD_CHAT_MODEL` (default `gpt-6-luna`). Every model call goes
through the spend ledger at `IRPD_LEDGER` (default `data/spend.jsonl`) and
its cap. One chat runs at a time per process.

`?demo=1` plays back a recorded answer and calls no model, so the page works
without a key. It is gpt-6-luna answering one of the eval questions ("Paracetamol
500mg Nova is out..."), with its real tool calls and outputs. Long lists in the
outputs are cut to a few rows. The body is ignored.

With `Accept: text/event-stream` the run streams as server-sent events while
it happens:

```
tool_call    {id, name, args}                   a tool starts
tool_result  {id, name, output, truncated}      it returns; output over 4,000 chars is cut
                                                to a string for display (the agent sees all)
message      {text}                             text the model wrote between calls
answer       {text}                             the final answer
done         {model, as_of, stopped, usage {input_tokens, output_tokens, cost_usd}}
error        {code, message}                    ends the stream; code is no_key, busy,
                                                budget (spend cap) or failed. A failed
                                                run gets a plain message; the details
                                                go to the server log
```

Without that header it returns the whole transcript as JSON (an error is a
503, or 429 when busy):

```
question, as_of, model
steps [{type: "tool_call", id, name, args, output}
       | {type: "message", text}]
answer                      # final text; every number in it comes from a tool output
usage {input_tokens, output_tokens, cost_usd}
```
