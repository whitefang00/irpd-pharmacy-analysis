import { useMemo, useState } from "react";
import type { ForecastListRow } from "../api";
import { api, day, num, rp } from "../api";
import { ForecastChart } from "../charts/ForecastChart";
import { Seg } from "./Contest";
import { H2Sentence } from "./Drug";
import { useLoad } from "./useLoad";

const PAGE = 10;
const HISTORY_WEEKS = 26;
const TIERS = { all: "All", fast: "Fast", regular: "Regular", slow: "Slow", rare: "Rare" };
const SORTS = { revenue: "Revenue", gap: "Biggest disagreement" };
const H2_SORTS = { ...SORTS, ml_worse: "ML did worse", ml_better: "ML did better" };
type Sort = keyof typeof H2_SORTS;

const gap = (r: ForecastListRow) =>
  r.total_ml == null || r.total_current == null || r.total_current <= 0
    ? null : r.total_ml / r.total_current - 1;
// ML's extra miss in H2, as a share of what sold per window (negative: ML was better).
const mlEdge = (r: ForecastListRow) =>
  r.h2?.miss_ml == null || !r.h2.avg_sold ? null : (r.h2.miss_ml - r.h2.miss_current) / r.h2.avg_sold;

/** Every drug's forecasts in a table, and the current ten as stacked charts. */
export function ForecastsPage() {
  const [holdout, setHoldout] = useState(false);
  const [tier, setTier] = useState<keyof typeof TIERS>("all");
  const [sort, setSort] = useState<Sort>("revenue");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const list = useLoad(() => api.forecasts(holdout), `list:${holdout}`);
  const sortBy: Sort = !holdout && (sort === "ml_worse" || sort === "ml_better") ? "revenue" : sort;

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    let r = (list.data?.drugs ?? []).filter((d) =>
      (tier === "all" || d.tier === tier) &&
      (!needle || d.name.toLowerCase().includes(needle) || d.code.toLowerCase().startsWith(needle)));
    const by = (f: (d: ForecastListRow) => number | null) =>
      [...r].sort((a, b) => (f(b) ?? -Infinity) - (f(a) ?? -Infinity));
    if (sortBy === "gap") r = by((d) => { const g = gap(d); return g == null ? null : Math.abs(g); });
    if (sortBy === "ml_worse") r = by(mlEdge);
    if (sortBy === "ml_better") r = by((d) => { const e = mlEdge(d); return e == null ? null : -e; });
    return r;
  }, [list.data, tier, sortBy, q]);

  const start = Math.min(offset, Math.max(0, rows.length - 1) - (Math.max(0, rows.length - 1) % PAGE));
  const shown = rows.slice(start, start + PAGE);
  const codes = shown.map((r) => r.code);
  const charts = useLoad(() => (codes.length ? api.forecastCharts(codes, holdout) : Promise.resolve([])),
                         `charts:${codes.join(",")}:${holdout}`);
  const reset = <T,>(set: (v: T) => void) => (v: T) => { set(v); setOffset(0); };

  return (
    <div className="stack">
      <section className="stack" style={{ gap: 6 }}>
        <h1>Forecasts</h1>
        <p className="lede">
          Both forecasters for every drug sold by {day(list.data?.as_of ?? "2025-06-30")}: next week's
          forecast and the next 4 weeks. Turn on the holdout to see how each did in H2 2025, when
          both forecast the next 4 weeks every week from the data up to then.
        </p>
        <p className="lede small">
          The forecasts are nearly flat on purpose. Outside Eid, a drug's week-to-week ups and
          downs are mostly random, so the best forecast is a good estimate of its normal level.
          Reorder plans cover the randomness with a safety buffer: they order up to the 95th
          percentile of demand, not the average.
        </p>
      </section>

      <section className="card stack" style={{ gap: 10 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <input type="search" placeholder="Filter by name or code" value={q} style={{ maxWidth: 280 }}
                 onChange={(e) => reset(setQ)(e.target.value)} aria-label="Filter drugs" />
          <div className="row">
            <Seg value={tier} set={reset(setTier)} labels={TIERS} name="Tier" />
            <label className="check">
              <input type="checkbox" checked={holdout} onChange={(e) => reset(setHoldout)(e.target.checked)} />
              Show H2 2025 (holdout)
            </label>
          </div>
        </div>
        <div className="row">
          <span className="small">Sort by</span>
          <Seg value={sortBy} set={reset(setSort)} labels={holdout ? H2_SORTS : SORTS} name="Sort" />
        </div>
        {list.error && <p className="error">{list.error}</p>}
        <div className="table-wrap" style={{ opacity: list.loading ? 0.6 : 1 }}>
          <table>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>Drug</th>
                <th>Revenue</th>
                <th>Next week<br /><span className="muted">current · ML</span></th>
                <th>Next 4 weeks<br /><span className="muted">current · ML</span></th>
                <th>ML vs current</th>
                {holdout && <th>H2 average miss<br /><span className="muted">current · ML</span></th>}
                {holdout && <th>Closer in H2<br /><span className="muted">current · ML</span></th>}
              </tr>
            </thead>
            <tbody>
              {shown.map((r, i) => {
                const g = gap(r);
                return (
                  <tr key={r.code}>
                    <td style={{ textAlign: "left" }} className="wrap">
                      <span className="muted">{start + i + 1}. </span>
                      <a href={`#/drug/${r.code}`}>{r.name}</a> <span className="small muted">{r.tier}</span>
                    </td>
                    <td>{rp(r.revenue)}</td>
                    <td>{pair(r.next_current, r.next_ml)}</td>
                    <td>{pair(r.total_current, r.total_ml)}</td>
                    <td>{g == null ? "–" : `${g > 0 ? "+" : ""}${(g * 100).toFixed(0)}%`}</td>
                    {holdout && <td>{r.h2 === undefined ? "…" : r.h2 ? pair(r.h2.miss_current, r.h2.miss_ml) : <span className="muted">no clean window</span>}</td>}
                    {holdout && <td>{r.h2?.ml_closer != null ? `${r.h2.current_closer} · ${r.h2.ml_closer}` : "–"}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <span className="small">
            {rows.length ? `${start + 1}–${start + shown.length} of ${num(rows.length)} drugs` : "No drugs match"}
            {holdout && " · H2 scores count only 4-week windows with no stock-out"}
          </span>
          <div className="row">
            <button className="linkish" disabled={start === 0} onClick={() => setOffset(Math.max(0, start - PAGE))}>← Previous 10</button>
            <button className="linkish" disabled={start + PAGE >= rows.length} onClick={() => setOffset(start + PAGE)}>Next 10 →</button>
          </div>
        </div>
      </section>

      <section className="stack" style={{ gap: 12 }}>
        <div className="legend" aria-hidden>
          <span><i className="key-box" style={{ background: "var(--bar)" }} />Units sold per week</span>
          <span><i className="key-line" style={{ background: "var(--s1)" }} />Current method</span>
          <span><i className="key-line" style={{ background: "var(--s2)" }} />ML model</span>
          {!holdout && <span><i className="key-line" style={{ background: "var(--s1)", opacity: 0.4 }} />High case (p90)</span>}
          {holdout && <span className="muted">After the cutoff: each week's forecast, remade from the data up to then</span>}
        </div>
        {charts.error && <p className="error">{charts.error}</p>}
        {(charts.data ?? []).map((c, i) => {
          const row = shown.find((r) => r.code === c.drug.code);
          return (
            <section key={c.drug.code} className="card stack" style={{ gap: 4, opacity: charts.loading ? 0.6 : 1 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3 style={{ margin: 0 }}>
                  <span className="muted">{start + i + 1}. </span>
                  <a href={`#/drug/${c.drug.code}`}>{c.drug.name}</a>
                </h3>
                <span className="small">{c.drug.tier} · {rp(c.drug.revenue)} · next week {pair(row?.next_current ?? null, row?.next_ml ?? null)}</span>
              </div>
              <ForecastChart sales={[...c.sales, ...(c.holdout?.sales ?? [])]}
                             stockouts={[...c.stockouts, ...(c.holdout?.stockouts ?? [])]}
                             current={c.current} ml={c.ml} historyWeeks={HISTORY_WEEKS}
                             showActuals={holdout} rolling={c.holdout?.rolling} height={130} compact />
              {holdout && <p className="small" style={{ margin: 0 }}><H2Sentence s={row?.h2 ?? null} /></p>}
            </section>
          );
        })}
      </section>
    </div>
  );
}

function pair(a: number | null, b: number | null) {
  const f = (v: number | null) => (v == null ? "–" : num(v));
  return <>{f(a)} <span className="muted">·</span> {f(b)}</>;
}
