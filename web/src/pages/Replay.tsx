import { useState } from "react";
import type { Replay, ReplayRow } from "../api";
import { api, day, num, pct, rp } from "../api";
import { METRIC_LABEL, type Metric, type Series, TradeoffChart } from "../charts/TradeoffChart";
import { ContestView, Seg } from "./Contest";
import { useLoad } from "./useLoad";

const NAMES: Record<string, string> = {
  "pharmacist (actual)": "Pharmacist, real ledger",
  mimic: "Mimic of the pharmacist",
  textbook: "Textbook (s, S)",
  "mimic at matched stock": "Mimic, at matched stock",
  "textbook at matched stock": "Textbook, at matched stock",
};

export function ReplayPage() {
  const { data: c, error: cErr, loading: cLoading } = useLoad(api.contest, "contest");
  const { data: r, error } = useLoad(api.replay, "replay");
  const [metric, setMetric] = useState<Metric>("stockout_days");

  return (
    <div className="stack">
      <section className="stack" style={{ gap: 10 }}>
        <h1>Agent vs pharmacist</h1>
        <p className="lede">
          Each reorder policy is set from data up to 30 June 2025. It is then replayed day by day
          on the real demand of July–December 2025 and scored the same way as what the pharmacist
          did.{" "}
          {c ? "The agent froze its policy before the holdout was opened."
            : "The agent's policy shows up here once it has run. Until then, two reference policies are shown."}
        </p>
      </section>
      {(error || cErr) && <p className="error">{error ?? cErr}</p>}
      {c && <ContestView c={c} fairContest={r?.fair_contest} />}
      {!c && !cLoading && !r && !error && <p className="small muted">Running the replay (about 2 seconds the first time)…</p>}
      {!c && !cLoading && r && <Body r={r} metric={metric} setMetric={setMetric} />}
      {c && r && <PerDrug r={r} />}
    </div>
  );
}

function Body({ r, metric, setMetric }: { r: Replay; metric: Metric; setMetric: (m: Metric) => void }) {
  const ledger = r.summary.find((s) => s.kind === "ledger")!;
  const color = (policy: string) => (policy.startsWith("mimic") ? "var(--s1)" : "var(--s2)");
  const series: Series[] = [
    ...Object.entries(r.tradeoff).map(([name, curve]): Series => ({
      label: NAMES[name] ?? name, color: color(name), curve, legend: "line" })),
    { label: NAMES[ledger.policy] ?? ledger.policy, color: "var(--ink)", point: ledger, name: true, legend: "dot" },
    ...r.summary.filter((s) => s.kind !== "ledger").map((s): Series => ({
      label: NAMES[s.policy] ?? s.policy, color: color(s.policy), point: s, hollow: s.kind === "matched" })),
  ];
  return (
    <>
      <div className="note-box"><b>The fair contest.</b> {r.fair_contest}</div>

      <section className="card stack" style={{ gap: 8 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Stock held vs {METRIC_LABEL[metric].toLowerCase()}</h2>
          <Seg value={metric} set={setMetric} labels={METRIC_LABEL} name="Measure" />
        </div>
        <p className="small">
          Each curve scales one policy's reorder levels from 0.5× to 2×. Lower and to the left is
          better. Hollow points are policies scaled to the pharmacist's average stock
          ({rp(ledger.avg_stock_value)}).
        </p>
        <TradeoffChart series={series} metric={metric} stock={ledger.avg_stock_value} />
      </section>

      <section className="card stack" style={{ gap: 8 }}>
        <h2>Scores, {day(r.period.first)} – {day(r.period.last)}</h2>
        <p className="small">{num(ledger.drugs)} {r.scope}. Stock valued at selling price.</p>
        <SummaryTable rows={r.summary} />
        <p className="small muted">
          "Of which long" counts stock-out days in spells over 14 days. In the real ledger these
          look like drugs left unstocked or unavailable. The simulation can't produce them.
        </p>
      </section>

      <PerDrug r={r} />
    </>
  );
}

function SummaryTable({ rows }: { rows: ReplayRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>Policy</th><th>Avg stock</th><th>Stock-out days</th><th>of which long</th>
            <th>Share of days</th><th>Shelf-out days</th><th>Revenue lost</th></tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.policy} className={s.kind === "matched" ? "em" : ""}>
              <td>{NAMES[s.policy] ?? s.policy}</td>
              <td>{rp(s.avg_stock_value)}</td>
              <td>{num(s.stockout_days)}</td>
              <td>{num(s.stockout_days_long)}</td>
              <td>{pct(s.stockout_share)}</td>
              <td>{num(s.shelfout_days)}</td>
              <td>{rp(s.lost_revenue)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type Scores = { stockout_days: number; shelfout_days: number; lost_revenue: number; avg_stock_value: number };

function PerDrug({ r }: { r: Replay }) {
  const a = "mimic at matched stock", b = "textbook at matched stock";
  if (!r.per_drug.length || !(a in r.per_drug[0]) || !(b in r.per_drug[0])) return null;
  const rows = r.per_drug
    .map((d) => ({ ...d, A: d[a] as Scores, B: d[b] as Scores, P: d["pharmacist (actual)"] as Scores }))
    .map((d) => ({ ...d, gap: d.A.lost_revenue - d.B.lost_revenue }));
  const top = [...rows].sort((x, y) => Math.abs(y.gap) - Math.abs(x.gap)).slice(0, 10);
  return (
    <section className="card stack" style={{ gap: 8 }}>
      <h2>Where the two matched policies differ most</h2>
      <p className="small">
        Drugs with the biggest gap in revenue lost, mimic vs textbook at matched stock, on recorded sales.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Drug</th><th>Mimic: out days</th><th>lost</th><th>Textbook: out days</th><th>lost</th>
              <th>Real ledger: out days</th></tr>
          </thead>
          <tbody>
            {top.map((d) => (
              <tr key={d.code}>
                <td><a href={`#/drug/${d.code}`}>{d.name}</a></td>
                <td>{num(d.A.stockout_days)}</td><td>{rp(d.A.lost_revenue)}</td>
                <td>{num(d.B.stockout_days)}</td><td>{rp(d.B.lost_revenue)}</td>
                <td>{num(d.P.stockout_days)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
