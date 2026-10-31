import { useState } from "react";
import type { Contest, ContestRow, ContestRun, Demand } from "../api";
import { num, rp } from "../api";
import { METRIC_LABEL, type Metric, type Series, TradeoffChart } from "../charts/TradeoffChart";

const LABEL: Record<string, string> = {
  "pharmacist (ledger)": "Pharmacist, real ledger",
  mimic: "Copy of the pharmacist (mimic)",
  textbook: "Textbook (s, S)",
  "plain tools": "Plain tools, no LLM",
};
const COLOR: Record<string, string> = {
  "pharmacist (ledger)": "var(--ink)", mimic: "var(--s1)", textbook: "var(--s2)", "plain tools": "var(--s5)",
};
const AGENT_COLORS = ["var(--s3)", "var(--s4)"];
const DEMAND: Record<Demand, string> = { uncensored: "Uncensored demand", recorded: "Recorded sales" };

const agentKey = (r: ContestRun) => `agent (${r.model})`;
const find = (rows: ContestRow[], policy: string) => rows.find((r) => r.policy === policy);
const label = (policy: string) => LABEL[policy] ?? policy.replace(/^agent/, "Agent");

/** A ratio in words: "about half", "about a third", ... */
function share(a: number, b: number): string {
  const r = a / b;
  const words: [number, string][] = [[0.25, "a quarter of"], [1 / 3, "a third of"], [0.5, "half"],
    [2 / 3, "two thirds of"], [0.75, "three quarters of"]];
  const [, w] = words.reduce((best, c) => (Math.abs(c[0] - r) < Math.abs(best[0] - r) ? c : best));
  return `about ${w}`;
}

/** How the agent's lost revenue compares with the textbook rule's. Within 10% counts as a tie. */
function vsTextbook(a: ContestRow, tb: ContestRow): "ties" | "beats" | "loses to" {
  const gap = (a.lost_revenue - tb.lost_revenue) / tb.lost_revenue;
  return Math.abs(gap) < 0.1 ? "ties" : gap < 0 ? "beats" : "loses to";
}

export function ContestView({ c, fairContest }: { c: Contest; fairContest?: string }) {
  const [metric, setMetric] = useState<Metric>("stockout_days");
  const [demand, setDemand] = useState<Demand>("uncensored");
  const [pick, setPick] = useState(0);
  const rows = c.table[demand];
  const color = (policy: string) =>
    COLOR[policy] ?? AGENT_COLORS[c.runs.findIndex((r) => agentKey(r) === policy)] ?? "var(--ink-2)";
  const ledger = find(rows, "pharmacist (ledger)");
  const stock = ledger?.avg_stock_value ?? rows[0]?.avg_stock_value ?? 0;

  const series: Series[] = [
    { label: label("mimic"), color: color("mimic"), curve: c.mimic_tradeoff[demand], legend: "line" },
    ...c.runs.map((r, i): Series => ({ label: label(agentKey(r)), color: AGENT_COLORS[i], curve: r.tradeoff[demand], legend: "line" })),
    { label: label("textbook"), color: color("textbook"), legend: "dot" },
    { label: label("plain tools"), color: color("plain tools"), legend: "dot" },
    ...(ledger ? [{ label: label(ledger.policy), color: color(ledger.policy), point: ledger, name: true,
      legend: "dot" as const }] : []),
    ...rows.filter((r) => r !== ledger).map((r) => ({ label: `${label(r.policy)}, at matched stock`,
      color: color(r.policy), point: r, hollow: true })),
  ];

  return (
    <>
      <Headline c={c} />
      {fairContest && <div className="note-box"><b>The fair contest.</b> {fairContest}</div>}

      <section className="card stack" style={{ gap: 8 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Stock held vs {METRIC_LABEL[metric].toLowerCase()}</h2>
          <div className="row" style={{ gap: 8 }}>
            <Seg value={metric} set={setMetric} labels={METRIC_LABEL} name="Measure" />
            <Seg value={demand} set={setDemand} labels={DEMAND} name="Demand" />
          </div>
        </div>
        <p className="small">
          Each curve scales one policy's reorder levels from 0.5× to 2×. Lower and to the left is
          better. Hollow points are each policy scaled to the pharmacist's average stock
          ({rp(stock)}). The textbook and plain-tool policies were scored only at that point.
        </p>
        <TradeoffChart series={series} metric={metric} stock={stock} />
      </section>

      <section className="card stack" style={{ gap: 8 }}>
        <h2>Scores, July – December 2025</h2>
        <p className="small">
          {num(c.drugs)} fast and regular drugs, every policy at the pharmacist's average stock value.{" "}
          {demand === "uncensored"
            ? "Demand includes an estimate of the sales lost while a drug was out."
            : "Demand is recorded sales only, so sales lost to stock-outs are missing."}
        </p>
        <ContestTable rows={rows} lead={agentKey(c.runs[0])} color={color} />
      </section>

      <section className="card stack" style={{ gap: 10 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>What the agent decided</h2>
          {c.runs.length > 1 && (
            <Seg value={pick} set={setPick} name="Run"
                 labels={Object.fromEntries(c.runs.map((r, i) => [i, r.model]))} />
          )}
        </div>
        <Decided run={c.runs[pick]} />
      </section>

      <section className="card stack" style={{ gap: 10 }}>
        <h2>In-sample vs holdout, {c.runs[pick].model}</h2>
        <InSample run={c.runs[pick]} holdoutRuns={c.holdout_runs} runs={c.runs.length} />
      </section>

      <section className="card stack" style={{ gap: 8 }}>
        <h2>Caveats</h2>
        <ul className="caveats">
          <li>The simulated shop floor has no shelf-space limit, so the shelf-out win is an upper bound.</li>
          <li>Deliveries always arrive on time in the simulation, so the real ledger looks much
            worse than any simulated policy. Compare with the copy of the pharmacist instead.</li>
          <li>One frozen run per model. Another run of the same model could land elsewhere.</li>
          <DataCaveats c={c} />
        </ul>
      </section>
    </>
  );
}

export function Seg<T extends string | number>({ value, set, labels, name }: {
  value: T; set: (v: T) => void; labels: Record<string, string>; name: string;
}) {
  return (
    <div className="seg" role="group" aria-label={name}>
      {Object.entries(labels).map(([k, l]) => {
        const v = (typeof value === "number" ? Number(k) : k) as T;
        return (
          <button key={k} className={v === value ? "on" : ""} aria-pressed={v === value}
                  onClick={() => set(v)}>{l}</button>
        );
      })}
    </div>
  );
}

function Headline({ c }: { c: Contest }) {
  const t = c.table.uncensored;
  const lead = c.runs[0];
  const a = find(t, agentKey(lead)), mi = find(t, "mimic"), tb = find(t, "textbook"), pl = find(t, "plain tools");
  if (!a) return <p className="error">No holdout score for {lead.model}'s run.</p>;
  const vs = (n: string) => (mi ? `vs ${n} for the copy of the pharmacist` : "");
  const tiles: [string, string, string][] = [
    ["Stock-out days", num(a.stockout_days), vs(num(mi?.stockout_days ?? 0))],
    ["Shelf-out days", num(a.shelfout_days), vs(num(mi?.shelfout_days ?? 0))],
    ["Revenue lost", rp(a.lost_revenue), vs(rp(mi?.lost_revenue ?? 0))],
  ];
  return (
    <section className="stack" style={{ gap: 14 }}>
      {mi && (
        <p className="hero">
          At the pharmacist's stock value, {lead.model}'s frozen policy has {share(a.stockout_days, mi.stockout_days)} the
          stock-out days of the copy of the pharmacist, {share(a.shelfout_days, mi.shelfout_days)} its shelf-out days
          and {share(a.lost_revenue, mi.lost_revenue)} its lost revenue.
        </p>
      )}
      <p className="lede">
        {tb && <>It {vsTextbook(a, tb)} the textbook rule on lost revenue ({rp(a.lost_revenue)} vs {rp(tb.lost_revenue)}).{" "}</>}
        {pl && (pl.stockout_days < a.stockout_days
          ? <>The plain tools, with no LLM, have fewer stock-out days ({num(pl.stockout_days)}) but many
            more shelf-outs ({num(pl.shelfout_days)}).</>
          : <>It also has fewer stock-out days than the plain tools with no LLM ({num(pl.stockout_days)}).</>)}
        {" "}Scored once on July–December 2025, after the policy was frozen, with uncensored demand.
      </p>
      <div className="tiles three">
        {tiles.map(([l, v, n]) => (
          <div className="tile" key={l}>
            <div className="label">{l}, {lead.model}</div>
            <div className="value">{v}</div>
            <div className="note">{n}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

/** The caveats that depend on the scores, worded from them and shown only when they apply. */
function DataCaveats({ c }: { c: Contest }) {
  const t = c.table.uncensored;
  const model = c.runs[0].model;
  const a = find(t, agentKey(c.runs[0])), tb = find(t, "textbook"), pl = find(t, "plain tools");
  if (!a) return null;
  const tie = tb && vsTextbook(a, tb) === "ties";
  const morePlain = pl && pl.stockout_days > 0 && a.stockout_days > pl.stockout_days;
  return (
    <>
      {tie && <li>On lost revenue, {model} and the textbook rule are within 10% of each other
        ({rp(a.lost_revenue)} vs {rp(tb.lost_revenue)}). With one run each, that is a tie.</li>}
      {morePlain && <li>{model} has more stock-out days than the plain tools with no LLM
        ({num(a.stockout_days)} vs {num(pl.stockout_days)}). Its rules below show where it chose
        to hold less.</li>}
    </>
  );
}

function ContestTable({ rows, lead, color }: { rows: ContestRow[]; lead: string; color: (p: string) => string }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>Policy</th><th>Stock-out days</th><th>Shelf-out days</th><th>Revenue lost</th>
            <th>Avg stock</th><th>Scale</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.policy} className={r.policy === lead ? "em" : ""}>
              <td><i className="key-box" style={{ background: color(r.policy), marginRight: 8 }} />{label(r.policy)}</td>
              <td>{num(r.stockout_days)}</td>
              <td>{num(r.shelfout_days)}</td>
              <td>{rp(r.lost_revenue)}</td>
              <td>{rp(r.avg_stock_value)}</td>
              <td>{r.scale == null ? "–" : `${r.scale.toFixed(2)}×`}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted" style={{ marginTop: 6 }}>
        Scale: how much each policy's reorder levels were multiplied by to hold the pharmacist's stock value.
      </p>
    </div>
  );
}

function Decided({ run }: { run: ContestRun }) {
  const reasons = run.rationale.split(/(?:^|\s)(?:\d+\.|-)\s(?=[A-Z])/).map((s) => s.trim()).filter(Boolean);
  return (
    <>
      <p className="small">
        On 30 June 2025, {run.model} set a few rules instead of one setting per drug. The first rule
        that matches a drug wins. Anything a rule leaves unset comes from the default.
      </p>
      <ol className="rules">
        {run.rules.map((r) => (
          <li key={r.name}>
            <b>{r.name}</b> <span className="muted">· {num(r.drugs)} drugs</span>
            <div className="small">{r.when}: {r.sets}.</div>
          </li>
        ))}
      </ol>
      <h3>Its reasons</h3>
      <ul className="reasons small">{reasons.map((r) => <li key={r}>{r}</li>)}</ul>
      {run.usd != null && <p className="small muted">Planning cost ${run.usd.toFixed(3)} in model calls.</p>}
    </>
  );
}

const half = (w: string) => {
  const [a, b] = w.split(" to ").map((d) => new Date(d + "T00:00:00Z"));
  const m = (d: Date) => d.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });
  return `${m(a)}–${m(b)} ${b.getUTCFullYear()}`;
};

function InSample({ run, holdoutRuns, runs }: { run: ContestRun; holdoutRuns: number | null; runs: number }) {
  const pickRows = (rows: ContestRow[]) => ({ a: find(rows, "agent"), m: find(rows, "mimic") });
  const lines = [
    ...[...run.in_sample].reverse().map((w) => ({ when: half(w.window), kind: "backtest", ...pickRows(w.rows) })),
    { when: "Jul–Dec 2025", kind: "holdout", ...pickRows(run.score.uncensored) },
  ].flatMap(({ a, m, ...l }) => (a && m ? [{ ...l, a, m }] : []));
  return (
    <>
      <p className="small">
        While planning, {run.model} could only backtest its rules on the past half-years below.
        Compare them with the holdout row.{" "}
        {lines.length > 0 && lines.every((l) => l.a.lost_revenue < l.m.lost_revenue) &&
          "It lost less revenue than the mimic in every period, so the backtest called the ranking."}
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Period</th><th>Stock-outs</th><th>Shelf-outs</th><th>Lost</th><th>Mimic lost</th></tr>
          </thead>
          <tbody>
            {lines.map((l) => (
              <tr key={l.when} className={l.kind === "holdout" ? "em" : ""}>
                <td>{l.when}<div className="muted" style={{ fontSize: 12, fontWeight: 400 }}>{l.kind}</div></td>
                <td>{num(l.a.stockout_days)}</td>
                <td>{num(l.a.shelfout_days)}</td>
                <td>{rp(l.a.lost_revenue)}</td>
                <td>{rp(l.m.lost_revenue)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        All at the pharmacist's stock value in each period, uncensored demand. The policy was frozen
        before the holdout was opened{holdoutRuns != null && <>; the holdout has been scored{" "}
        {holdoutRuns === 2 ? "twice" : `${holdoutRuns} times`} in all{holdoutRuns === runs && ", once per frozen run"}</>}.
      </p>
    </>
  );
}
