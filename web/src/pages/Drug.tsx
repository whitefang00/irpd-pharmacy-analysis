import { useEffect, useMemo, useState } from "react";
import type { DrugPage, DrugRow } from "../api";
import { api, day, num, rp } from "../api";
import { StockChart } from "../charts/StockChart";
import { DAY, iso, ms } from "../charts/util";
import { Seg } from "./Contest";
import { useLoad } from "./useLoad";

const DEFAULT = "SW2106002001"; // Omeprazole 20mg Novell: stock-outs, shelf-outs and alternatives
const RANGES = { "6 months": 182, "1 year": 365, All: 0 } as const;
type Range = keyof typeof RANGES;
const RANGE_LABELS = Object.fromEntries(Object.keys(RANGES).map((r) => [r, r]));

export function DrugPageView({ code }: { code?: string }) {
  const c = code ?? DEFAULT;
  const [holdout, setHoldout] = useState(false);
  const [range, setRange] = useState<Range>("1 year");
  const { data: page, error, loading } = useLoad(() => api.drug(c, holdout), `${c}:${holdout}`);

  return (
    <div className="stack">
      <Search />
      {error && <p className="error">{error}</p>}
      {page && !error && (
        <div className="stack" style={{ opacity: loading ? 0.6 : 1, transition: "opacity .15s" }}>
          <Header page={page} />
          <section className="card stack" style={{ gap: 8 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>Stock and sales</h2>
              <div className="row">
                <Seg value={range} set={setRange} labels={RANGE_LABELS} name="Time range" />
                <label className="check">
                  <input type="checkbox" checked={holdout} onChange={(e) => setHoldout(e.target.checked)} />
                  Show H2 2025 (holdout)
                </label>
              </div>
            </div>
            <Charts page={page} range={range} />
            <SpellSummary page={page} />
          </section>
          <div className="grid2">
            <Alternatives page={page} />
            <div className="stack">
              <Purchases page={page} />
              <Prices page={page} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Charts({ page, range }: { page: DrugPage; range: Range }) {
  const h = page.holdout;
  const to = h?.stock.length ? h.stock[h.stock.length - 1].date : page.as_of;
  const days = RANGES[range];
  const from = useMemo(() => {
    const start = days ? iso(ms(page.as_of) - (days - 1) * DAY) : page.stock_start;
    return start < page.stock_start ? page.stock_start : start;
  }, [days, page.as_of, page.stock_start]);
  return (
    <StockChart
      stock={[...page.stock, ...(h?.stock ?? [])]}
      sales={[...page.sales, ...(h?.sales ?? [])]}
      purchases={[...page.purchases, ...(h?.purchases ?? [])]}
      stockouts={[...page.stockouts, ...(h?.stockouts ?? [])]}
      shelfouts={[...page.shelfouts, ...(h?.shelfouts ?? [])]}
      from={from} to={to} cutoff={page.as_of} showHoldout={!!h}
    />
  );
}

function Header({ page }: { page: DrugPage }) {
  const d = page.drug;
  return (
    <section className="stack" style={{ gap: 6 }}>
      <h1>{d.name}</h1>
      <div className="row small">
        <span className="pill">{d.tier}</span>
        <span>{d.category}</span>
        <span>{rp(d.revenue)} revenue to {day(page.as_of)}</span>
        <span className="muted">{d.code}</span>
      </div>
      <p className="small">
        {d.ingredients
          ? <>Active ingredient{d.ingredients.includes("+") ? "s" : ""}: <b>{d.ingredients}</b>
              {d.strength && <> · {d.strength}</>}{d.form && <> · {d.form}</>}</>
          : <span className="muted">Ingredients not mapped ({d.kind || "unknown kind"}).</span>}
      </p>
    </section>
  );
}

function SpellSummary({ page }: { page: DrugPage }) {
  const sum = (s: { days: number }[]) => s.reduce((a, b) => a + b.days, 0);
  const h = page.holdout;
  return (
    <p className="small">
      To the cutoff: {num(sum(page.stockouts))} stock-out days in {page.stockouts.length} spells,{" "}
      {num(sum(page.shelfouts))} shelf-out days, {page.purchases.length} deliveries.
      {h && <> In the holdout: {num(sum(h.stockouts))} stock-out days, {num(sum(h.shelfouts))} shelf-out
        days, {h.purchases.length} deliveries.</>}
      {" "}<span className="muted">Stock before {day(page.stock_start)} is left out: opening balances were wrong until the 2023 counts.</span>
    </p>
  );
}

function Alternatives({ page }: { page: DrugPage }) {
  const alts = page.alternatives;
  return (
    <section className="card stack" style={{ gap: 8 }}>
      <h2>Alternatives</h2>
      <p className="small">
        Other drugs with the same active ingredients. When this one runs out, customers mostly
        buy one of these instead.
      </p>
      {!alts.length ? <p className="small muted">None found.</p> : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Drug</th><th>Strength</th><th>Tier</th><th>Revenue</th><th>Stock at cutoff</th></tr></thead>
            <tbody>
              {alts.map((a) => (
                <tr key={a.code}>
                  <td><a href={`#/drug/${a.code}`}>{a.name}</a></td>
                  <td>{a.strength || "–"}{a.same_strength_form ? "" : " ·"}</td>
                  <td>{a.tier}</td><td>{rp(a.revenue)}</td>
                  <td>{a.stock_at_cutoff == null ? "–" : num(a.stock_at_cutoff)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {alts.some((a) => !a.same_strength_form) && (
        <p className="small muted">· different strength or form; same-strength matches are listed first.</p>
      )}
    </section>
  );
}

function Purchases({ page }: { page: DrugPage }) {
  const recent = [...page.purchases].reverse().slice(0, 8);
  const held = [...(page.holdout?.purchases ?? [])].reverse().slice(0, 6);
  const suppliers = new Set(page.purchases.map((p) => p.supplier));
  return (
    <section className="card stack" style={{ gap: 8 }}>
      <h2>Deliveries</h2>
      <p className="small">{page.purchases.length} to the cutoff from {suppliers.size} supplier{suppliers.size === 1 ? "" : "s"}; the latest are listed.</p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Date</th><th>Qty</th><th style={{ textAlign: "left" }}>Supplier</th></tr></thead>
          <tbody>
            {held.map((p, i) => (
              <tr key={`h${i}`}><td>{day(p.date)} <span className="pill">holdout</span></td><td>{num(p.qty)}</td><td className="wrap">{p.supplier}</td></tr>
            ))}
            {recent.map((p, i) => (
              <tr key={i}><td>{day(p.date)}</td><td>{num(p.qty)}</td><td className="wrap">{p.supplier}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Prices({ page }: { page: DrugPage }) {
  return (
    <section className="card stack" style={{ gap: 8 }}>
      <h2>Price history</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>From</th><th>Unit</th><th>Price</th></tr></thead>
          <tbody>
            {[...page.prices].reverse().map((p, i) => (
              <tr key={i}><td>{day(p.valid_from)}</td><td>{p.unit}</td><td>{rp(p.price)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Search() {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<DrugRow[]>([]);
  useEffect(() => {
    if (!q.trim()) { setRows([]); return; }
    let live = true;
    const t = setTimeout(() => api.search(q)
      .then((r) => live && setRows(r))
      .catch(() => live && setRows([])), 150);
    return () => { live = false; clearTimeout(t); };
  }, [q]);
  const go = (code: string) => { window.location.hash = `#/drug/${code}`; setQ(""); };
  return (
    <div>
      <input type="search" placeholder="Search drugs by name, e.g. amlodipine, paracetamol, Flazen"
             value={q} onChange={(e) => setQ(e.target.value)}
             onKeyDown={(e) => {
               if (e.key === "Enter" && rows[0]) go(rows[0].code);
               if (e.key === "Escape") setQ("");
             }}
             aria-label="Search drugs" />
      {rows.length > 0 && (
        <div className="results">
          {rows.map((r) => (
            <button key={r.code} onClick={() => go(r.code)}>
              <span>{r.name} <span className="small muted">{r.ingredients ?? ""}</span></span>
              <span className="small">{r.tier} · {rp(r.revenue)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
