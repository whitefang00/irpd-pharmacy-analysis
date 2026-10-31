import { useMemo, useState } from "react";
import type { Purchase, SaleDay, Spell, StockDay } from "../api";
import { day as fmtDay, num } from "../api";
import {
  DAY, bandPath, compact, iso, linePath, linear, monthLabel, monthTicks, ms, niceTicks, useWidth,
} from "./util";

interface Props {
  stock: StockDay[];
  sales: SaleDay[];
  purchases: Purchase[];
  stockouts: Spell[];
  shelfouts: Spell[];
  from: string;
  to: string;
  cutoff: string;
  showHoldout: boolean;
}

const M = { l: 44, r: 12, t: 22 };
const STOCK_H = 210;
const STRIP = 18; // deliveries row
const GAP = 30;
const SALES_H = 70;
const AXIS = 22;

/**
 * Stock on the shop floor and in the upstairs store (stacked), with stock-out and
 * shelf-out spells shaded and deliveries marked; units sold below on its own axis.
 */
export function StockChart(p: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const t0 = ms(p.from), t1 = ms(p.to);
  const span = Math.round((t1 - t0) / DAY);
  const weekly = span > 200; // daily columns get too thin past ~200 days
  const iw = width - M.l - M.r;
  const x = linear(t0, t1, M.l, M.l + iw);
  const stockTop = M.t, stockBot = M.t + STOCK_H;
  const stripY = stockBot + 4;
  const salesTop = stockBot + STRIP + GAP, salesBot = salesTop + SALES_H;
  const H = salesBot + AXIS;

  const d = useMemo(() => {
    const stock = p.stock.filter((s) => s.date >= p.from && s.date <= p.to);
    const byDate = new Map(stock.map((s) => [s.date, s]));
    const sold = new Map(p.sales.map((s) => [s.date, s.units]));
    const buys = new Map<string, Purchase[]>();
    for (const b of p.purchases) {
      if (b.date < p.from || b.date > p.to) continue;
      buys.set(b.date, [...(buys.get(b.date) ?? []), b]);
    }
    // Sales bars: one per day, or one per Monday-start week.
    const bars: { t: number; units: number; len: number }[] = [];
    if (weekly) {
      const wk = new Map<number, number>();
      for (const [date, u] of sold) {
        if (date < p.from || date > p.to) continue;
        const t = ms(date);
        const dow = (new Date(t).getUTCDay() + 6) % 7;
        const k = t - dow * DAY;
        wk.set(k, (wk.get(k) ?? 0) + u);
      }
      for (const [t, units] of wk) bars.push({ t, units, len: 7 });
    } else {
      for (const [date, units] of sold)
        if (date >= p.from && date <= p.to) bars.push({ t: ms(date), units, len: 1 });
    }
    const maxStock = Math.max(1, ...stock.map((s) => Math.max(0, s.floor ?? 0) + Math.max(0, s.store ?? 0)));
    const maxSold = Math.max(1, ...bars.map((b) => b.units));
    return { stock, byDate, sold, buys, bars, maxStock, maxSold };
  }, [p.stock, p.sales, p.purchases, p.from, p.to, weekly]);

  const yTicks = niceTicks(d.maxStock, 4);
  const y = linear(0, yTicks[yTicks.length - 1] || 1, stockBot, stockTop);
  const sTicks = niceTicks(d.maxSold, 2);
  const ys = linear(0, sTicks[sTicks.length - 1] || 1, salesBot, salesTop);

  const xs = d.stock.map((s) => x(ms(s.date)));
  const floor = d.stock.map((s) => Math.max(0, s.floor ?? 0));
  const total = d.stock.map((s, i) => floor[i] + Math.max(0, s.store ?? 0));
  const floorY = floor.map(y), totalY = total.map(y);
  const zero = xs.map(() => stockBot);

  const spellRect = (s: Spell) => {
    const a = Math.max(ms(s.start), t0), b = Math.min(ms(s.stop) + DAY, t1 + DAY);
    if (b <= t0 || a > t1) return null;
    return { x: x(a), w: Math.max(1.5, x(b) - x(a)) };
  };
  const barW = Math.max(1, Math.min(24, (iw / span) * (weekly ? 7 : 1) - 2));
  const hoverIso = hover == null ? null : iso(hover);
  const hs = hoverIso ? d.byDate.get(hoverIso) : undefined;
  const cut = ms(p.cutoff) + DAY / 2;

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const t = x.invert(e.clientX - rect.left);
    const snapped = Math.round((t - t0) / DAY) * DAY + t0;
    setHover(Math.min(t1, Math.max(t0, snapped)));
  }

  const ticks = monthTicks(t0, t1, iw);
  const inSpell = (list: Spell[]) =>
    hoverIso != null && list.some((s) => s.start <= hoverIso && hoverIso <= s.stop);

  return (
    <div className="chart" ref={ref}>
      <div className="legend" aria-hidden>
        <span><i className="key-box" style={{ background: "var(--s1)" }} />Shop floor</span>
        <span><i className="key-box" style={{ background: "var(--s2)" }} />Upstairs store</span>
        <span><i className="key-box" style={{ background: "var(--critical)", opacity: 0.35 }} />Stock-out</span>
        <span><i className="key-box" style={{ background: "var(--warning)", opacity: 0.5 }} />Shelf-out (floor empty, store has stock)</span>
        <span><svg width="10" height="10"><path d="M5 1 L9 9 L1 9 Z" fill="var(--ink-2)" /></svg>Delivery</span>
      </div>
      <svg width={width} height={H} role="img"
           aria-label="Daily stock on the shop floor and in the store, with stock-outs, shelf-outs, deliveries and units sold">
        {/* Holdout background */}
        {p.showHoldout && cut < t1 && (
          <g>
            <rect x={x(cut)} y={stockTop - 18} width={M.l + iw - x(cut)} height={salesBot - stockTop + 18}
                  fill="var(--holdout)" />
            <line x1={x(cut)} x2={x(cut)} y1={stockTop - 18} y2={salesBot} stroke="var(--ink-2)" strokeWidth={1} />
            <text x={x(cut) + 6} y={stockTop - 6} style={{ fill: "var(--ink-2)" }}>
              Holdout · H2 2025, never shown to the agent
            </text>
          </g>
        )}
        {/* Spells */}
        {p.shelfouts.map((s, i) => {
          const r = spellRect(s);
          return r && <rect key={`sh${i}`} x={r.x} y={stockTop} width={r.w} height={STOCK_H}
                            fill="var(--warning)" opacity={0.3} />;
        })}
        {p.stockouts.map((s, i) => {
          const r = spellRect(s);
          return r && <rect key={`so${i}`} x={r.x} y={stockTop} width={r.w} height={STOCK_H}
                            fill="var(--critical)" opacity={0.32} />;
        })}
        {/* Grid and y axis */}
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={M.l + iw} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth={1} />
            <text x={M.l - 6} y={y(v) + 3.5} textAnchor="end">{compact(v)}</text>
          </g>
        ))}
        <line x1={M.l} x2={M.l + iw} y1={stockBot} y2={stockBot} stroke="var(--axis)" />
        {/* Stacked stock */}
        <path d={bandPath(xs, floorY, totalY)} fill="var(--s2)" opacity={0.16} />
        <path d={bandPath(xs, zero, floorY)} fill="var(--s1)" opacity={0.16} />
        <path d={linePath(xs.map((v, i) => [v, totalY[i]]))} fill="none" stroke="var(--s2)"
              strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <path d={linePath(xs.map((v, i) => [v, floorY[i]]))} fill="none" stroke="var(--s1)"
              strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {/* Deliveries */}
        {[...d.buys.keys()].map((date) => {
          const cx = x(ms(date));
          return <path key={date} d={`M${cx} ${stripY} l4.5 9 h-9 Z`} fill="var(--ink-2)" />;
        })}
        {/* Sales */}
        {sTicks.map((v) => (
          <g key={`s${v}`}>
            <line x1={M.l} x2={M.l + iw} y1={ys(v)} y2={ys(v)} stroke="var(--grid)" />
            <text x={M.l - 6} y={ys(v) + 3.5} textAnchor="end">{compact(v)}</text>
          </g>
        ))}
        <text x={M.l} y={salesTop - 10} style={{ fill: "var(--ink-2)" }}>
          Units sold per {weekly ? "week" : "day"}
        </text>
        <clipPath id="sales-clip"><rect x={M.l} y={salesTop - 2} width={iw} height={SALES_H + 2} /></clipPath>
        {d.bars.map((b) => {
          const bx = x(b.t + (b.len * DAY) / 2) - barW / 2;
          const h = salesBot - ys(b.units);
          const r = Math.min(barW / 2, 2, h);
          return <path key={b.t} fill="var(--bar)" clipPath="url(#sales-clip)"
                       d={`M${bx},${salesBot}V${salesBot - h + r}q0,${-r} ${r},${-r}h${barW - 2 * r}q${r},0 ${r},${r}V${salesBot}Z`} />;
        })}
        <line x1={M.l} x2={M.l + iw} y1={salesBot} y2={salesBot} stroke="var(--axis)" />
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={salesBot + 15} textAnchor="middle">{monthLabel(t)}</text>
        ))}
        {/* Crosshair */}
        {hover != null && (
          <g pointerEvents="none">
            <line x1={x(hover)} x2={x(hover)} y1={stockTop} y2={salesBot} stroke="var(--ink-2)" strokeWidth={1} />
            {hs && <>
              <circle cx={x(hover)} cy={y(Math.max(0, hs.floor ?? 0))} r={4} fill="var(--s1)" stroke="var(--surface)" strokeWidth={2} />
              <circle cx={x(hover)} cy={y(Math.max(0, hs.floor ?? 0) + Math.max(0, hs.store ?? 0))} r={4} fill="var(--s2)" stroke="var(--surface)" strokeWidth={2} />
            </>}
          </g>
        )}
        <rect x={M.l} y={stockTop - 18} width={iw} height={salesBot - stockTop + 18} fill="transparent"
              onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover != null && hoverIso && (
        <div className="tip" style={{
          left: Math.min(x(hover) + 12, width - 200), top: stockTop + 10,
        }}>
          <div className="t">{fmtDay(hoverIso)}{hoverIso > p.cutoff ? " · holdout" : ""}</div>
          {hs ? <>
            <Row color="var(--s1)" label="Floor" value={num(hs.floor ?? 0)} />
            <Row color="var(--s2)" label="Store" value={num(hs.store ?? 0)} />
            <Row label="Total" value={num(hs.total ?? 0)} />
          </> : <div className="muted">no stock record</div>}
          <Row color="var(--bar)" label="Sold" value={num(d.sold.get(hoverIso) ?? 0)} />
          {inSpell(p.stockouts) && <div className="r"><span><i className="key-box" style={{ background: "var(--critical)" }} />Stock-out</span></div>}
          {inSpell(p.shelfouts) && <div className="r"><span><i className="key-box" style={{ background: "var(--warning)" }} />Shelf-out</span></div>}
          {(d.buys.get(hoverIso) ?? []).map((b, i) => (
            <div className="r" key={i}><span>Delivery {b.supplier}</span><b>{num(b.qty)}</b></div>
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ color, label, value }: { color?: string; label: string; value: string }) {
  return (
    <div className="r">
      <span>{color && <i className="key-line" style={{ background: color }} />}{label}</span>
      <b>{value}</b>
    </div>
  );
}
