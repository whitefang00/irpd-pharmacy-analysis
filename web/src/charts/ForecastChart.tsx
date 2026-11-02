import { useMemo, useState } from "react";
import type { ForecastLine, RollingPoint, SaleDay, Spell } from "../api";
import { day as fmtDay, num } from "../api";
import { DAY, compact, iso, linear, linePath, ms, niceTicks, useWidth } from "./util";

interface Props {
  sales: SaleDay[];       // daily, history and (if shown) holdout
  stockouts: Spell[];
  current: ForecastLine;
  ml: ForecastLine | null;
  historyWeeks: number;
  showActuals: boolean;   // holdout on: bars continue into the forecast weeks
  // Holdout on: each H2 week's forecast, remade weekly from the data up to then
  // (a 4-week total; drawn per week). Replaces the flat 30 June lines.
  rolling?: RollingPoint[] | null;
  height?: number;        // plot height (default 230)
  compact?: boolean;      // no legend: the page shows one shared legend
}

const M = { l: 44, r: 58, t: 26, b: 24 };
const SERIES = [
  { key: "current", label: "Current", color: "var(--s1)" },
  { key: "ml", label: "ML", color: "var(--s2)" },
] as const;

const monday = (t: number) => t - ((new Date(t).getUTCDay() + 6) % 7) * DAY;

/** Weekly units sold (bars), then each forecaster's weekly mean (lines) from the cutoff. */
export function ForecastChart(p: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = p.height ?? 230;
  const first = ms(p.current.weekly[0].week_start);
  const roll = p.rolling && p.rolling.length ? p.rolling : null;
  // Holdout off: only next week's forecast, the one the pharmacist acts on now.
  const nf = roll ? roll.length : 1;

  const d = useMemo(() => {
    const weeks = Array.from({ length: p.historyWeeks + nf }, (_, i) => first + (i - p.historyWeeks) * 7 * DAY);
    const sold = new Map<number, number>();
    for (const s of p.sales) {
      const k = monday(ms(s.date));
      sold.set(k, (sold.get(k) ?? 0) + s.units);
    }
    const outDays = new Map<number, number>();
    for (const s of p.stockouts) {
      for (let t = ms(s.start); t <= ms(s.stop); t += DAY) {
        const k = monday(t);
        outDays.set(k, (outDays.get(k) ?? 0) + 1);
      }
    }
    const actual = weeks.map((t) => (t < first || p.showActuals ? sold.get(t) ?? 0 : null));
    const runs: { current: (number | null)[]; ml: (number | null)[] | null } = roll
      ? { current: roll.map((r) => r.current / 4),
          ml: roll.some((r) => r.ml != null) ? roll.map((r) => (r.ml == null ? null : r.ml / 4)) : null }
      : { current: p.current.weekly.slice(0, nf).map((w) => w.mean),
          ml: p.ml?.weekly.slice(0, nf).map((w) => w.mean) ?? null };
    const high = roll ? [] : p.current.weekly.slice(0, nf).map((w) => w.p90);
    const max = Math.max(1, ...actual.map((v) => v ?? 0), ...high,
                         ...runs.current.map((v) => v ?? 0), ...(runs.ml ?? []).map((v) => v ?? 0));
    return { weeks, actual, outDays, runs, high, max };
  }, [p.sales, p.stockouts, p.current, p.ml, p.historyWeeks, p.showActuals, first, nf, roll]);

  const iw = width - M.l - M.r;
  const n = d.weeks.length;
  const slot = iw / n;
  const xi = (i: number) => M.l + (i + 0.5) * slot;
  const ticks = niceTicks(d.max, 4);
  const y = linear(0, ticks[ticks.length - 1] || 1, M.t + H, M.t);
  const bot = M.t + H;
  const barW = Math.max(2, Math.min(18, slot - 2));
  const cutX = M.l + p.historyWeeks * slot;
  const fIdx = (j: number) => p.historyWeeks + j;
  // One point per forecaster: set them side by side so close values don't hide each other.
  const nudge = (key: string) => (roll ? 0 : key === "current" ? -5 : 5);

  const lines = SERIES.flatMap((s) => {
    const vals = d.runs[s.key];
    if (!vals) return [];
    return [{ ...s, vals, path: linePath(vals.map((v, j) => [xi(fIdx(j)), v == null ? null : y(v)])) }];
  });
  // Direct labels at the right end, nudged apart if they'd overlap.
  const ends = lines.map((l) => ({ key: l.key, y: y(l.vals[l.vals.length - 1] ?? 0) }));
  if (ends.length === 2 && Math.abs(ends[0].y - ends[1].y) < 13) {
    const mid = (ends[0].y + ends[1].y) / 2;
    const up = ends[0].y <= ends[1].y ? 0 : 1;
    ends[up].y = mid - 6.5;
    ends[1 - up].y = mid + 6.5;
  }

  const monthStarts = d.weeks
    .map((t, i) => ({ t, i }))
    .filter(({ t, i }) => i > 0 && new Date(t).getUTCMonth() !== new Date(d.weeks[i - 1]).getUTCMonth())
    .filter((_, k, all) => k % Math.max(1, Math.ceil(all.length / Math.max(2, Math.floor(iw / 60)))) === 0);

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const i = Math.floor((e.clientX - rect.left - M.l) / slot);
    setHover(i >= 0 && i < n ? i : null);
  }
  const hw = hover == null ? null : d.weeks[hover];
  const j = hover == null ? -1 : hover - p.historyWeeks;

  return (
    <div className="chart" ref={ref}>
      {!p.compact && <div className="legend" aria-hidden>
        <span><i className="key-box" style={{ background: "var(--bar)" }} />Units sold per week</span>
        <span><i className="key-line" style={{ background: "var(--s1)" }} />Current method</span>
        {!roll && <span><i className="key-line" style={{ background: "var(--s1)", opacity: 0.4 }} />Its high case (p90)</span>}
        {p.ml && <span><i className="key-line" style={{ background: "var(--s2)" }} />ML model</span>}
      </div>}
      <svg width={width} height={M.t + H + M.b} role="img"
           aria-label="Weekly units sold, and the weekly demand forecast from each forecaster after the cutoff">
        <rect x={cutX} y={M.t - 18} width={M.l + iw - cutX} height={H + 18} fill="var(--holdout)" />
        <line x1={cutX} x2={cutX} y1={M.t - 18} y2={bot} stroke="var(--ink-2)" />
        <text x={roll ? cutX + 6 : cutX - 6} y={M.t - 6} textAnchor={roll ? "start" : "end"}
              style={{ fill: "var(--ink-2)" }}>
          {roll ? `Forecast remade each week from ${fmtDay(p.current.weekly[0].week_start)}`
                : `Next week's forecast (week of ${fmtDay(p.current.weekly[0].week_start)}) →`}
        </text>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={M.l + iw} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
            <text x={M.l - 6} y={y(v) + 3.5} textAnchor="end">{compact(v)}</text>
          </g>
        ))}
        {d.actual.map((v, i) => {
          if (v == null || v <= 0) return null;
          const h = bot - y(v);
          const r = Math.min(barW / 2, 2, h);
          const bx = xi(i) - barW / 2;
          return <path key={i} fill="var(--bar)" opacity={hover == null || hover === i ? 1 : 0.55}
                       d={`M${bx},${bot}V${bot - h + r}q0,${-r} ${r},${-r}h${barW - 2 * r}q${r},0 ${r},${r}V${bot}Z`} />;
        })}
        <line x1={M.l} x2={M.l + iw} y1={bot} y2={bot} stroke="var(--axis)" />
        {/* The default forecaster's range: expected up to the high case (p90). */}
        {/* The default forecaster's high case: a whisker from its forecast up to p90. */}
        {d.high.map((hi, j) => (
          <g key={`hi${j}`} stroke="var(--s1)" strokeWidth={2} opacity={0.4}>
            <line x1={xi(fIdx(j)) - 5} x2={xi(fIdx(j)) - 5} y1={y(d.runs.current[j] ?? 0)} y2={y(hi)} />
            <line x1={xi(fIdx(j)) - 10} x2={xi(fIdx(j))} y1={y(hi)} y2={y(hi)} />
          </g>
        ))}
        {lines.map((l) => (
          <path key={l.key} d={l.path} fill="none" stroke={l.color} strokeWidth={2}
                strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {lines.map((l) => l.vals.map((v, j) => v != null && (
          <circle key={`${l.key}${j}`} cx={xi(fIdx(j)) + nudge(l.key)} cy={y(v)} r={roll ? 3.5 : 4} fill={l.color}
                  stroke="var(--surface)" strokeWidth={2} />
        )))}
        {lines.map((l, k) => (
          <text key={l.key} x={xi(n - 1) + 8} y={ends[k].y + 3.5} style={{ fill: "var(--ink-2)" }}>{l.label}</text>
        ))}
        {monthStarts.map(({ t, i }) => (
          <text key={t} x={xi(i)} y={bot + 15} textAnchor="middle">
            {new Date(t).toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" })}
          </text>
        ))}
        {hover != null && (
          <g pointerEvents="none">
            <line x1={xi(hover)} x2={xi(hover)} y1={M.t} y2={bot} stroke="var(--ink-2)" />
            {j >= 0 && lines.map((l) => l.vals[j] != null && (
              <circle key={l.key} cx={xi(hover) + nudge(l.key)} cy={y(l.vals[j]!)} r={5} fill={l.color}
                      stroke="var(--surface)" strokeWidth={2} />
            ))}
          </g>
        )}
        <rect x={M.l} y={M.t - 18} width={iw} height={H + 18} fill="transparent"
              onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover != null && hw != null && (
        <div className="tip" style={{ left: Math.min(xi(hover) + 12, width - 200), top: M.t + 10 }}>
          <div className="t">Week of {fmtDay(iso(hw))}{j >= 0 && p.showActuals ? " · holdout" : ""}</div>
          {d.actual[hover] != null && <Row color="var(--bar)" box label="Sold" value={num(d.actual[hover] ?? 0)} />}
          {j >= 0 && lines.map((l) => l.vals[j] != null && (
            <Row key={l.key} color={l.color} label={l.key === "ml" ? "ML model" : "Current method"}
                 value={num(Math.round(l.vals[j]!))} />
          ))}
          {j >= 0 && !roll && <Row color="var(--s1)" box label="High case (p90)" value={num(d.high[j])} />}
          {(d.outDays.get(hw) ?? 0) > 0 && (
            <div className="r muted"><span>{d.outDays.get(hw)} stock-out day{d.outDays.get(hw) === 1 ? "" : "s"}: sales understate demand</span></div>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ color, label, value, box }: { color: string; label: string; value: string; box?: boolean }) {
  return (
    <div className="r">
      <span><i className={box ? "key-box" : "key-line"} style={{ background: color }} />{label}</span>
      <b>{value}</b>
    </div>
  );
}
