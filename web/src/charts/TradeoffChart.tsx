import { useState } from "react";
import type { CurvePoint } from "../api";
import { num, rp } from "../api";
import { compact, linePath, linear, niceTicks, useWidth } from "./util";

export type Metric = "stockout_days" | "shelfout_days" | "lost_revenue";
export const METRIC_LABEL: Record<Metric, string> = {
  stockout_days: "Stock-out days",
  shelfout_days: "Shelf-out days",
  lost_revenue: "Revenue lost",
};

type Scores = Pick<CurvePoint, "avg_stock_value" | Metric>;

/** One policy: optionally a curve (scaled up and down) and a scored point. */
export interface Series {
  label: string;
  color: string;
  curve?: CurvePoint[];
  point?: Scores;
  hollow?: boolean;        // a point at matched stock
  name?: boolean;          // label the point on the chart
  legend?: "line" | "dot"; // how the legend shows it; none if unset
}

const M = { l: 52, r: 16, t: 14, b: 40 };
const H = 300;

interface Pt { x: number; y: number; label: string }

/** Stock held vs a cost of running short. `stock` marks the pharmacist's average stock. */
export function TradeoffChart({ series, metric, stock }: { series: Series[]; metric: Metric; stock: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<Pt | null>(null);
  const fmtY = (v: number) => (metric === "lost_revenue" ? rp(v) : num(v));
  const scored = series.flatMap((s) => (s.point ? [s.point] : []));

  // Frame the scored points; the far ends of the curves run off the plot (clipped).
  const maxX = 1.6 * Math.max(...scored.map((r) => r.avg_stock_value));
  const maxY = 1.4 * Math.max(...scored.map((r) => r[metric]));
  const xt = niceTicks(maxX, 4), yt = niceTicks(maxY, 4);
  const iw = width - M.l - M.r;
  const x = linear(0, xt[xt.length - 1], M.l, M.l + iw);
  const y = linear(0, yt[yt.length - 1], H - M.b, M.t);

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const cands: Pt[] = series.flatMap((s) => [
      ...(s.point ? [{ x: s.point.avg_stock_value, y: s.point[metric], label: s.label }] : []),
      ...(s.curve ?? []).map((q) => ({ x: q.avg_stock_value, y: q[metric], label: `${s.label} × ${q.factor}` })),
    ]);
    let best: Pt | null = null, bd = 24 * 24;
    for (const c of cands) {
      const dd = (x(c.x) - px) ** 2 + (y(c.y) - py) ** 2;
      if (dd < bd) { bd = dd; best = c; }
    }
    setHover(best);
  }

  return (
    <div className="chart" ref={ref}>
      <div className="legend">
        {series.filter((s) => s.legend).map((s) => (
          <span key={s.label}>
            {s.legend === "line"
              ? <i className="key-line" style={{ background: s.color }} />
              : <svg width="12" height="12"><circle cx="6" cy="6" r="4" fill={s.color} /></svg>}
            {s.label}
          </span>
        ))}
        <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--ink-2)" strokeWidth="2" /></svg>At matched stock</span>
      </div>
      <svg width={width} height={H} role="img"
           aria-label={`Average stock held against ${METRIC_LABEL[metric].toLowerCase()} for each policy`}>
        {yt.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={M.l + iw} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
            <text x={M.l - 6} y={y(v) + 3.5} textAnchor="end">{compact(v)}</text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={v} x={x(v)} y={H - M.b + 16} textAnchor="middle">{compact(v)}</text>
        ))}
        <line x1={M.l} x2={M.l + iw} y1={H - M.b} y2={H - M.b} stroke="var(--axis)" />
        <text x={M.l + iw} y={H - 6} textAnchor="end">Average stock held (Rp, at selling price) →</text>
        {/* The matched-stock line */}
        <line x1={x(stock)} x2={x(stock)} y1={M.t} y2={H - M.b} stroke="var(--axis)" />
        <text x={x(stock) + 5} y={M.t + 10}>pharmacist's stock</text>
        <clipPath id="plot"><rect x={M.l} y={M.t} width={iw} height={H - M.t - M.b} /></clipPath>
        {series.filter((s) => s.curve).map((s) => (
          <path key={s.label} clipPath="url(#plot)"
                d={linePath(s.curve!.map((q) => [x(q.avg_stock_value), y(q[metric])]))} fill="none"
                stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {series.filter((s) => s.point && s.name).map((s) => (
          <text key={s.label} x={x(s.point!.avg_stock_value) + 9} y={y(s.point![metric]) + 4}
                style={{ fill: "var(--ink-2)" }}>{s.label}</text>
        ))}
        {series.filter((s) => s.point).map((s) => (
          <circle key={s.label} cx={x(s.point!.avg_stock_value)} cy={y(s.point![metric])} r={s.hollow ? 5 : 4}
                  fill={s.hollow ? "var(--surface)" : s.color} stroke={s.hollow ? s.color : "var(--surface)"}
                  strokeWidth={2} />
        ))}
        {hover && <circle cx={x(hover.x)} cy={y(hover.y)} r={8} fill="none" stroke="var(--ink-2)" strokeWidth={1} />}
        <rect x={M.l} y={M.t} width={iw} height={H - M.t - M.b} fill="transparent"
              onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover && (
        <div className="tip" style={{ left: Math.min(x(hover.x) + 12, width - 210), top: Math.max(0, y(hover.y) - 20) }}>
          <div className="t">{hover.label}</div>
          <div className="r"><span>{METRIC_LABEL[metric]}</span><b>{fmtY(hover.y)}</b></div>
          <div className="r"><span>Avg stock</span><b>{rp(hover.x)}</b></div>
        </div>
      )}
    </div>
  );
}
