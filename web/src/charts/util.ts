import { useEffect, useRef, useState } from "react";

/** Width of a container, kept up to date. */
export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export const DAY = 86_400_000;
export const ms = (iso: string) => Date.parse(iso + "T00:00:00Z");
export const iso = (t: number) => new Date(t).toISOString().slice(0, 10);

export function linear(d0: number, d1: number, r0: number, r1: number) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = (v: number) => r0 + (v - d0) * k;
  f.invert = (p: number) => d0 + (p - r0) / (k || 1);
  return f;
}

/** Clean ticks 0..max, about `n` of them. */
export function niceTicks(max: number, n = 4): number[] {
  if (max <= 0) return [0];
  const raw = max / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [0];
  while (out[out.length - 1] < max - 1e-9) out.push(+(out.length * step).toFixed(6));
  return out;
}

export const compact = (v: number) =>
  Math.abs(v) >= 1e9 ? `${+(v / 1e9).toFixed(1)}B`
  : Math.abs(v) >= 1e6 ? `${+(v / 1e6).toFixed(1)}M`
  : Math.abs(v) >= 1e3 ? `${+(v / 1e3).toFixed(1)}K`
  : `${+v.toFixed(1)}`;

/** Month ticks between two times, thinned to fit the width. */
export function monthTicks(t0: number, t1: number, width: number): number[] {
  const out: number[] = [];
  const d = new Date(t0);
  let m = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1));
  while (m.getTime() <= t1) {
    out.push(m.getTime());
    m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1));
  }
  const every = Math.max(1, Math.ceil(out.length / Math.max(2, Math.floor(width / 70))));
  const steps = [1, 2, 3, 6, 12];
  const step = steps.find((s) => s >= every) ?? 12;
  return out.filter((t) => new Date(t).getUTCMonth() % step === 0);
}

export const monthLabel = (t: number) => {
  const d = new Date(t);
  return d.getUTCMonth() === 0
    ? String(d.getUTCFullYear())
    : d.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });
};

/** An SVG path through points, breaking at nulls. */
export function linePath(pts: [number, number | null][]): string {
  let d = "";
  let pen = false;
  for (const [x, y] of pts) {
    if (y == null) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** A filled band between two lines (same xs). */
export function bandPath(xs: number[], lo: number[], hi: number[]): string {
  if (!xs.length) return "";
  const top = xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)},${hi[i].toFixed(1)}`).join("");
  const bot = xs.map((_, j) => {
    const i = xs.length - 1 - j;
    return `L${xs[i].toFixed(1)},${lo[i].toFixed(1)}`;
  }).join("");
  return top + bot + "Z";
}
