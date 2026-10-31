// Types and fetchers for the API in docs/api.md. In dev, Vite proxies /api to uvicorn.

export type Tier = "fast" | "regular" | "slow" | "rare" | "none";

export interface DrugRow {
  code: string;
  name: string;
  category: string;
  tier: Tier;
  revenue: number;
  ingredients: string | null;
  strength: string | null;
  form: string | null;
  kind: string | null;
  mapped: boolean;
}

export interface StockDay { date: string; floor: number | null; store: number | null; total: number | null }
export interface SaleDay { date: string; units: number; revenue: number }
export interface Spell { start: string; stop: string; days: number; ended_by?: string | null }
export interface Purchase { date: string; qty: number; supplier: string; location: string }
export interface Price { unit: string; price: number; valid_from: string; valid_to: string | null }
export interface Alternative {
  code: string; name: string; tier: Tier; revenue: number; strength: string; form: string;
  same_strength_form: boolean; stock_at_cutoff: number | null;
}

export interface Actuals {
  stock: StockDay[]; sales: SaleDay[]; purchases: Purchase[]; stockouts: Spell[]; shelfouts: Spell[];
}

export interface DrugPage extends Omit<Actuals, "stockouts"> {
  drug: DrugRow;
  as_of: string;
  stock_start: string;
  stockouts: Spell[];
  prices: Price[];
  alternatives: Alternative[];
  holdout: Actuals | null;
}

export interface TierStats {
  tier: Tier; drugs: number; drug_days: number; stockout_days: number; shelfout_days: number;
  stockout_share: number; shelfout_share: number;
}

export interface Overview {
  as_of: string;
  period: { first: string; last: string };
  revenue_total: number;
  revenue_by_year: { year: number; revenue: number; drugs_sold: number }[];
  drugs: { listed: number; sold: number; fast_or_regular: number };
  tiers: TierStats[];
  revenue_share_mapped: number;
}

export interface ReplayRow {
  policy: string;
  kind: "ledger" | "reference" | "matched";
  drugs: number; days: number;
  stockout_days: number; stockout_days_long: number; shelfout_days: number;
  stockout_share: number; shelfout_share: number;
  lost_revenue_stockout: number; lost_revenue_shelfout: number; lost_revenue: number;
  avg_stock_value: number; units_sold: number;
}

export interface CurvePoint {
  factor: number; avg_stock_value: number; stockout_days: number; shelfout_days: number;
  lost_revenue: number;
}

export interface Replay {
  period: { first: string; last: string };
  scope: string;
  fair_contest: string;
  matched_factor: Record<string, number | null>;
  summary: ReplayRow[];
  tradeoff: Record<string, CurvePoint[]>;
  per_drug: ({ code: string; name: string } & Record<string, unknown>)[];
}

export type Demand = "uncensored" | "recorded";

export interface ContestRow {
  policy: string;
  stockout_days: number; shelfout_days: number; lost_revenue: number; avg_stock_value: number;
  stockout_days_long?: number; scale?: number | null;
}

export interface ContestRun {
  run_id: string; model: string; as_of: string; frozen_at: string; usd: number | null;
  rationale: string;
  rules: { name: string; when: string; sets: string; drugs: number }[];
  score: Record<Demand, ContestRow[]>;
  tradeoff: Record<Demand, CurvePoint[]>;
  in_sample: { window: string; rows: ContestRow[] }[];
}

export interface Contest {
  drugs: number;
  holdout_runs: number | null;
  table: Record<Demand, ContestRow[]>;
  mimic_tradeoff: Record<Demand, CurvePoint[]>;
  runs: ContestRun[];  // best first
}

// /chat streams these server-sent events (docs/api.md).
export type ChatEvent =
  | { event: "tool_call"; data: { id: string; name: string; args: unknown } }
  | { event: "tool_result"; data: { id: string; name: string; output: unknown; truncated: boolean } }
  | { event: "message"; data: { text: string } }
  | { event: "answer"; data: { text: string } }
  | { event: "done"; data: { model: string; as_of: string; stopped: string; usage: Usage } }
  | { event: "error"; data: { code: string; message: string } };

export interface Usage { input_tokens: number; output_tokens: number; cost_usd: number }

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${path}`);
  return r.json() as Promise<T>;
}

export const api = {
  overview: () => get<Overview>("/overview"),
  search: (q: string) => get<DrugRow[]>(`/drugs?q=${encodeURIComponent(q)}&limit=12`),
  drug: (code: string, holdout: boolean) =>
    get<DrugPage>(`/drugs/${encodeURIComponent(code)}?holdout=${holdout}`),
  replay: () => get<Replay>("/replay"),
  contest: () => get<Contest | null>("/contest"),
};

// Formatting
export const rp = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1e9) return `Rp ${(v / 1e9).toFixed(a >= 1e10 ? 1 : 2)}B`;
  if (a >= 1e6) return `Rp ${(v / 1e6).toFixed(a >= 1e8 ? 0 : 1)}M`;
  if (a >= 1e3) return `Rp ${(v / 1e3).toFixed(0)}K`;
  return `Rp ${Math.round(v)}`;
};
export const num = (v: number) => Math.round(v).toLocaleString("en-US");
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
export const day = (iso: string) =>
  new Date(iso + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
