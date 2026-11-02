import { api, day, num, pct, rp } from "../api";
import { useLoad } from "./useLoad";

export function OverviewPage() {
  const { data: o, error } = useLoad(api.overview, "overview");

  return (
    <div className="stack" style={{ gap: 28 }}>
      <section className="stack" style={{ gap: 12 }}>
        <h1>Where a real pharmacy loses sales</h1>
        <p className="hero">
          A study of two and a half years of one Indonesian pharmacy's sales and stock ledger:
          which stock-outs and empty shelves cost sales, and when customers just bought another
          brand. Reorder policies were tested against the pharmacist's own habits on six months
          kept sealed until scoring. An AI assistant answers questions on the data.
        </p>
        <p className="lede small">
          Data: the IRPD dataset (CC BY 4.0). It covers one retail pharmacy in Central Java, with
          sales and a full stock ledger for 2023–2025. There are no costs, so all money is revenue
          at selling price. The agent sees data up to {o ? day(o.as_of) : "30 Jun 2025"}. The
          second half of 2025 is held back for scoring.
        </p>
      </section>

      <section className="card stack" style={{ gap: 8 }}>
        <h2>What it found</h2>
        <ul className="small">
          <li><b>2.75% of revenue was lost</b> to stock-outs and empty shelves (Rp 194M, Oct 2023 to
            the cutoff). Only about 30% of those lost sales moved to another brand of the same
            medicine, so about 2.3% was truly lost.</li>
          <li><b>Consistent reorder rules beat habit, with or without AI.</b> On the sealed H2 2025,
            a textbook rule lost 62% less revenue than a copy of the pharmacist's habits. The AI
            agent's policy lost 63% less, a tie. It pulled ahead only on empty shelves.</li>
          <li><b>Better forecasting barely helps.</b> A machine-learning model was only slightly more
            accurate than simple smoothing. Outside Eid, weekly demand is mostly random, so the
            forecasts are nearly flat and the safety buffer does the work.</li>
        </ul>
      </section>

      {error && <p className="error">{error}</p>}
      {o && (
        <>
          <div className="tiles">
            <Tile label="Revenue to the cutoff" value={rp(o.revenue_total)}
                  note={`${day(o.period.first)} – ${day(o.period.last)}`} />
            <Tile label="Drugs sold" value={num(o.drugs.sold)}
                  note={`${num(o.drugs.fast_or_regular)} fast or regular`} />
            <Tile label="Revenue with ingredients mapped" value={pct(o.revenue_share_mapped, 0)}
                  note="needed to find alternatives" />
            <Tile label="Fast drugs out of stock" value={pct(o.tiers.find((t) => t.tier === "fast")?.stockout_share ?? 0)}
                  note="of days in their selling period" />
          </div>

          <section className="card stack" style={{ gap: 10 }}>
            <h2>Running out, by how often a drug sells</h2>
            <p className="small">
              Share of days in each drug's selling period (from Oct 2023) when the whole pharmacy
              was out (stock-out), or the shop floor was empty while the upstairs store had stock
              (shelf-out).
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Tier</th><th>Drugs</th><th>Stock-out days</th><th>Share</th>
                    <th>Shelf-out days</th><th>Share</th></tr>
                </thead>
                <tbody>
                  {o.tiers.map((t) => (
                    <tr key={t.tier}>
                      <td>{t.tier}</td><td>{num(t.drugs)}</td>
                      <td>{num(t.stockout_days)}</td><td>{pct(t.stockout_share)}</td>
                      <td>{num(t.shelfout_days)}</td><td>{pct(t.shelfout_share)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small muted">
              Tiers by sale days a year: fast 156+, regular 52+, slow 12+, rare fewer.
            </p>
          </section>

          <section className="tiles">
            <Link href="#/replay" title="Agent vs pharmacist"
                  text="Reorder policies replayed on July–December 2025 and scored against a copy of the pharmacist's habits." />
            <Link href="#/forecasts" title="Forecasts"
                  text="Both forecasters for every drug, and how each did on H2 2025." />
            <Link href="#/drug" title="A drug's stock line"
                  text="Floor and store stock, stock-outs and shelf-outs, deliveries and sales." />
            <Link href="#/chat" title="Ask the agent"
                  text="Every number in its answer comes from a tool call you can open." />
          </section>
        </>
      )}
    </div>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

function Link({ href, title, text }: { href: string; title: string; text: string }) {
  return (
    <a className="tile" href={href} style={{ color: "inherit", textDecoration: "none" }}>
      <h3 style={{ color: "var(--accent)" }}>{title} →</h3>
      <p className="small" style={{ marginTop: 4 }}>{text}</p>
    </a>
  );
}
