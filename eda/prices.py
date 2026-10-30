"""EDA route 4: prices and revenue.

Price history per drug and unit, a first look at how demand responds to price
changes, revenue breakdowns, discounts, and why revenue dipped in 2024.

Revenue here is `Subtotal` (qty x price - discount + fees). There are no
costs in the data, so nothing here is profit.

Run from the repo root:
    uvx --with duckdb --with pandas python -I eda/prices.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "irpd" / "processed_dataset"
SALES = DATA / "sales_processed.csv"
STOCK = DATA / "stock_processed.csv"

# A price level that lasts at most this many sale lines, with the same price
# on both sides, is treated as a one-off line price rather than a change.
BLIP_MAX_LINES = 2
# Price response: weeks either side of a change, and minimum volume.
WINDOW_WEEKS = 8
MIN_UNITS_BEFORE = 40
MIN_INSTOCK_WEEKS = 4

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

con = duckdb.connect()


def q(sql: str) -> pd.DataFrame:
    return con.sql(sql).df()


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n--- {title}")
    print(df.to_string(index=False))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


con.sql(f"""
    CREATE TABLE sales AS
    SELECT Date, Year, Time, InvoiceNumber, CashierID, ShiftPeriod,
           DrugCode, DrugName, Category, DrugClass, Manufacturer,
           Quantity, Unit, PriceTier, UnitPrice, DiscountPct, DiscountAmount,
           ServiceFee, PackagingFee, Subtotal, InvoiceDiscountPct,
           PaymentType, TotalPrice
    FROM read_csv_auto('{SALES}')
""")

# ---------------------------------------------------------------------------
section("1. PRICE HISTORY (per DrugCode + Unit)")
# ---------------------------------------------------------------------------

# Units are a trap: the same drug sells as Tab and Strip (or Pcs and Box) at
# the same time, at ~10x different prices. Keying on DrugCode alone would
# count every unit switch as a price change.
show("Units per drug (and are they sold side by side?)", q("""
    WITH u AS (
        SELECT DrugCode, Unit, min(Date) AS first_sale, max(Date) AS last_sale
        FROM sales GROUP BY ALL
    ),
    -- pairs of units of one drug whose sale date ranges overlap
    ov AS (
        SELECT DISTINCT a.DrugCode FROM u a JOIN u b
          ON a.DrugCode = b.DrugCode AND a.Unit < b.Unit
         AND a.first_sale <= b.last_sale AND b.first_sale <= a.last_sale
    ),
    d AS (
        SELECT DrugCode, count(*) AS n_units, DrugCode IN (SELECT DrugCode FROM ov) AS overlap
        FROM u GROUP BY 1
    )
    SELECT n_units, count(*) AS drugs, sum(overlap::INT) AS units_overlap_in_time
    FROM d GROUP BY 1 ORDER BY 1
"""))

show("Same drug+unit sold at more than one price on the same day", q("""
    SELECT count(*) AS drug_unit_days,
           sum((n_prices > 1)::INT) AS with_two_plus_prices
    FROM (SELECT DrugCode, Unit, Date, count(DISTINCT UnitPrice) AS n_prices
          FROM sales GROUP BY ALL)
"""))

show("PriceTier (2 and 3 are near-unused)", q("""
    SELECT PriceTier, count(*) AS lines FROM sales GROUP BY 1 ORDER BY 1
"""))

# Daily price per drug+unit (the mode; same-day price splits are rare), then
# run-length encode into price spells.
con.sql("""
    CREATE TABLE daily_price AS
    SELECT DrugCode, Unit, Date, mode(UnitPrice) AS price,
           count(*) AS lines, sum(Quantity) AS units
    FROM sales GROUP BY ALL
""")


def build_spells(src: str) -> None:
    """Collapse consecutive days at one price into spells, into table `spells`."""
    con.sql(f"""
        CREATE OR REPLACE TABLE spells AS
        WITH flagged AS (
            SELECT *, (price IS DISTINCT FROM lag(price) OVER w)::INT AS new_spell
            FROM {src}
            WINDOW w AS (PARTITION BY DrugCode, Unit ORDER BY Date)
        ),
        numbered AS (
            SELECT *, sum(new_spell) OVER (PARTITION BY DrugCode, Unit ORDER BY Date) AS spell
            FROM flagged
        )
        SELECT DrugCode, Unit, spell, any_value(price) AS price,
               min(Date) AS start_date, max(Date) AS end_date,
               sum(lines) AS lines, sum(units) AS units
        FROM numbered GROUP BY DrugCode, Unit, spell
    """)
    con.sql("""
        CREATE OR REPLACE TABLE spells AS
        SELECT *, lag(price) OVER w AS prev_price, lead(price) OVER w AS next_price,
               lag(end_date) OVER w AS prev_end
        FROM spells
        WINDOW w AS (PARTITION BY DrugCode, Unit ORDER BY spell)
    """)


build_spells("daily_price")
raw_changes = q("SELECT count(*) AS n FROM spells WHERE prev_price IS NOT NULL").n[0]

# One-offs: a short spell with the same price before and after (A-B-A).
# Drop those days and rebuild, so A-B-A becomes one spell at A.
con.sql(f"""
    CREATE TABLE blip_days AS
    SELECT d.* FROM daily_price d JOIN spells s
      ON d.DrugCode = s.DrugCode AND d.Unit = s.Unit
     AND d.Date BETWEEN s.start_date AND s.end_date
    WHERE s.lines <= {BLIP_MAX_LINES} AND s.prev_price = s.next_price
""")
n_blips = q("SELECT count(*) AS n FROM blip_days").n[0]
build_spells("(SELECT * FROM daily_price EXCEPT SELECT * FROM blip_days)")

con.sql("""
    CREATE TABLE changes AS
    SELECT DrugCode, Unit, start_date AS change_date, prev_end AS last_old_date,
           prev_price AS old_price, price AS new_price,
           price / prev_price - 1 AS pct,
           -- A-B-A where B lasted longer than a one-off: temporary price
           (next_price = prev_price) AS reverted_later,
           lines AS lines_at_new_price
    FROM spells WHERE prev_price IS NOT NULL
""")

show("Raw vs real price changes", q(f"""
    SELECT {raw_changes} AS raw_transitions,
           {n_blips} AS one_off_days_removed,
           count(*) AS real_changes,
           count(DISTINCT (DrugCode, Unit)) AS drug_units_with_change,
           (SELECT count(DISTINCT (DrugCode, Unit)) FROM daily_price) AS drug_units_total,
           sum(reverted_later::INT) AS changes_later_reverted
    FROM changes
"""))

show("Real changes per drug+unit", q("""
    WITH n AS (
        SELECT d.DrugCode, d.Unit, count(c.change_date) AS n_changes
        FROM (SELECT DISTINCT DrugCode, Unit FROM daily_price) d
        LEFT JOIN changes c USING (DrugCode, Unit) GROUP BY ALL
    )
    SELECT CASE WHEN n_changes >= 5 THEN '5+' ELSE n_changes::VARCHAR END AS changes,
           count(*) AS drug_units
    FROM n GROUP BY 1 ORDER BY 1
"""))

show("Direction and size of real changes", q("""
    SELECT CASE WHEN pct > 0 THEN 'up' ELSE 'down' END AS direction,
           count(*) AS n,
           round(100 * quantile_cont(pct, 0.25), 1) AS p25_pct,
           round(100 * median(pct), 1) AS median_pct,
           round(100 * quantile_cont(pct, 0.75), 1) AS p75_pct,
           sum((reverted_later)::INT) AS reverted_later
    FROM changes GROUP BY 1
"""))

# Jumps of 2x or more are suspicious: likely the unit was redefined (e.g. a
# 'Tab' line priced as a strip), not a real repricing.
show("Extreme ratios (possible unit/pack redefinition)", q("""
    SELECT count(*) FILTER (WHERE new_price >= 2 * old_price) AS up_2x_plus,
           count(*) FILTER (WHERE new_price <= 0.5 * old_price) AS down_half_or_more,
           count(*) AS all_changes
    FROM changes
"""))
show("Examples of extreme ratios", q("""
    SELECT n.DrugName, c.Unit, c.change_date, c.old_price, c.new_price,
           c.lines_at_new_price, c.reverted_later
    FROM changes c
    JOIN (SELECT DrugCode, any_value(DrugName) AS DrugName FROM sales GROUP BY 1) n
      USING (DrugCode)
    WHERE c.new_price >= 2 * c.old_price OR c.new_price <= 0.5 * c.old_price
    ORDER BY abs(ln(c.new_price / c.old_price)) DESC LIMIT 8
"""))

# Timing. change_date is the first sale at the new price, so for slow movers
# the actual repricing may be earlier. The gap to the last old-price sale
# bounds it.
show("Gap between last old-price sale and first new-price sale (days)", q("""
    SELECT median(change_date - last_old_date) AS median_gap,
           quantile_cont(change_date - last_old_date, 0.9) AS p90_gap
    FROM changes
"""))
show("Real changes per year (and share of drug+units sold that year that changed)", q("""
    WITH sold AS (SELECT year(Date) AS y, count(DISTINCT (DrugCode, Unit)) AS n
                  FROM daily_price GROUP BY 1)
    SELECT year(change_date) AS year, count(*) AS changes,
           count(DISTINCT (DrugCode, Unit)) AS drug_units_changed,
           any_value(sold.n) AS drug_units_sold,
           round(100 * count(*) FILTER (WHERE pct > 0) / count(*), 1) AS pct_up
    FROM changes JOIN sold ON sold.y = year(change_date) GROUP BY 1 ORDER BY 1
"""))
show("Busiest change dates (a repricing day would stand out)", q("""
    SELECT change_date, strftime(change_date, '%a') AS dow, count(*) AS changes,
           round(100 * avg((pct > 0)::INT), 0) AS pct_up,
           round(100 * median(pct), 1) AS median_pct
    FROM changes GROUP BY 1 ORDER BY changes DESC LIMIT 10
"""))
show("How concentrated are change dates?", q("""
    WITH d AS (SELECT change_date, count(*) AS n FROM changes GROUP BY 1)
    SELECT count(*) AS dates_with_a_change,
           round(avg(n), 2) AS mean_changes_per_date,
           max(n) AS max_on_one_date,
           sum(n) FILTER (WHERE n >= 5) AS changes_on_dates_with_5_plus
    FROM d
"""))

# Same as above, but only for fast movers (sold most days), where the first
# new-price sale is a good proxy for the true repricing date.
show("Fast movers only (gap <= 2 days): busiest change dates", q("""
    SELECT change_date, count(*) AS changes, round(100 * median(pct), 1) AS median_pct
    FROM changes WHERE change_date - last_old_date <= 2
    GROUP BY 1 ORDER BY 2 DESC LIMIT 8
"""))
# Do price changes follow deliveries? Share of changes with a purchase of
# that drug in the 14 days up to the change, vs the same share on mid-spell
# dates with no change (base rate).
con.sql(f"""
    CREATE TABLE purchases AS
    SELECT DISTINCT DrugCode, Date FROM read_csv_auto('{STOCK}')
    WHERE TransactionType LIKE 'Pembelian%'
""")
show("Purchase in the 14 days before: price changes vs no-change dates", q("""
    WITH pts AS (
        SELECT 'price change' AS kind, DrugCode, change_date AS d FROM changes
        UNION ALL
        SELECT 'mid-spell, no change', DrugCode,
               start_date + ((end_date - start_date) // 2)::INT
        FROM spells WHERE end_date - start_date > 60
    )
    SELECT kind, count(*) AS dates,
           round(100 * avg((EXISTS (SELECT 1 FROM purchases p WHERE p.DrugCode = pts.DrugCode
                   AND p.Date BETWEEN pts.d - 14 AND pts.d))::INT), 1) AS pct_with_purchase
    FROM pts GROUP BY 1
"""))
show("Changes by month of year", q("""
    SELECT month(change_date) AS month, count(*) AS changes,
           round(100 * avg((pct > 0)::INT), 0) AS pct_up
    FROM changes GROUP BY 1 ORDER BY 1
"""))

# Inflation: a fixed-basket (Laspeyres) shelf-price index. Basket = drug+units
# sold in 2023, weighted by 2023 revenue. Each month uses the price in force on
# the 15th (latest real spell; before the first sale, the first price).
spells = q("SELECT DrugCode, Unit, price, start_date FROM spells ORDER BY start_date")
spells["start_date"] = pd.to_datetime(spells["start_date"]).astype("datetime64[ns]")
weights = q("""
    SELECT DrugCode, Unit, sum(Subtotal) AS w FROM sales WHERE Year = 2023 GROUP BY ALL
""")
months = pd.date_range("2023-01-01", "2025-12-01", freq="MS") + pd.Timedelta(days=14)
grid = weights[["DrugCode", "Unit"]].merge(
    pd.DataFrame({"on_date": months.astype("datetime64[ns]")}), how="cross")
grid = grid.sort_values("on_date")
priced = pd.merge_asof(grid, spells, left_on="on_date", right_on="start_date",
                       by=["DrugCode", "Unit"], direction="backward")
first = spells.groupby(["DrugCode", "Unit"]).price.first().rename("first_price")
priced = priced.merge(first, on=["DrugCode", "Unit"])
priced["price"] = priced["price"].fillna(priced["first_price"])
base = priced[priced.on_date == months[0]].set_index(["DrugCode", "Unit"]).price.rename("p0")
priced = priced.join(base, on=["DrugCode", "Unit"]).merge(weights, on=["DrugCode", "Unit"])
priced["rel"] = priced["price"] / priced["p0"]
index = (priced.groupby("on_date")
         .apply(lambda g: 100 * np.average(g.rel, weights=g.w), include_groups=False)
         .rename("index"))
idx = index.reset_index()
idx["month"] = idx.on_date.dt.strftime("%Y-%m")
print("\n--- Fixed-basket shelf-price index (Jan 2023 = 100, 2023 revenue weights)")
print(idx.iloc[::3][["month", "index"]].to_string(index=False))
for y in (2023, 2024, 2025):
    jan = index[index.index.year == y].iloc[0]
    dec = index[index.index.year == y].iloc[-1]
    print(f"{y}: Jan {jan:.1f} -> Dec {dec:.1f} ({100 * (dec / jan - 1):+.1f}% within year)")
print(f"Jan 2023 -> Dec 2025: {index.iloc[-1] - 100:+.1f}%  "
      "(Indonesian CPI ran in the low single digits a year over this period)")

show("Year-average price index by category (quantity-weighted list price, "
     "drug+units sold in all three years, 2023 = 100)", q("""
    WITH p AS (
        SELECT DrugCode, Unit, Category, Year,
               sum(Quantity * UnitPrice) / sum(Quantity) AS p,
               sum(Subtotal) FILTER (WHERE Year = 2023) AS r23
        FROM sales GROUP BY ALL
    ),
    w AS (SELECT DrugCode, Unit, sum(r23) AS w FROM p GROUP BY ALL),
    keep AS (SELECT DrugCode, Unit FROM p GROUP BY ALL HAVING count(DISTINCT Year) = 3),
    rel AS (
        SELECT p.Category, p.Year, w.w,
               p.p / first(p.p) OVER (PARTITION BY p.DrugCode, p.Unit ORDER BY p.Year) AS rel
        FROM p JOIN keep USING (DrugCode, Unit) JOIN w USING (DrugCode, Unit)
    )
    PIVOT (SELECT coalesce(Category, 'ALL') AS Category, Year,
                  round(100 * sum(rel * w) / sum(w), 1) AS idx
           FROM rel GROUP BY GROUPING SETS ((Category, Year), (Year)))
    ON Year USING any_value(idx) ORDER BY Category
"""))

# ---------------------------------------------------------------------------
section("2. PRICE RESPONSE (first look)")
# ---------------------------------------------------------------------------
# For each real change on a drug+unit with 2+ real changes: compare weekly
# units in the 8 weeks before vs the 8 weeks after (change week dropped).
# Demand is measured as the drug's share of all store units that week, which
# takes out store-wide trend and season. Weeks the drug looked out of stock
# are dropped. A placebo runs the same test on dates with no price change.

con.sql("""
    CREATE TABLE weekly AS
    SELECT DrugCode, Unit, date_trunc('week', Date) AS wk, sum(Quantity) AS units
    FROM sales GROUP BY ALL
""")
con.sql("""
    CREATE TABLE store_weekly AS
    SELECT date_trunc('week', Date) AS wk, sum(Quantity) AS store_units
    FROM sales GROUP BY 1
""")

# Rough in-stock signal from the processed ledger (it merges the shop floor
# and the store room, so this is only approximate): last balance of each day,
# carried forward; a week is out of stock if any day ends at or below zero.
con.sql(f"""
    CREATE TABLE stock_eod AS
    SELECT DrugCode, Date,
           arg_max(RemainingStock, Time) AS eod_stock
    FROM read_csv_auto('{STOCK}') GROUP BY ALL
""")
con.sql("""
    CREATE TABLE week_instock AS
    WITH days AS (
        SELECT d.DrugCode, cal.Date
        FROM (SELECT DrugCode, min(Date) AS f FROM stock_eod GROUP BY 1) d
        JOIN (SELECT unnest(generate_series(DATE '2023-01-02', DATE '2025-12-31',
                                            INTERVAL 1 DAY))::DATE AS Date) cal
          ON cal.Date >= d.f
    ),
    filled AS (
        SELECT days.DrugCode, days.Date,
               last_value(e.eod_stock IGNORE NULLS) OVER (
                   PARTITION BY days.DrugCode ORDER BY days.Date) AS stock
        FROM days LEFT JOIN stock_eod e USING (DrugCode, Date)
    )
    SELECT DrugCode, date_trunc('week', Date) AS wk, min(stock) > 0 AS in_stock
    FROM filled GROUP BY ALL
""")


def window_effect(events: pd.DataFrame) -> pd.DataFrame:
    """Before/after log change in a drug's share of store units around a date."""
    con.register("ev", events)
    out = q(f"""
        WITH e AS (SELECT *, date_trunc('week', event_date) AS ev_wk FROM ev),
        weeks AS (
            SELECT e.*, unnest(range(-{WINDOW_WEEKS}, {WINDOW_WEEKS} + 1)) AS k FROM e
        ),
        w AS (
            SELECT weeks.*, weeks.ev_wk + k * INTERVAL 7 DAY AS wk FROM weeks WHERE k <> 0
        ),
        j AS (
            SELECT w.event_id, w.k, coalesce(x.units, 0) AS units, s.store_units,
                   coalesce(i.in_stock, true) AS in_stock
            FROM w
            JOIN store_weekly s ON s.wk = w.wk  -- drops windows past the data
            LEFT JOIN weekly x ON x.DrugCode = w.DrugCode AND x.Unit = w.Unit AND x.wk = w.wk
            LEFT JOIN week_instock i ON i.DrugCode = w.DrugCode AND i.wk = w.wk
        )
        SELECT event_id,
               count(*) FILTER (WHERE k < 0) AS weeks_before,
               count(*) FILTER (WHERE k > 0) AS weeks_after,
               count(*) FILTER (WHERE k < 0 AND in_stock) AS instock_before,
               count(*) FILTER (WHERE k > 0 AND in_stock) AS instock_after,
               sum(units) FILTER (WHERE k < 0) AS units_before,
               sum(units / store_units) FILTER (WHERE k < 0 AND in_stock) AS sh_b,
               sum(units / store_units) FILTER (WHERE k > 0 AND in_stock) AS sh_a
        FROM j GROUP BY 1
    """)
    con.unregister("ev")
    out = events.merge(out, on="event_id")
    ok = ((out.weeks_before == WINDOW_WEEKS) & (out.weeks_after == WINDOW_WEEKS)
          & (out.units_before >= MIN_UNITS_BEFORE)
          & (out.instock_before >= MIN_INSTOCK_WEEKS)
          & (out.instock_after >= MIN_INSTOCK_WEEKS) & (out.sh_b > 0) & (out.sh_a > 0))
    out = out[ok].copy()
    # mean weekly share over in-stock weeks, after vs before
    out["dlog_q"] = np.log((out.sh_a / out.instock_after) / (out.sh_b / out.instock_before))
    return out


# Real change events with no other change within the window on either side.
events = q(f"""
    WITH c AS (
        SELECT *, count(*) OVER (PARTITION BY DrugCode, Unit) AS n_changes,
               lag(change_date) OVER w AS prev_change, lead(change_date) OVER w AS next_change
        FROM changes WINDOW w AS (PARTITION BY DrugCode, Unit ORDER BY change_date)
    )
    SELECT row_number() OVER () AS event_id, DrugCode, Unit,
           change_date AS event_date, ln(new_price / old_price) AS dlog_p
    FROM c
    WHERE n_changes >= 2
      AND new_price < 2 * old_price AND new_price > 0.5 * old_price
      AND coalesce(change_date - prev_change, 999) > {7 * WINDOW_WEEKS}
      AND coalesce(next_change - change_date, 999) > {7 * WINDOW_WEEKS}
""")
ev = window_effect(events)

# Placebo: the midpoint of each long stable spell of the same kind of drugs.
placebo = q(f"""
    SELECT 1000000 + row_number() OVER () AS event_id, DrugCode, Unit,
           start_date + ((end_date - start_date) // 2)::INT AS event_date, 0.0 AS dlog_p
    FROM spells
    WHERE (DrugCode, Unit) IN (SELECT DrugCode, Unit FROM changes
                               GROUP BY ALL HAVING count(*) >= 2)
      AND end_date - start_date > {2 * 7 * WINDOW_WEEKS + 14}
""")
pl = window_effect(placebo)


def summarise(df: pd.DataFrame, label: str) -> dict:
    return {
        "set": label,
        "events": len(df),
        "drug_units": df[["DrugCode", "Unit"]].drop_duplicates().shape[0],
        "median_price_chg_pct": 100 * np.expm1(df.dlog_p.median()),
        "median_demand_chg_pct": 100 * np.expm1(df.dlog_q.median()),
        "mean_dlog_q": df.dlog_q.mean(),
        "sd_dlog_q": df.dlog_q.std(),
        "share_demand_down": (df.dlog_q < 0).mean(),
    }


ups, downs = ev[ev.dlog_p > 0], ev[ev.dlog_p < 0]
show("Demand share change, 8 weeks after vs before (in-stock weeks)", pd.DataFrame([
    summarise(ups, "price up"), summarise(downs, "price down"), summarise(pl, "placebo"),
]))

# Slope of demand change on price change, net of the placebo's average drift,
# with a bootstrap interval. Price cuts are few and look like clearance of
# fading drugs, so the slope is also shown on price rises only.
rng = np.random.default_rng(0)
for label, d in (("all changes", ev), ("price rises only", ups)):
    x, y = d.dlog_p.to_numpy(), d.dlog_q.to_numpy() - pl.dlog_q.mean()
    slope = np.polyfit(x, y, 1)[0]
    boot = [np.polyfit(x[i], y[i], 1)[0]
            for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
    print(f"\nImplied elasticity, {label} (slope of dlog demand on dlog price, "
          f"placebo-adjusted): {slope:.2f}  (bootstrap 90% CI "
          f"{np.percentile(boot, 5):.2f} to {np.percentile(boot, 95):.2f})")
diff_up = ups.dlog_q.mean() - pl.dlog_q.mean()
se_up = np.sqrt(ups.dlog_q.var() / len(ups) + pl.dlog_q.var() / len(pl))
print(f"Price-up events vs placebo, mean dlog demand: {diff_up:+.3f} "
      f"(se {se_up:.3f}, z {diff_up / se_up:+.1f})")
print("Price-up events by size of rise:")
bins = pd.cut(np.expm1(ups.dlog_p) * 100, [0, 5, 10, 20, 100])
print(ups.groupby(bins, observed=True).dlog_q
      .agg(events="size", median_dlog_q="median", mean_dlog_q="mean").to_string())

# ---------------------------------------------------------------------------
section("3. REVENUE")
# ---------------------------------------------------------------------------

show("By year", q("""
    SELECT Year, round(sum(Subtotal) / 1e9, 3) AS revenue_bn,
           count(DISTINCT InvoiceNumber) AS invoices,
           count(DISTINCT Date) AS trading_days,
           round(count(DISTINCT InvoiceNumber) / count(DISTINCT Date), 1) AS invoices_per_day,
           round(count(*) / count(DISTINCT InvoiceNumber), 3) AS lines_per_invoice,
           round(sum(Subtotal) / count(DISTINCT InvoiceNumber)) AS rev_per_invoice,
           round(sum(Subtotal) / count(*)) AS rev_per_line
    FROM sales GROUP BY 1 ORDER BY 1
"""))

show("Revenue per invoice: distribution by year (IDR)", q("""
    WITH i AS (SELECT InvoiceNumber, any_value(Year) AS Year, sum(Subtotal) AS r, count(*) AS n
               FROM sales GROUP BY 1)
    SELECT Year, round(median(r)) AS median, round(quantile_cont(r, 0.9)) AS p90,
           round(avg(r)) AS mean, round(100 * avg((n = 1)::INT), 1) AS pct_single_line
    FROM i GROUP BY 1 ORDER BY 1
"""))

show("By category (revenue in m IDR)", q("""
    PIVOT (SELECT Category, Year, round(sum(Subtotal) / 1e6) AS rev FROM sales GROUP BY ALL)
    ON Year USING any_value(rev) ORDER BY "2023" DESC
"""))
show("Category share of revenue (%)", q("""
    WITH c AS (SELECT Category, Year, sum(Subtotal) AS r FROM sales GROUP BY ALL)
    PIVOT (SELECT Category, Year, round(100 * r / sum(r) OVER (PARTITION BY Year), 1) AS sh
           FROM c)
    ON Year USING any_value(sh) ORDER BY "2023" DESC
"""))

show("Top 15 manufacturers ('-' = not recorded)", q("""
    WITH m AS (SELECT Manufacturer, Year, sum(Subtotal) AS r FROM sales GROUP BY ALL)
    PIVOT (SELECT Manufacturer, Year, round(r / 1e6) AS rev_m FROM m)
    ON Year USING any_value(rev_m)
    ORDER BY coalesce("2023", 0) + coalesce("2024", 0) + coalesce("2025", 0) DESC LIMIT 15
"""))
show("Manufacturer field coverage", q("""
    SELECT round(100 * sum(Subtotal) FILTER (WHERE Manufacturer = '-') / sum(Subtotal), 1)
               AS pct_revenue_no_manufacturer,
           count(DISTINCT Manufacturer) AS manufacturers
    FROM sales
"""))

show("By DrugClass (revenue in m IDR)", q("""
    PIVOT (SELECT DrugClass, Year, round(sum(Subtotal) / 1e6) AS rev FROM sales GROUP BY ALL)
    ON Year USING any_value(rev) ORDER BY "2023" DESC NULLS LAST
"""))

show("By payment type", q("""
    SELECT PaymentType, Year, count(DISTINCT InvoiceNumber) AS invoices,
           round(sum(Subtotal) / 1e6) AS rev_m,
           round(sum(Subtotal) / count(DISTINCT InvoiceNumber)) AS rev_per_invoice
    FROM sales GROUP BY ALL ORDER BY 1, 2
"""))

# Discounts. Cashier_003 is a near-dormant login whose lines are almost all
# discounted (bulk sales, mostly May 2024), which explains the 2024 bump.
# Line discounts (DiscountAmount) are inside Subtotal. Invoice
# discounts (InvoiceDiscountPct) only show in TotalPrice, which is
# Subtotal x (1 - pct) on those rows.
show("Line discounts by year", q("""
    SELECT Year, count(*) FILTER (WHERE DiscountAmount > 0) AS lines_discounted,
           round(100 * avg((DiscountAmount > 0)::INT), 2) AS pct_lines,
           round(sum(DiscountAmount) / 1e6, 1) AS discount_m,
           round(100 * sum(DiscountAmount) / sum(Quantity * UnitPrice), 2) AS pct_of_gross,
           round(median(DiscountPct) FILTER (WHERE DiscountAmount > 0), 1) AS median_pct
    FROM sales GROUP BY 1 ORDER BY 1
"""))
show("Line discounts by category", q("""
    SELECT Category, count(*) FILTER (WHERE DiscountAmount > 0) AS lines_discounted,
           round(100 * avg((DiscountAmount > 0)::INT), 2) AS pct_lines,
           round(sum(DiscountAmount) / 1e6, 1) AS discount_m
    FROM sales GROUP BY 1 ORDER BY discount_m DESC
"""))
show("Discount % used", q("""
    SELECT DiscountPct, count(*) AS lines FROM sales WHERE DiscountAmount > 0
    GROUP BY 1 ORDER BY 2 DESC LIMIT 8
"""))
show("Who gives line discounts (cashier)", q("""
    SELECT CashierID, count(*) AS lines,
           count(*) FILTER (WHERE DiscountAmount > 0) AS discounted,
           round(100 * avg((DiscountAmount > 0)::INT), 2) AS pct_lines,
           round(sum(DiscountAmount) / 1e6, 2) AS discount_m
    FROM sales GROUP BY 1 HAVING count(*) FILTER (WHERE DiscountAmount > 0) > 0
    ORDER BY discounted DESC LIMIT 10
"""))
show("Most-discounted drugs", q("""
    SELECT DrugName, Unit, count(*) FILTER (WHERE DiscountAmount > 0) AS discounted_lines,
           count(*) AS lines, round(sum(DiscountAmount) / 1e6, 2) AS discount_m
    FROM sales GROUP BY ALL ORDER BY discounted_lines DESC LIMIT 8
"""))
show("Line discounts by month (lines discounted)", q("""
    PIVOT (SELECT month(Date) AS m, Year, count(*) FILTER (WHERE DiscountAmount > 0) AS n
           FROM sales GROUP BY ALL)
    ON Year USING any_value(n) ORDER BY m
"""))
show("Invoice-level discounts and fees", q("""
    SELECT count(DISTINCT InvoiceNumber) FILTER (WHERE InvoiceDiscountPct > 0) AS invoices_with_inv_discount,
           round(sum(Subtotal - TotalPrice)) AS inv_discount_idr,
           count(*) FILTER (WHERE ServiceFee > 0) AS lines_service_fee,
           round(sum(ServiceFee)) AS service_fee_idr,
           count(*) FILTER (WHERE PackagingFee > 0) AS lines_packaging_fee,
           round(sum(PackagingFee)) AS packaging_fee_idr
    FROM sales
"""))

# ---------------------------------------------------------------------------
section("4. WHY REVENUE DIPPED IN 2024 AND RECOVERED IN 2025")
# ---------------------------------------------------------------------------

# (a) Traffic vs basket: revenue = invoices x lines/invoice x revenue/line.
# Log shares add up to the total log change.
yr = q("""
    SELECT Year, sum(Subtotal) AS rev, count(DISTINCT InvoiceNumber) AS inv,
           count(*) AS lines, sum(Quantity) AS units
    FROM sales GROUP BY 1 ORDER BY 1
""").set_index("Year")
rows = []
for a, b in ((2023, 2024), (2024, 2025)):
    r0, r1 = yr.loc[a], yr.loc[b]
    parts = {
        "invoices": np.log(r1.inv / r0.inv),
        "lines_per_invoice": np.log((r1.lines / r1.inv) / (r0.lines / r0.inv)),
        "rev_per_line": np.log((r1.rev / r1.lines) / (r0.rev / r0.lines)),
    }
    rows.append({"change": f"{a}->{b}", "revenue_pct": 100 * (r1.rev / r0.rev - 1),
                 **{k: 100 * v for k, v in parts.items()}})
show("(a) Revenue change split into traffic and basket (log-points x100)", pd.DataFrame(rows))

show("Invoices by quarter (traffic trend)", q("""
    SELECT Year, quarter(Date) AS q, count(DISTINCT InvoiceNumber) AS invoices,
           round(sum(Subtotal) / count(DISTINCT InvoiceNumber)) AS rev_per_invoice
    FROM sales GROUP BY ALL ORDER BY ALL
"""))

# (b) Price vs volume vs range, on drug+unit level. For items sold in both
# years: price effect = sum q1 (p1 - p0), volume effect = sum p0 (q1 - q0),
# with p = revenue per unit. Items only in one year count as new / dropped.
dy = q("""
    SELECT DrugCode, Unit, Year, sum(Subtotal) AS rev, sum(Quantity) AS qty
    FROM sales GROUP BY ALL
""")
rows = []
for a, b in ((2023, 2024), (2024, 2025)):
    m = (dy[dy.Year == a].merge(dy[dy.Year == b], on=["DrugCode", "Unit"], how="outer",
                                suffixes=("0", "1")).fillna(0))
    both = m[(m.qty0 > 0) & (m.qty1 > 0)]
    p0, p1 = both.rev0 / both.qty0, both.rev1 / both.qty1
    rows.append({
        "change": f"{a}->{b}",
        "total_m": (m.rev1.sum() - m.rev0.sum()) / 1e6,
        "price_m": (both.qty1 * (p1 - p0)).sum() / 1e6,
        "volume_m": (p0 * (both.qty1 - both.qty0)).sum() / 1e6,
        "new_items_m": m.loc[m.qty0 == 0, "rev1"].sum() / 1e6,
        "dropped_items_m": -m.loc[m.qty1 == 0, "rev0"].sum() / 1e6,
        "items_both": len(both),
        "price_index_pct": 100 * ((both.qty1 * p1).sum() / (both.qty1 * p0).sum() - 1),
    })
show("(b) Revenue change: price, volume, new and dropped items (m IDR)", pd.DataFrame(rows))

show("(c) Contribution to the change by category (m IDR)", q("""
    WITH c AS (SELECT Category, Year, sum(Subtotal) / 1e6 AS r FROM sales GROUP BY ALL),
    p AS (PIVOT c ON Year USING any_value(r))
    SELECT Category, round(coalesce("2024", 0) - coalesce("2023", 0)) AS d_2023_24,
           round(coalesce("2025", 0) - coalesce("2024", 0)) AS d_2024_25
    FROM p ORDER BY d_2023_24
"""))

# (d) Which drugs drove the 2024 volume drop? Top losers 2023->2024 and how
# they did in 2025.
show("(d) Biggest revenue losers 2023->2024 (m IDR)", q("""
    WITH d AS (
        SELECT DrugCode, any_value(DrugName) AS name, any_value(Category) AS cat,
               sum(Subtotal) FILTER (WHERE Year = 2023) / 1e6 AS r23,
               sum(Subtotal) FILTER (WHERE Year = 2024) / 1e6 AS r24,
               sum(Subtotal) FILTER (WHERE Year = 2025) / 1e6 AS r25
        FROM sales GROUP BY 1
    )
    SELECT name, cat, round(r23, 1) AS r23, round(coalesce(r24, 0), 1) AS r24,
           round(coalesce(r25, 0), 1) AS r25
    FROM d ORDER BY coalesce(r24, 0) - coalesce(r23, 0) LIMIT 12
"""))
show("(e) How concentrated is the 2023->2024 loss?", q("""
    WITH d AS (
        SELECT DrugCode,
               coalesce(sum(Subtotal) FILTER (WHERE Year = 2024), 0)
             - coalesce(sum(Subtotal) FILTER (WHERE Year = 2023), 0) AS delta
        FROM sales GROUP BY 1
    ),
    r AS (SELECT delta, row_number() OVER (ORDER BY delta) AS rk FROM d)
    SELECT round(sum(delta) / 1e6) AS net_change_m,
           round(sum(delta) FILTER (WHERE delta < 0) / 1e6) AS all_losers_m,
           round(sum(delta) FILTER (WHERE delta > 0) / 1e6) AS all_gainers_m,
           round(sum(delta) FILTER (WHERE rk <= 20) / 1e6) AS top20_losers_m,
           count(*) FILTER (WHERE delta < 0) AS n_losers,
           count(*) FILTER (WHERE delta > 0) AS n_gainers
    FROM r
"""))
