"""EDA route 3: demand and time patterns.

Trend, weekday, hour, shift, Ramadan and Eid, illness seasons, concentration,
demand classes (ADI / CV², Syntetos-Boylan), baskets and drug life cycle.

Run from the repo root:
    uvx --with duckdb --with pandas python -I eda/demand.py

Reads data/raw/irpd/ and prints results. Writes nothing.
"""

import time
from pathlib import Path

import duckdb
import pandas as pd

T0 = time.time()
DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "irpd"
SALES = DATA / "processed_dataset" / "sales_processed.csv"
STOCK = DATA / "processed_dataset" / "stock_processed.csv"
STOCK_RAW = DATA / "raw_dataset" / "stock_raw.csv"

CUTOFF = "2025-06-30"  # provisional replay cutoff from PLAN.md

# Ramadan and Eid al-Fitr dates as set by Indonesia's Ministry of Religious
# Affairs (Kemenag). Eid day itself is one of the shop's closed days.
# 1444 H: Ramadan 23 Mar 2023, Eid 22 Apr 2023
# 1445 H: Ramadan 12 Mar 2024, Eid 10 Apr 2024
# 1446 H: Ramadan  1 Mar 2025, Eid 31 Mar 2025
RAMADAN = {2023: ("2023-03-23", "2023-04-22"),
           2024: ("2024-03-12", "2024-04-10"),
           2025: ("2025-03-01", "2025-03-31")}  # (first day of fasting, Eid)

# Syntetos-Boylan cut-offs
ADI_CUT, CV2_CUT = 1.32, 0.49

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

con = duckdb.connect()
q = lambda sql: con.sql(sql).df()  # noqa: E731


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(df, index=False):
    print(df.to_string(index=index))


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
con.sql(f"""
    CREATE TABLE sales AS
    SELECT Date AS d, Year AS yr, Month AS mo, DayOfWeek AS dow, Hour AS hr,
           InvoiceNumber AS inv, ShiftPeriod AS shift, DrugCode AS drug,
           DrugName AS name, Category AS cat, Quantity AS qty, Unit AS unit,
           Subtotal AS rev
    FROM read_csv_auto('{SALES}')
""")
# Trading days = days with any sale (the shop is closed on 8 days).
con.sql("""
    CREATE TABLE cal AS
    SELECT d, date_trunc('week', d)::DATE AS wk, row_number() OVER (ORDER BY d) AS di
    FROM (SELECT DISTINCT d FROM sales)
""")
con.sql("""
    CREATE TABLE drug_day AS
    SELECT drug, d, sum(qty) AS qty, sum(rev) AS rev, count(*) AS lines
    FROM sales GROUP BY ALL
""")

# End-of-day stock per drug. The processed stock file merges two storage
# locations into one running column, so we take the location (Warehouse)
# from the raw file by RowNumber, keep the last balance per location per day,
# carry it forward, and sum the locations.
con.sql(f"""
    CREATE TABLE mv AS
    SELECT k.RowNumber AS rn, k.Date AS d, k.DrugCode AS drug, r.Warehouse AS loc,
           k.InitialStock AS opening, k.RemainingStock AS closing
    FROM read_csv_auto('{STOCK}') k
    JOIN read_csv_auto('{STOCK_RAW}') r USING (RowNumber)
""")
con.sql("""
    CREATE TABLE eod AS
    WITH loc_day AS (
        SELECT drug, loc, d, arg_max(closing, rn) AS closing FROM mv GROUP BY ALL
    ), first_open AS (  -- stock before a location's first movement
        SELECT drug, loc, arg_min(opening, rn) AS opening FROM mv GROUP BY ALL
    ), grid AS (
        SELECT f.drug, f.loc, f.opening, c.d
        FROM first_open f CROSS JOIN (SELECT DISTINCT d FROM cal) c
    ), filled AS (
        SELECT g.drug, g.d,
               coalesce(last_value(l.closing IGNORE NULLS) OVER (
                   PARTITION BY g.drug, g.loc ORDER BY g.d
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), g.opening) AS stock
        FROM grid g LEFT JOIN loc_day l USING (drug, loc, d)
    )
    SELECT drug, d, sum(stock) AS stock FROM filled GROUP BY ALL
""")
# A drug-day counts as in stock if it opened with stock (previous trading
# day's close), closed with stock, or sold anything. Drugs that never appear
# in the ledger have no in-stock information.
con.sql("""
    CREATE TABLE avail AS
    SELECT e.drug, e.d,
           (coalesce(lag(e.stock) OVER (PARTITION BY e.drug ORDER BY e.d), e.stock) > 0
            OR e.stock > 0 OR coalesce(s.qty, 0) > 0) AS in_stock
    FROM eod e LEFT JOIN drug_day s USING (drug, d)
""")
print(f"Loaded in {time.time() - T0:.0f}s")

# --------------------------------------------------------------------------
# 1. Trend
# --------------------------------------------------------------------------
section("1. Trend 2023-2025")
yearly = q("""
    SELECT yr, count(DISTINCT d) AS trading_days, count(DISTINCT inv) AS invoices,
           count(*) AS lines, sum(qty)::BIGINT AS units, sum(rev) / 1e9 AS rev_bn,
           count(DISTINCT drug) AS drugs_sold
    FROM sales GROUP BY yr ORDER BY yr
""")
yearly["inv_per_day"] = yearly.invoices / yearly.trading_days
yearly["rev_per_inv"] = yearly.rev_bn * 1e9 / yearly.invoices
yearly["units_per_line"] = yearly.units / yearly.lines
show(yearly)

monthly = q("""
    SELECT yr, mo, count(DISTINCT d) AS days, count(DISTINCT inv) AS invoices,
           sum(qty)::BIGINT AS units, sum(rev) / 1e6 AS rev_m
    FROM sales GROUP BY ALL ORDER BY ALL
""")
monthly["inv_per_day"] = monthly.invoices / monthly.days
print("\nInvoices per trading day, by month:")
show(monthly.pivot(index="mo", columns="yr", values="inv_per_day"), index=True)
print("\nRevenue per month (million IDR):")
show(monthly.pivot(index="mo", columns="yr", values="rev_m"), index=True)
print("\nUnits per month:")
show(monthly.pivot(index="mo", columns="yr", values="units"), index=True)

# --------------------------------------------------------------------------
# 2. Weekday, hour, shift
# --------------------------------------------------------------------------
section("2. Weekday, hour, shift")
wd = q("""
    WITH daily AS (
        SELECT d, any_value(dow) AS dow, isodow(d) AS n, count(DISTINCT inv) AS inv,
               sum(rev) AS rev
        FROM sales GROUP BY d
    )
    SELECT dow, count(*) AS days, avg(inv) AS inv_per_day, avg(rev) / 1e6 AS rev_m_per_day
    FROM daily GROUP BY dow, n ORDER BY n
""")
wd["index_vs_mean"] = wd.inv_per_day / wd.inv_per_day.mean()
show(wd)

hours = q("""
    SELECT hr, count(DISTINCT inv) AS invoices, sum(rev) / 1e6 AS rev_m
    FROM sales GROUP BY hr ORDER BY hr
""")
hours["inv_share_%"] = 100 * hours.invoices / hours.invoices.sum()
print("\nBy hour:")
show(hours)

shifts = q("""
    SELECT shift, count(DISTINCT inv) AS invoices, sum(rev) / 1e9 AS rev_bn,
           quantile_cont(hr, 0.05) AS hr_p05, median(hr) AS hr_median,
           quantile_cont(hr, 0.95) AS hr_p95
    FROM sales GROUP BY shift ORDER BY shift
""")
print("\nBy shift (PAGI = morning, SORE = afternoon/evening):")
show(shifts)
print("Shift is the cashier's shift label, so its hours overlap; use Hour for time of day.")

# --------------------------------------------------------------------------
# 3. Ramadan and Eid
# --------------------------------------------------------------------------
section("3. Ramadan and Eid al-Fitr")
# Each day's invoices divided by the mean for the same weekday in the same
# year, so weekday mix doesn't bias the windows.
daily = q("""
    SELECT d, year(d) AS yr, isodow(d) AS n, count(DISTINCT inv) AS inv, sum(rev) AS rev
    FROM sales GROUP BY d ORDER BY d
""")
daily["d"] = pd.to_datetime(daily.d)
base = daily.groupby(["yr", "n"])[["inv", "rev"]].transform("mean")
daily["inv_idx"] = daily.inv / base.inv
daily["rev_idx"] = daily.rev / base.rev

rows = []
for yr, (start, eid) in RAMADAN.items():
    start, eid = pd.Timestamp(start), pd.Timestamp(eid)
    windows = {
        "4 wks before Ramadan": (start - pd.Timedelta(days=28), start - pd.Timedelta(days=1)),
        "Ramadan, first 3 wks": (start, start + pd.Timedelta(days=20)),
        "last 7 days before Eid": (eid - pd.Timedelta(days=7), eid - pd.Timedelta(days=1)),
        "Eid +1 to +7": (eid + pd.Timedelta(days=1), eid + pd.Timedelta(days=7)),
        "Eid +8 to +28": (eid + pd.Timedelta(days=8), eid + pd.Timedelta(days=28)),
    }
    for label, (a, b) in windows.items():
        w = daily[(daily.d >= a) & (daily.d <= b)]
        rows.append({"year": yr, "window": label, "days": len(w),
                     "inv_idx": w.inv_idx.mean(), "rev_idx": w.rev_idx.mean()})
ram = pd.DataFrame(rows)
print("Index = invoices (or revenue) / same-weekday mean of that year. 1.00 = normal.")
show(ram.pivot(index="window", columns="year", values="inv_idx")
        .reindex([r["window"] for r in rows[:5]]), index=True)
print("\nRevenue index:")
show(ram.pivot(index="window", columns="year", values="rev_idx")
        .reindex([r["window"] for r in rows[:5]]), index=True)
print("\nDays around Eid (invoice index, day 0 = Eid, closed):")
around = []
for yr, (_, eid) in RAMADAN.items():
    eid = pd.Timestamp(eid)
    for k in range(-5, 6):
        hit = daily[daily.d == eid + pd.Timedelta(days=k)]
        around.append({"year": yr, "day": k,
                       "idx": hit.inv_idx.iloc[0] if len(hit) else float("nan")})
show(pd.DataFrame(around).pivot(index="year", columns="day", values="idx"), index=True)

# --------------------------------------------------------------------------
# 4. Illness seasons
# --------------------------------------------------------------------------
section("4. Illness seasons (name-based drug groups)")
# Rough groups from drug names (brands common in Indonesia). Good enough to
# see a seasonal shape; the ingredient mapping in pillar 3 will replace this.
GROUPS = {
    "cough_cold_flu": r"flu|grip|batuk|obh|paratusin|neozep|decolgen|procold|inza|"
                      r"woods|siladex|vicks|bisolvon|ambroxol|anacetin|alpara|lapisiv|"
                      r"demacolin|fg troches|alba pastil|tolak angin|actifed|lapifed|"
                      r"tremenza|rhinos|intunal|grantusif|dextamine|tuzalos|konidin",
    "fever_paracetamol": r"paracetamol|zetamol|sanmol|panadol|paramex|bodrex|tempra|"
                         r"termorex|fasidol|bye bye fever|proris|praxion|pamol",
    "diarrhoea": r"oralit|entrostop|diapet|diatabs|imodium|neo kaominal|lodia|"
                 r"new diatab|zinc|norit",
    "allergy_itch": r"cetirizin|loratadin|ctm|chlorphen|incidal|interhistin|caladine|"
                    r"cerini|alleron",
}
case = " ".join(f"WHEN regexp_matches(lower(name), '{p}') THEN '{g}'"
                for g, p in GROUPS.items())
con.sql(f"CREATE TABLE grp AS SELECT DISTINCT drug, CASE {case} ELSE 'other' END AS g FROM sales")
print(q("""
    SELECT g, count(*) AS drugs, sum(s.qty)::BIGINT AS units, sum(s.rev) / 1e9 AS rev_bn
    FROM grp JOIN (SELECT drug, sum(qty) qty, sum(rev) rev FROM sales GROUP BY 1) s USING (drug)
    GROUP BY g ORDER BY units DESC
""").to_string(index=False))

# Monthly seasonal index: units per trading day in the month divided by the
# group's mean for that year, then averaged over the three years.
season = q("""
    SELECT g, s.yr, s.mo, sum(qty) / count(DISTINCT s.d) AS upd
    FROM sales s JOIN grp USING (drug) GROUP BY ALL
    UNION ALL
    SELECT 'ALL' AS g, yr, mo, sum(qty) / count(DISTINCT d) FROM sales GROUP BY ALL
""")
season["idx"] = season.upd / season.groupby(["g", "yr"]).upd.transform("mean")
avg_idx = season.pivot_table(index="mo", columns="g", values="idx", aggfunc="mean")
print("\nSeasonal index by month (mean of 2023-25, 1.00 = that group's yearly average):")
show(avg_idx, index=True)
print("\nCough/cold/flu index by year (does the shape repeat?):")
cold = season[season.g == "cough_cold_flu"].pivot(index="mo", columns="yr", values="idx")
show(cold, index=True)
print("Year-to-year correlation of the cough/cold monthly shape:")
show(cold.corr(), index=True)
print("Relative to all drugs (group index / ALL index, mean of years):")
rel = avg_idx.drop(columns="ALL").div(avg_idx["ALL"], axis=0)
show(rel, index=True)

# --------------------------------------------------------------------------
# 5. Concentration
# --------------------------------------------------------------------------
section("5. Concentration (Pareto)")
per_drug = q("""
    SELECT drug, any_value(name) AS name, sum(qty) AS units, sum(rev) AS rev,
           count(DISTINCT d) AS sale_days
    FROM sales GROUP BY drug
""")
n = len(per_drug)
out = []
for col in ["units", "rev"]:
    v = per_drug[col].sort_values(ascending=False).reset_index(drop=True)
    cum = v.cumsum() / v.sum()
    row = {"measure": col}
    for p in [1, 5, 10, 20, 50]:
        row[f"top{p}%"] = 100 * cum.iloc[max(int(n * p / 100) - 1, 0)]
    for share in [0.5, 0.8, 0.95]:
        row[f"drugs_for_{int(share * 100)}%"] = int((cum < share).sum() + 1)
    out.append(row)
print(f"{n} drugs sold. Share of total held by the top x% of drugs:")
show(pd.DataFrame(out))
top_u = set(per_drug.nlargest(int(n * 0.1), "units").drug)
top_r = set(per_drug.nlargest(int(n * 0.1), "rev").drug)
print(f"Overlap of top-10% by units and top-10% by revenue: "
      f"{len(top_u & top_r)} of {len(top_u)} drugs")
print(f"Drugs with a single sale line in 3 years: "
      f"{(per_drug.sale_days == 1).sum()} ({100 * (per_drug.sale_days == 1).mean():.1f}%)")
print("Note: units mix tablets, bottles, tubes etc., so revenue is the fairer weight.")
print("\nTop 15 by revenue:")
show(per_drug.nlargest(15, "rev")[["name", "units", "rev", "sale_days"]]
     .assign(rev=lambda x: x.rev / 1e6).rename(columns={"rev": "rev_m"}))

# --------------------------------------------------------------------------
# 6. Demand classes (Syntetos-Boylan)
# --------------------------------------------------------------------------
section("6. Demand classes: ADI and CV² (Syntetos-Boylan)")
print(f"""ADI = periods in the drug's active window / periods with a sale.
Active window = first to last sale. Periods are trading days (closed days
dropped) or Monday weeks. 'In-stock' ADI counts only in-stock periods in the
window: a day is in stock if it opened or closed with stock or had a sale; a
week if it had a sale or was in stock on at least half its trading days.
CV² = (sd / mean)² of non-zero period demand (sample sd). Stock-outs only
change ADI, since non-zero sizes are the same. Cut-offs ADI {ADI_CUT}, CV² {CV2_CUT}.
Drugs with fewer than 3 non-zero periods are 'too few'.""")

# Every trading day in each drug's active window, with sales and stock flag.
con.sql("""
    CREATE TABLE win AS
    WITH span AS (SELECT drug, min(d) AS d0, max(d) AS d1 FROM drug_day GROUP BY drug)
    SELECT sp.drug, c.d, c.wk, coalesce(s.qty, 0) AS qty, s.rev,
           a.in_stock  -- NULL for drugs missing from the ledger
    FROM span sp JOIN cal c ON c.d BETWEEN sp.d0 AND sp.d1
    LEFT JOIN drug_day s ON s.drug = sp.drug AND s.d = c.d
    LEFT JOIN avail a ON a.drug = sp.drug AND a.d = c.d
""")
oos = q("""
    WITH top AS (SELECT drug FROM drug_day GROUP BY drug ORDER BY sum(rev) DESC LIMIT 300)
    SELECT count(*) AS drug_days, avg((NOT in_stock)::INT) * 100 AS pct_out_of_stock,
           avg((NOT in_stock)::INT) FILTER (WHERE drug IN (SELECT drug FROM top)) * 100
               AS pct_oos_top300_rev,
           count(DISTINCT drug) FILTER (WHERE in_stock IS NULL) AS drugs_no_ledger
    FROM win
""")
print("\nWithin active windows:")
show(oos)


def classify(adi, cv2, nz):
    if nz < 3:
        return "too few"
    if adi < ADI_CUT:
        return "smooth" if cv2 < CV2_CUT else "erratic"
    return "intermittent" if cv2 < CV2_CUT else "lumpy"


def sb_classes(grain):
    """Per-drug ADI (all / in-stock), CV² and class at day or week grain."""
    if grain == "day":
        per = q("SELECT drug, d AS p, qty, rev, in_stock FROM win")
    else:
        per = q("""
            SELECT drug, wk AS p, sum(qty) AS qty, sum(rev) AS rev,
                   bool_or(qty > 0) OR avg(in_stock::INT) >= 0.5 AS in_stock
            FROM win GROUP BY drug, wk
        """)
    per["nz"] = per.qty > 0
    g = per.groupby("drug")
    stats = pd.DataFrame({
        "periods": g.size(),
        "in_stock_periods": g.in_stock.sum(min_count=1),
        "nz": g.nz.sum(),
        "units": g.qty.sum(),
        "rev": g.rev.sum(),
    })
    nzq = per[per.nz].groupby("drug").qty
    stats["cv2"] = (nzq.std(ddof=1) / nzq.mean()) ** 2
    stats["adi_all"] = stats.periods / stats.nz
    stats["adi_instock"] = stats.in_stock_periods / stats.nz
    stats["class_all"] = [classify(a, c, z) for a, c, z in
                          zip(stats.adi_all, stats.cv2.fillna(0), stats.nz)]
    stats["class_instock"] = [classify(a, c, z) if pd.notna(a) else "no ledger"
                              for a, c, z in zip(stats.adi_instock, stats.cv2.fillna(0), stats.nz)]
    return stats


ORDER = ["smooth", "erratic", "intermittent", "lumpy", "too few", "no ledger"]
classes = {}
for grain in ["day", "week"]:
    st = sb_classes(grain)
    classes[grain] = st
    print(f"\n--- {grain.upper()} grain ---")
    summ = pd.DataFrame({
        "drugs_all": st.class_all.value_counts(),
        "drugs_instock": st.class_instock.value_counts(),
        "rev%_all": 100 * st.groupby("class_all").rev.sum() / st.rev.sum(),
        "rev%_instock": 100 * st.groupby("class_instock").rev.sum() / st.rev.sum(),
    }).reindex(ORDER).fillna(0)
    show(summ, index=True)
    ok = st.class_instock != "no ledger"
    moved = (st.class_all != st.class_instock) & ok
    print(f"Drugs whose class changes when stock-out periods are dropped: {moved.sum()} "
          f"of {ok.sum()} ({100 * moved.sum() / ok.sum():.1f}%), "
          f"{100 * st[moved].rev.sum() / st.rev.sum():.1f}% of revenue")
    print("Transitions (rows = all periods, columns = in-stock periods only):")
    show(pd.crosstab(st.class_all[ok], st.class_instock[ok]), index=True)
    med = st[st.nz >= 3][["adi_all", "adi_instock", "cv2"]].median()
    print(f"Median ADI all {med.adi_all:.2f}, in-stock {med.adi_instock:.2f}; "
          f"median CV² {med.cv2:.2f}")
    top = st.nlargest(300, "rev")
    print("Top 300 drugs by revenue, class (in-stock):",
          top.class_instock.value_counts().to_dict())

# --------------------------------------------------------------------------
# 7. Baskets
# --------------------------------------------------------------------------
section("7. Baskets")
bask = q("""
    SELECT CASE WHEN lines >= 6 THEN '6+' ELSE lines::VARCHAR END AS lines,
           count(*) AS invoices
    FROM (SELECT inv, count(*) AS lines FROM sales GROUP BY inv) GROUP BY 1 ORDER BY 1
""")
bask["share_%"] = 100 * bask.invoices / bask.invoices.sum()
show(bask)
print(q("""
    SELECT avg(lines) AS mean_lines, avg(drugs) AS mean_distinct_drugs, max(lines) AS max_lines,
           avg((lines > drugs)::INT) * 100 AS pct_with_repeat_drug
    FROM (SELECT inv, count(*) AS lines, count(DISTINCT drug) AS drugs FROM sales GROUP BY inv)
""").to_string(index=False))
print("Mean lines per invoice by year:")
print(q("""SELECT yr, count(*) / count(DISTINCT inv) AS lines_per_inv FROM sales
           GROUP BY yr ORDER BY yr""").to_string(index=False))

# Pairs of different drugs bought on the same invoice. Lift = how much more
# often than if the two were bought independently.
pairs = q("""
    WITH b AS (SELECT DISTINCT inv, drug FROM sales),
    n AS (SELECT count(DISTINCT inv) AS n FROM b),
    single AS (SELECT drug, count(*) AS c FROM b GROUP BY drug),
    pr AS (SELECT x.drug AS a, y.drug AS b, count(*) AS together
           FROM b x JOIN b y ON x.inv = y.inv AND x.drug < y.drug GROUP BY ALL)
    SELECT na.name AS drug_a, nb.name AS drug_b, together,
           together * n.n / (sa.c * sb.c) AS lift
    FROM pr, n
    JOIN single sa ON sa.drug = pr.a JOIN single sb ON sb.drug = pr.b
    JOIN (SELECT drug, any_value(name) AS name FROM sales GROUP BY 1) na ON na.drug = pr.a
    JOIN (SELECT drug, any_value(name) AS name FROM sales GROUP BY 1) nb ON nb.drug = pr.b
    ORDER BY together DESC LIMIT 20
""")
print("\nMost common drug pairs:")
show(pairs)

# --------------------------------------------------------------------------
# 8. Life cycle and forecastable history
# --------------------------------------------------------------------------
section("8. Drug life cycle")
life = q("""
    SELECT drug, min(d) AS first, max(d) AS last, count(*) AS sale_days,
           sum(rev) AS rev,
           count(*) FILTER (WHERE d <= DATE '2025-06-30') AS sale_days_pre,
           sum(rev) FILTER (WHERE d > DATE '2025-06-30') AS rev_post
    FROM drug_day GROUP BY drug
""")
life["first"] = pd.to_datetime(life["first"])
life["last"] = pd.to_datetime(life["last"])
# First and last 90 days are a buffer: a slow drug may simply not have sold yet.
new = life["first"] > "2023-04-01"
gone = life["last"] < "2025-10-01"
print(f"Drugs sold: {len(life)}")
print(f"  first sale after 1 Apr 2023 (introduced): {new.sum()}")
print(f"  last sale before 1 Oct 2025 (discontinued or dormant): {gone.sum()}")
print(f"  both (short-lived): {(new & gone).sum()}")
print(f"  present throughout: {(~new & ~gone).sum()}, "
      f"{100 * life[~new & ~gone].rev.sum() / life.rev.sum():.1f}% of revenue")
print("\nFirst-sale and last-sale counts by quarter:")
lc = pd.DataFrame({
    "first_sales": life["first"].dt.to_period("Q").value_counts(),
    "last_sales": life["last"].dt.to_period("Q").value_counts(),
}).sort_index().fillna(0).astype(int)
show(lc, index=True)

# Who has enough history for a forecaster at the replay cutoff?
pre_days = q(f"SELECT count(*) AS n FROM cal WHERE d <= DATE '{CUTOFF}'").n.iloc[0]
alive = life.rev_post.notna()  # still selling after the cutoff
long_hist = life["first"] <= "2024-06-30"  # a full year before the cutoff
tiers = pd.cut(life.sale_days_pre / pre_days, [-1, 1 / 30, 1 / 7, 0.5, 1.01],
               labels=["<1 sale/month", "monthly-weekly", "weekly-every other day",
                       ">=every other day"])
t = (life[alive & long_hist].assign(tier=tiers)
     .groupby("tier", observed=False)
     .agg(drugs=("drug", "size"), rev_post_m=("rev_post", lambda x: x.sum() / 1e6)))
t["share_of_post_rev_%"] = 100 * t.rev_post_m / (life.rev_post.sum() / 1e6)
print(f"\nDrugs sold after the {CUTOFF} cutoff: {alive.sum()}, "
      f"of which first sold by 30 Jun 2024 (1+ year history): {(alive & long_hist).sum()}")
print(f"Tiers by share of the {pre_days} pre-cutoff trading days with a sale "
      "(drugs with 1+ year history that sell after the cutoff):")
show(t, index=True)
short = alive & ~long_hist
print(f"Drugs selling after the cutoff with < 1 year history: {short.sum()}, "
      f"{100 * life[short].rev_post.sum() / life.rev_post.sum():.1f}% of post-cutoff revenue")

print(f"\nDone in {time.time() - T0:.0f}s")
