# IRPD analysis

An AI pharmacy analyst built on three years of one Indonesian pharmacy's sales
and full stock ledger ([IRPD](#data), CC BY 4.0). It answers questions by
calling tools over a DuckDB warehouse, and every number it states has to come
from a tool. It only sees data up to 30 June 2025. The second half of 2025 is
sealed in a file only the scorer can open, and that's where the agent competes
with the pharmacist.

A few terms used below. A stock-out is a day when the whole pharmacy has none
of a drug. A shelf-out is a day when the shop floor is empty but the upstairs
store still has some. Fast and regular drugs are the ones that sell on at
least 3 days a week or at least once a week. The agent runs on two OpenAI
models: gpt-6.1-sol, the larger one, and gpt-6-luna, about 15 times cheaper.

![The agent vs pharmacist page: headline, stock held vs stock-out days, scores and the agent's rules](docs/img/contest.gif)

## What it found

The agent beat the pharmacist's own habits. Standing on 30 June 2025, the
agent (on gpt-6.1-sol) set the reorder and shop-floor policy for the next six
months. Replayed on the sealed H2 2025 data at the same stock value (about
Rp 312M), it had 54% fewer stock-out days, 66% fewer shelf-out days and 63%
less lost revenue than a copy of the pharmacist's habits. It ties a textbook
reorder rule on lost revenue, and the shelf-out win is an upper bound because
the simulated floor has no space limit.

Rp 194M of sales were lost to stock-outs and shelf-outs from October 2023 to
the cutoff, 2.75% of revenue. A fake stock-out test checks the estimate: it
lands within 1% on fast drugs and 6% on regular ones.

Only about 30% of those lost sales moved to another brand of the same
medicine, not the 84% the first look suggested. It depends on the medicine:
nearly everyone switches for amlodipine, almost no one for paracetamol. So
Rp 163M (2.3%) was truly lost.

The agent doesn't make numbers up. Over the 91 eval questions, gpt-6-luna
stated a number that no tool produced in 2 answers. It turned down every
question about profit, margin or dates past the cutoff. At first it didn't
check false claims in a question. The last prompt revision fixed that.

The bigger model only helped with planning. On questions, gpt-6.1-sol and
gpt-6-luna both scored 90%. On the reorder policy, sol had 326 shelf-out days
to luna's 3,635.

![Live chat: the agent's tool calls stream in, then its answer; asked about August 2025, it says it can't see past the cutoff](docs/img/chat.gif)

## More detail

- [Results](docs/results.md): the contest table and its caveats, lost demand,
  switching, both eval sets, and what the data can't tell us.
- [How it works](docs/how-it-works.md): the warehouse, the sealed holdout, the
  tools, the agent loop, the forecaster, the replay, and every command.
- [FINDINGS.md](FINDINGS.md): the exploratory analysis.

## Quick start

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and Node.

```sh
uv sync
# download the data into data/raw/irpd/ (see Data below)
uv run irpd build                   # warehouse + sealed holdout, a few seconds
uv run irpd replay --uncensored     # the replay table on H2 2025
uv run uvicorn irpd.api.app:app --port 8000
cd web && npm install && npm run dev   # http://localhost:5173
```

Model calls need `OPENAI_API_KEY` in a `.env` file. Without one, the chat page
plays an example transcript.

## Data

[IRPD: Indonesian Retail Pharmacy Sales and Stock Transaction Dataset (2023–2025)](https://data.mendeley.com/datasets/w6zjtdjk9m/1),
CC BY 4.0. It is not committed. Download the zip from the Mendeley page and
unzip it so that `data/raw/irpd/raw_dataset/` and
`data/raw/irpd/processed_dataset/` exist.

Two reference drug lists are used to check the ingredient mapping. They are
not committed either.

- [Indonesian Pharmaceutical Dataset](https://data.mendeley.com/datasets/z45bptxptf/2),
  CC BY 4.0. Put `processed_data_obat.csv` in `data/reference/indo_pharma/`.
- [FDA National Drug Code directory](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory),
  public domain. Unzip `ndctext.zip` into `data/reference/fda_ndc/`.

Then run `uvx --with duckdb --with pandas python -I scripts/build_reference_labels.py`.

## Licence

The code is MIT licensed (see [LICENSE](LICENSE)). The IRPD data and the
reference drug lists keep their own licences, credited above.
