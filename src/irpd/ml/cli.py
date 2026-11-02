"""`irpd ml ...` subcommands. Filled in as each stage lands."""

import argparse

from irpd.ml import split


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="irpd ml")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("seal", help=f"write {split.SELECTION_DB.name}: every table as of "
                                f"{split.SELECTION_END} (run once, after irpd build)")
    sub.add_parser("serve", help="train the frozen ML forecaster on the app's day and save "
                                 "its forecasts for the app (needs the ml extra)")
    sub.add_parser("rolling", help="forecast each week of H2 2025 with both forecasters, "
                                   "retraining ML each week, for the drug page's holdout view")
    args = ap.parse_args(argv)
    if args.command == "serve":
        from irpd.ml import serve

        out = serve.train()
        print(f"wrote {serve.FORECAST_FILE.name}: {out.drug_code.nunique()} drugs, "
              f"weeks {out.week_start.min()} to {out.week_start.max()}")
    elif args.command == "rolling":
        from irpd.ml import rolling

        out = rolling.build()
        print(f"wrote {rolling.FILE.name}: {out.origin.nunique()} weeks, "
              f"{out.drug_code.nunique()} drugs")
    elif args.command == "seal":
        path = split.seal_selection()
        print(f"wrote {path.name}: data up to {split.SELECTION_END}")
