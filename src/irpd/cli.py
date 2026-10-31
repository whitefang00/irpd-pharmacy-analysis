"""The `irpd` command: build, map-ingredients, spend, replay, evals and contest."""

import argparse
import time

from irpd import config, holdout, ingredients, llm, warehouse


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="irpd")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build the DuckDB warehouse from the IRPD files")
    m = sub.add_parser("map-ingredients", help="map drug names to ingredients with the LLM")
    m.add_argument("--limit", type=int, help="map at most this many new drugs")
    m.add_argument("--model", default="luna",
                   help="luna or sol (short for the model names), or a full model name")
    m.add_argument("--prompt", default=ingredients.PROMPT_VERSION,
                   choices=sorted(ingredients.PROMPTS), help="prompt version")
    sub.add_parser("spend", help="show model spend against the cap")
    r = sub.add_parser("replay", help="score reference policies against the pharmacist on H2 2025")
    r.add_argument("--uncensored", action="store_true",
                   help="add back demand lost on the pharmacist's stock-out days")
    sub.add_parser("evals", help="run the agent on the eval questions (see irpd evals -h)",
                   add_help=False)
    sub.add_parser("contest", help="agent vs pharmacist: plan or score (see irpd contest -h)",
                   add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.command == "evals":
        from irpd.agent import run_evals
        run_evals.main(rest)
        return
    if args.command == "contest":
        from irpd import contest
        contest.main(rest)
        return
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.command == "spend":
        print(f"spent ${llm.spent():.4f} of ${llm.CAP_USD:.2f}")
    if args.command == "map-ingredients":
        table = ingredients.run(limit=args.limit, model=args.model, version=args.prompt)
        s = ingredients.score(table)
        print(f"scored {s['scored']:,} drugs against the answer key: "
              f"covered {s['covered']:.1%} (n={s['covered_n']}), "
              f"exact on generic names {s['exact']:.1%} (n={s['exact_n']}), "
              f"abstained {s['abstained']:.1%}")
        print(f"spent ${llm.spent():.4f} of ${llm.CAP_USD:.2f}")

    if args.command == "build":
        start = time.time()
        path = warehouse.build()
        shown = path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path
        print(f"built {shown} in {time.time() - start:.1f}s")
        with warehouse.as_of(config.LAST_DAY, path) as con:
            for name in warehouse.date_columns():
                try:
                    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                except Exception:
                    continue
                print(f"  {name:<24} {n:>10,} rows")
        holdout.seal(path)
        print(f"sealed the holdout: {holdout.AGENT.name} (agent), {holdout.SEALED.name} (scorer)")

    if args.command == "replay":
        replay_report(uncensored=args.uncensored)


def replay_report(uncensored: bool = False) -> None:
    import pandas as pd

    from irpd import policies, replay

    if uncensored:
        from irpd import lost
        h = replay.load(demand=lost.uncensored_demand())
    else:
        h = replay.load()
    with holdout.agent_view() as con:
        refs = {"mimic": policies.mimic(con), "textbook": policies.textbook(con)}
    runs = dict(refs)
    for name, policy in refs.items():
        matched, k = replay.match_stock_value(policy, h)
        runs[f"{name} at matched stock (x{k:.2f})"] = matched
    table = replay.score(runs, h)
    cols = ["stockout_days", "shelfout_days", "lost_revenue", "avg_stock_value"]
    with pd.option_context("display.width", 140, "display.float_format", "{:,.0f}".format):
        demand = "uncensored" if uncensored else "recorded"
        print(f"H2 2025, {len(h.drugs)} fast and regular drugs, {demand} demand "
              "(stock valued at selling price)")
        print(table[[c for c in cols if c in table.columns]].to_string())


if __name__ == "__main__":
    main()
