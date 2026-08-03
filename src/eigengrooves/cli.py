"""
Command-line interface.

    eigengrooves recommend --playlist "Kill Bill" "Saturn" --strategy mmr
    eigengrooves analyze --k gavish_donoho
    eigengrooves evaluate --group-by artist
    eigengrooves fetch-data

Every subcommand accepts ``--synthetic`` so the tool is usable on a fresh clone
with no dataset present, and ``--json`` so the output can be consumed by
something other than a human.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .baselines import build_standard_rankers
from .catalog import DEFAULT_FEATURES, Catalog
from .console import Console, glyph, rule
from .evaluate import build_groups, compare_rankers, format_comparison
from .model import fit_latent_model
from .normalization import fit_scaler
from .recommend import AGGREGATIONS, STRATEGIES, Recommender
from .synthetic import make_synthetic_catalog

DEFAULT_DATA_PATH = Path("data") / "spotify_songs.csv"

DEFAULT_PLAYLIST = ["Kill Bill", "Saturn", "I Hate U", "Low", "Gone Girl"]


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="path to the catalogue CSV")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the built-in synthetic catalogue instead of a CSV")
    parser.add_argument("--synthetic-songs", type=int, default=3000,
                        help="size of the synthetic catalogue")
    parser.add_argument("--no-dedup", action="store_true",
                        help="skip track deduplication (reproduces the v1 bug; for demos)")
    parser.add_argument("--k", default="variance",
                        help="latent dimensions: an integer, or one of "
                             "variance|elbow|gavish_donoho")
    parser.add_argument("--variance-threshold", type=float, default=0.90,
                        help="cumulative variance target when --k variance")
    parser.add_argument("--scaling", choices=("zscore", "robust", "none"), default="zscore")
    parser.add_argument("--whiten", action="store_true",
                        help="equalise latent axes so LF1 stops dominating cosine")
    parser.add_argument("--backend", choices=("jacobi", "eigh", "randomized"),
                        default="jacobi", help="SVD algorithm")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")


def _parse_k(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _load_catalog(args: argparse.Namespace, console: Console) -> Catalog:
    if args.synthetic:
        console.ok(f"Using synthetic catalogue ({args.synthetic_songs} tracks)")
        return make_synthetic_catalog(
            n_songs=args.synthetic_songs, random_state=args.seed
        )
    catalog = Catalog.from_csv(
        args.data, DEFAULT_FEATURES, deduplicate=not args.no_dedup
    )
    console.ok(f"Loaded {len(catalog)} unique tracks from {args.data}")
    if catalog.n_duplicates_removed:
        console.ok(
            f"Collapsed {catalog.n_duplicates_removed} duplicate/invalid rows "
            "(chart re-entries)"
        )
    elif args.no_dedup:
        console.warn("Deduplication disabled - expect repeated and self-recommendations")
    return catalog


def _fit(args: argparse.Namespace, catalog: Catalog):
    return fit_latent_model(
        catalog.features,
        catalog.feature_names,
        k=_parse_k(args.k),
        scaling=args.scaling,
        whiten=args.whiten,
        backend=args.backend,
        variance_threshold=args.variance_threshold,
        random_state=args.seed,
    )


def _demo_playlist(catalog: Catalog, size: int = 4) -> list[str]:
    """Pick a stylistically coherent seed playlist from a catalogue.

    Takes tracks from the artist with the most entries, so the demo exercises
    the recommender on a playlist that actually has a direction.
    """
    counts: dict[str, list[int]] = {}
    for index, artist in enumerate(catalog.artists):
        counts.setdefault(artist, []).append(index)
    if not counts:
        return []
    best = max(counts.values(), key=len)
    # 'Title - Artist' so the resolver cannot pick a same-titled track elsewhere.
    return [f"{catalog.titles[i]} - {catalog.artists[i]}" for i in best[:size]]


def _print_model_summary(model, console: Console) -> None:
    console.section("Latent space")
    console.print(f"  {model.rank_selection}")
    evr = model.explained_variance()
    console.print(f"  Singular values : {np.round(model.singular_values, 2)}")
    console.print(
        f"  Variance/component: {(evr * 100).round(1)}%   "
        f"(retained total: {evr.sum() * 100:.1f}% of {len(model.full_spectrum)} components)"
    )
    if model.whiten:
        console.print("  Whitening       : on (latent axes equalised)")

    console.section("Latent feature interpretation")
    for i in range(1, model.k + 1):
        share = evr[i - 1] * 100
        console.print(
            f"\n  LF{i}  sigma={model.singular_values[i - 1]:.2f}  variance={share:.1f}%"
        )
        terms = []
        for name, weight in model.loadings(i)[:5]:
            sign = "+" if weight >= 0 else glyph("−")
            terms.append(f"{sign}{abs(weight):.3f}{glyph('×')}{name}")
        console.print("      " + "  ".join(terms))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_recommend(args: argparse.Namespace) -> int:
    console = Console(quiet=args.quiet or args.json)
    catalog = _load_catalog(args, console)
    model = _fit(args, catalog)
    if not args.json:
        _print_model_summary(model, console)

    recommender = Recommender(model, catalog)

    playlist = args.playlist
    if args.synthetic and playlist == DEFAULT_PLAYLIST:
        # The default playlist names real songs the synthetic catalogue cannot
        # contain. Substitute several tracks by one artist, so the demo shows a
        # *coherent* playlist rather than four unrelated songs.
        playlist = _demo_playlist(catalog, size=4)

    seeds, resolved, unresolved = catalog.resolve_playlist(playlist, fuzzy=not args.exact)

    console.section(f"Playlist ({len(playlist)} requested)")
    for query, match in resolved:
        note = "" if match.exact else f"  (fuzzy {match.score:.0%} from '{query}')"
        console.ok(f"{match.title} - {match.artist}{note}")
    for query in unresolved:
        console.warn(f"Not found: '{query}'")

    if not seeds:
        message = "No playlist tracks matched the catalogue."
        if args.json:
            print(json.dumps({"error": message, "unresolved": unresolved}, indent=2))
        else:
            console.print(f"\n{message} Try --synthetic, or check the titles.")
        return 1

    negatives: list[int] = []
    if args.avoid:
        negatives, _, _ = catalog.resolve_playlist(args.avoid, fuzzy=not args.exact)

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    payload: dict = {"catalog_size": len(catalog), "results": {}}

    for strategy in strategies:
        result = recommender.recommend(
            seeds,
            n=args.n,
            strategy=strategy,
            aggregation=args.aggregation,
            max_per_artist=None if args.max_per_artist <= 0 else args.max_per_artist,
            novelty_weight=args.novelty,
            negative_indices=negatives or None,
            mmr_lambda=args.mmr_lambda,
            explain=args.explain or args.json,
        )
        payload["results"][strategy] = result.as_dict()

        if not args.json:
            console.header(f"{strategy} ({len(result)} recommendations)")
            for position, item in enumerate(result, start=1):
                console.print(
                    f"  {position:>2}. {item.title} - {item.artist}   [{item.score:.4f}]"
                )
                if item.seed and strategy == "one_per_song":
                    console.print(f"      for: {item.seed}")
                if args.explain and item.explanation is not None:
                    console.print(f"      {glyph('└')} {item.explanation.summary()}")

    if args.json:
        payload["model"] = {
            "k": model.k,
            "rank_strategy": model.rank_selection.strategy,
            "explained_variance": model.explained_variance().tolist(),
            "components": {
                f"LF{i}": dict(model.loadings(i)) for i in range(1, model.k + 1)
            },
        }
        print(json.dumps(payload, indent=2))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    console = Console(quiet=args.json)
    catalog = _load_catalog(args, console)
    model = _fit(args, catalog)

    if args.json:
        print(json.dumps({
            "catalog_size": len(catalog),
            "duplicates_removed": catalog.n_duplicates_removed,
            "k": model.k,
            "rank_strategy": model.rank_selection.strategy,
            "rank_detail": model.rank_selection.detail,
            "singular_values": model.singular_values.tolist(),
            "full_spectrum": model.full_spectrum.tolist(),
            "explained_variance": model.explained_variance().tolist(),
            "cumulative_variance_retained": float(model.explained_variance().sum()),
            "components": {
                f"LF{i}": dict(model.loadings(i)) for i in range(1, model.k + 1)
            },
        }, indent=2))
        return 0

    _print_model_summary(model, console)

    console.section("Scree")
    spectrum = model.full_spectrum
    peak = spectrum[0] if spectrum.size else 1.0
    cumulative = np.cumsum(spectrum**2) / np.sum(spectrum**2)
    for i, value in enumerate(spectrum, start=1):
        bar = glyph("█") * max(int(round(40 * value / peak)), 1)
        marker = " <- selected k" if i == model.k else ""
        console.print(f"  {i:>2}  {value:8.3f}  {bar:<41} {cumulative[i - 1] * 100:5.1f}%{marker}")

    if args.save_model:
        model.save(args.save_model)
        console.print(f"\n  Model written to {args.save_model}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    console = Console(quiet=args.json)
    catalog = _load_catalog(args, console)

    scaled, _ = fit_scaler(catalog.features, method=args.scaling)

    # Compare several latent configurations against each other and against the
    # non-SVD baselines, all on identical queries.
    configurations = {
        "svd_k2": dict(k=2, whiten=False),
        "svd_k3": dict(k=3, whiten=False),
        "svd_k5": dict(k=5, whiten=False),
        "svd_k5_whiten": dict(k=5, whiten=True),
        "svd_k7": dict(k=7, whiten=False),
        "svd_auto": dict(k=_parse_k(args.k), whiten=args.whiten),
    }
    models = {}
    for name, settings in configurations.items():
        model = fit_latent_model(
            catalog.features,
            catalog.feature_names,
            scaling=args.scaling,
            backend=args.backend,
            variance_threshold=args.variance_threshold,
            random_state=args.seed,
            **settings,
        )
        models[f"{name}(k={model.k})" if name == "svd_auto" else name] = model

    rankers = build_standard_rankers(catalog, scaled, models, random_state=args.seed)

    groups = build_groups(
        catalog,
        group_by=args.group_by,
        seed_size=args.seed_size,
        min_group_size=args.min_group_size,
        max_groups=args.max_groups,
        random_state=args.seed,
    )
    console.ok(f"Built {len(groups)} evaluation queries grouped by {args.group_by}")

    results = compare_rankers(rankers, groups, catalog, scaled, k=args.at_k)

    if args.json:
        print(json.dumps({
            "protocol": {
                "group_by": args.group_by,
                "seed_size": args.seed_size,
                "min_group_size": args.min_group_size,
                "n_queries": len(groups),
                "at_k": args.at_k,
            },
            "systems": [
                {"name": r.name, "metrics": r.metrics} for r in results
            ],
        }, indent=2))
        return 0

    console.header(f"Evaluation - {len(groups)} queries, grouped by {args.group_by}")
    console.print(format_comparison(results, k=args.at_k))
    console.print("")
    console.print(f"  {rule(58)}")
    console.print("  Higher is better for every column. `random` is the floor;")
    console.print("  `raw_cosine` is the no-SVD control that the latent models")
    console.print("  must beat to justify the decomposition at all.")
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from .datasets import fetch

    return fetch(args.data, url=args.url, force=args.force)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eigengrooves",
        description="Music recommendation via latent audio-feature analysis (SVD).",
    )
    parser.add_argument("--version", action="version", version=f"eigengrooves {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rec = subparsers.add_parser("recommend", help="recommend songs for a playlist")
    _add_common_arguments(rec)
    rec.add_argument("--playlist", nargs="+", default=DEFAULT_PLAYLIST,
                     help="seed song titles; 'Title - Artist' disambiguates")
    rec.add_argument("--avoid", nargs="+", default=None,
                     help="songs to steer away from (Rocchio negative feedback)")
    rec.add_argument("--strategy", choices=(*STRATEGIES, "all"), default="mmr")
    rec.add_argument("--aggregation", choices=AGGREGATIONS, default="max")
    rec.add_argument("-n", type=int, default=10, help="number of recommendations")
    rec.add_argument("--max-per-artist", type=int, default=2,
                     help="cap per artist; 0 disables")
    rec.add_argument("--novelty", type=float, default=0.0,
                     help="popularity de-biasing weight (try 0.1)")
    rec.add_argument("--mmr-lambda", type=float, default=0.7,
                     help="mmr relevance/diversity trade-off; 1.0 is pure relevance")
    rec.add_argument("--explain", action="store_true",
                     help="show which latent axes drove each match")
    rec.add_argument("--exact", action="store_true", help="disable fuzzy title matching")
    rec.set_defaults(func=cmd_recommend)

    ana = subparsers.add_parser("analyze", help="inspect the latent space")
    _add_common_arguments(ana)
    ana.add_argument("--save-model", type=Path, default=None,
                     help="write the fitted model to an .npz file")
    ana.set_defaults(func=cmd_analyze)

    ev = subparsers.add_parser("evaluate", help="benchmark against baselines")
    _add_common_arguments(ev)
    ev.add_argument("--group-by", choices=("artist", "genre"), default="artist")
    ev.add_argument("--seed-size", type=int, default=3)
    ev.add_argument("--min-group-size", type=int, default=6)
    ev.add_argument("--max-groups", type=int, default=400)
    ev.add_argument("--at-k", type=int, default=10)
    ev.set_defaults(func=cmd_evaluate)

    fetch = subparsers.add_parser("fetch-data", help="install a catalogue CSV")
    fetch.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    fetch.add_argument("--url", default=None, help="direct link to a catalogue CSV")
    fetch.add_argument("--force", action="store_true", help="re-download even if present")
    fetch.set_defaults(func=cmd_fetch_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except (ValueError, IndexError, KeyError) as exc:
        print(f"\nerror: {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
