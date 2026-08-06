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
from .cluster import compare_to_labels, confusion_table
from .console import Console, glyph, rule
from .evaluate import (
    build_groups,
    compare_rankers,
    format_comparison,
    paired_bootstrap_test,
)
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
        # Showing what *is* in the catalogue turns a dead end into a next step;
        # the usual cause is running the default playlist against a dataset
        # that simply does not contain those songs.
        examples = [catalog.describe(i) for i in range(min(5, len(catalog)))]
        if args.json:
            print(json.dumps(
                {"error": message, "unresolved": unresolved, "examples": examples},
                indent=2,
            ))
        else:
            console.print(f"\n{message}")
            console.print("\n  Tracks that are in this catalogue:")
            for example in examples:
                console.print(f"    - {example}")
            console.print("\n  Pass one with --playlist, or use --synthetic.")
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
        bar = glyph("█") * max(round(40 * value / peak), 1)
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

    # The headline comparison: does any latent model actually beat the no-SVD
    # control? A difference in means is not an answer -- the queries are shared,
    # so the test must be paired. And every model is tested against the same
    # control, so the tests form one family and must be corrected together:
    # eight comparisons at an uncorrected 5% produce a spurious "significant"
    # about a third of the time.
    baseline = next((r for r in results if r.name == "raw_cosine"), None)
    tests = []
    if baseline is not None:
        tests = holm_correction([
            paired_bootstrap_test(r, baseline, metric=args.test_metric,
                                  random_state=args.seed)
            for r in results
            if r.name != baseline.name
        ])

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
                {
                    "name": r.name,
                    "metrics": r.metrics,
                    "ci": {
                        m: list(r.ci(m))
                        for m in ("ndcg", "recall", "mrr", "hit_rate", "diversity")
                    },
                }
                for r in results
            ],
            "paired_tests_vs_raw_cosine": [
                {
                    "system": t.name_a,
                    "metric": t.metric,
                    "difference": t.difference,
                    "ci": [t.ci_low, t.ci_high],
                    "p_value": t.p_value,
                    "p_value_adj": t.p_value_adj,
                    "correction": "holm",
                    "n_comparisons": t.n_comparisons,
                    "significant": t.significant,
                    "interval_excludes_zero": t.interval_excludes_zero,
                }
                for t in tests
            ],
        }, indent=2))
        return 0

    console.header(f"Evaluation - {len(groups)} queries, grouped by {args.group_by}")
    console.print(format_comparison(results, k=args.at_k))

    if tests:
        console.section(f"Paired bootstrap vs. raw_cosine ({args.test_metric})")
        console.print(
            f"  {'system':<16} {'difference':>11}  {'95% CI':>22}  {'p':>7}   verdict"
        )
        for t in tests:
            verdict = "significant" if t.significant else "n.s."
            console.print(
                f"  {t.name_a:<16} {t.difference:>+11.4f}  "
                f"[{t.ci_low:>+9.4f}, {t.ci_high:>+9.4f}]  {t.p_value:>7.3f}   {verdict}"
            )

    console.print("")
    console.print(f"  {rule(58)}")
    console.print("  Higher is better for every column. `random` is the floor;")
    console.print("  `raw_cosine` is the no-SVD control that the latent models")
    console.print("  must beat to justify the decomposition at all.")
    console.print("  'n.s.' means the 95% interval for the paired difference")
    console.print("  includes zero - the systems are not distinguishable here.")
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    """Answer the project's original question: does the latent space rebuild genre?"""
    console = Console(quiet=args.json)
    catalog = _load_catalog(args, console)

    if args.label_column not in catalog.frame.columns:
        message = (
            f"catalog has no '{args.label_column}' column, so there is nothing to "
            "compare the clustering against. Use --synthetic, or supply a CSV "
            "with a genre column."
        )
        if args.json:
            print(json.dumps({"error": message}, indent=2))
        else:
            console.print(f"\nerror: {message}\n")
        return 2

    model = _fit(args, catalog)
    latent = model.transform(catalog.features)
    labels = catalog.frame[args.label_column].astype(str).tolist()

    result, agreement = compare_to_labels(
        latent, labels, k=args.clusters, random_state=args.seed
    )

    if args.json:
        table, cluster_keys, class_keys = confusion_table(result.labels, labels)
        print(json.dumps({
            "k_latent": model.k,
            "n_clusters": agreement.k,
            "n_reference_classes": agreement.n_reference_classes,
            "adjusted_rand_index": agreement.adjusted_rand_index,
            "normalized_mutual_information": agreement.normalized_mutual_information,
            "purity": agreement.purity,
            "silhouette": agreement.silhouette,
            "verdict": agreement.verdict(),
            "confusion": {
                "clusters": [int(c) for c in cluster_keys],
                "classes": [str(c) for c in class_keys],
                "counts": table.tolist(),
            },
        }, indent=2))
        return 0

    console.header(f"Latent clustering vs. {args.label_column}")
    console.print(f"  Latent dimensions      : {model.k}")
    console.print(f"  Clusters               : {agreement.k}")
    console.print(f"  Reference classes      : {agreement.n_reference_classes}")
    console.print("")
    console.print(f"  Adjusted Rand Index    : {agreement.adjusted_rand_index:.4f}   (0 = chance, 1 = identical)")
    console.print(f"  Normalised Mutual Info : {agreement.normalized_mutual_information:.4f}")
    console.print(f"  Purity                 : {agreement.purity:.4f}   (not chance-corrected)")
    console.print(f"  Silhouette             : {agreement.silhouette:.4f}   (cluster separation)")
    console.print("")
    console.print(f"  Verdict: the latent partition {agreement.verdict()}.")

    table, cluster_keys, class_keys = confusion_table(result.labels, labels)
    console.section("Which clusters map to which classes")
    width = max((len(str(c)) for c in class_keys), default=8)
    header = "  cluster  " + "  ".join(str(c).rjust(width) for c in class_keys)
    console.print(header)
    for row_index, cluster in enumerate(cluster_keys):
        counts = "  ".join(str(v).rjust(width) for v in table[row_index])
        console.print(f"  {cluster!s:>7}  {counts}")
    console.print("")
    console.print("  Rows summing across several columns = genres the audio features")
    console.print("  consider interchangeable. Columns split across rows = genres the")
    console.print("  features consider internally inconsistent.")
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
    ev.add_argument("--test-metric", default="ndcg",
                    choices=("ndcg", "recall", "mrr", "hit_rate", "precision"),
                    help="metric used for the paired significance test")
    ev.set_defaults(func=cmd_evaluate)

    cl = subparsers.add_parser(
        "cluster",
        help="cluster the latent space and compare it to a genre taxonomy",
    )
    _add_common_arguments(cl)
    cl.add_argument("--clusters", type=int, default=None,
                    help="number of clusters (default: number of reference classes)")
    cl.add_argument("--label-column", default="genre",
                    help="column holding the reference taxonomy")
    cl.set_defaults(func=cmd_cluster)

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
