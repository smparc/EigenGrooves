#!/usr/bin/env python
"""
Generate every figure in the paper from the live codebase.

Run from the repository root:

    python paper/make_figures.py               # synthetic catalogue
    python paper/make_figures.py --data data/spotify_songs.csv

Nothing in the paper is a screenshot of a number typed by hand: each figure and
each generated table is produced here, so re-running after a code change
updates the paper rather than silently invalidating it. The numbers quoted in
the prose are written to ``figures/values.tex`` and pulled in as LaTeX macros.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from eigengrooves import (  # noqa: E402
    Catalog,
    build_groups,
    build_standard_rankers,
    compare_rankers,
    compare_to_labels,
    confusion_table,
    fit_latent_model,
    make_synthetic_catalog,
    paired_bootstrap_test,
)
from eigengrooves.linalg import svd  # noqa: E402
from eigengrooves.normalization import fit_scaler  # noqa: E402
from eigengrooves.rank import (  # noqa: E402
    rank_by_elbow,
    rank_by_gavish_donoho,
    rank_by_variance,
)

FIGURES = Path(__file__).resolve().parent / "figures"

# A single palette, applied everywhere, so the figures read as one document.
BLUE = "#3B6EA5"
CORAL = "#D96C5F"
GREY = "#B8BCC4"
GREEN = "#4F8A6B"
PURPLE = "#7A5C99"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_scaling(catalog, scaled) -> dict:
    """Why scaling is mandatory: raw variance is all tempo."""
    raw_variance = catalog.features.var(axis=0)
    share = raw_variance.max() / raw_variance.sum()
    dominant = catalog.feature_names[int(np.argmax(raw_variance))]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].barh(list(catalog.feature_names), raw_variance, color=CORAL)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("variance (log scale)")
    axes[0].set_title("Before scaling")
    axes[1].barh(list(catalog.feature_names), scaled.var(axis=0), color=BLUE)
    axes[1].set_xlabel("variance")
    axes[1].set_title("After z-scoring")
    fig.tight_layout()
    _save(fig, "fig_scaling.png")
    return {"dominantFeature": dominant, "dominantShare": f"{share * 100:.1f}"}


def figure_spectrum(scaled) -> dict:
    """Scree plot with the three rank-selection rules marked."""
    _, spectrum, _ = svd(scaled)
    cumulative = np.cumsum(spectrum**2) / np.sum(spectrum**2)

    k_var = rank_by_variance(spectrum, 0.90)
    k_elbow = rank_by_elbow(spectrum)
    k_gd = rank_by_gavish_donoho(spectrum, *scaled.shape)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].bar(range(1, len(spectrum) + 1), spectrum, color=BLUE)
    axes[0].set(xlabel="component", ylabel="singular value", title="Scree")

    axes[1].plot(range(1, len(spectrum) + 1), cumulative * 100, "o-", color=CORAL, ms=4)
    axes[1].axhline(90, color=GREY, ls="--", lw=1)
    for k, colour, label in [
        (k_var, GREEN, f"variance $k$={k_var}"),
        (k_elbow, PURPLE, f"elbow $k$={k_elbow}"),
        (k_gd, "black", f"Gavish--Donoho $k$={k_gd}"),
    ]:
        axes[1].axvline(k, color=colour, ls=":", lw=1.6, label=label)
    axes[1].set(xlabel="components kept", ylabel="cumulative variance (%)",
                title="Explained variance")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_spectrum.png")

    return {
        "kVariance": str(k_var),
        "kElbow": str(k_elbow),
        "kGavishDonoho": str(k_gd),
        "varAtFive": f"{cumulative[min(4, len(cumulative) - 1)] * 100:.1f}",
        "nComponents": str(len(spectrum)),
        "sigmaOne": f"{spectrum[0]:.2f}",
    }


def figure_backend_accuracy() -> dict:
    """Relative error per singular value: Jacobi against the A^T A route."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(800, 4)) @ rng.normal(size=(4, 9))
    ill = base + 1e-6 * rng.normal(size=(800, 9))
    truth = np.linalg.svd(ill, compute_uv=False)

    errors = {}
    for backend in ("jacobi", "eigh"):
        _, sigma, _ = svd(ill, backend=backend)
        n = min(len(sigma), len(truth))
        errors[backend] = np.abs(sigma[:n] - truth[:n]) / truth[:n]

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    index = np.arange(1, len(truth) + 1)
    ax.semilogy(index[: errors["jacobi"].size], np.maximum(errors["jacobi"], 1e-17),
                "o-", color=BLUE, ms=4, label="one-sided Jacobi")
    ax.semilogy(index[: errors["eigh"].size], np.maximum(errors["eigh"], 1e-17),
                "s--", color=CORAL, ms=4, label=r"via $A^{\mathsf{T}}A$")
    ax.axvspan(4.5, len(truth) + 0.5, color=GREY, alpha=0.18)
    ax.text(6.8, 1e-9, "noise-scale\ncomponents", fontsize=7, ha="center", color="#555")
    ax.set(xlabel="singular value index", ylabel="relative error",
           title="Accuracy on a near-collinear matrix")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_backend_accuracy.png")

    return {
        "jacobiMaxRelErr": f"{errors['jacobi'].max():.2e}".replace("e-", "e{-}"),
        "eighMaxRelErr": f"{errors['eigh'].max():.2e}".replace("e-", "e{-}"),
    }


def figure_loadings(model) -> None:
    """Latent feature loadings as a heatmap."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    data = model.components
    limit = np.abs(data).max()
    image = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(model.feature_names)))
    ax.set_xticklabels(model.feature_names, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(model.k))
    ax.set_yticklabels([f"LF{i}" for i in range(1, model.k + 1)], fontsize=8)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if abs(data[i, j]) > 0.28:
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white")
    fig.colorbar(image, ax=ax, shrink=0.85, label="loading")
    ax.set_title("Latent feature loadings (rows of $V_k^{\\mathsf{T}}$)")
    fig.tight_layout()
    _save(fig, "fig_loadings.png")


def figure_projection(catalog, scaled, model, seeds) -> None:
    """Original feature space against latent space.

    A seed playlist highlighted against the catalogue, shown before and after
    projection, so the concentrating effect of the reduction is visible.
    """
    latent = model.transform(catalog.features)
    fig = plt.figure(figsize=(7.4, 3.4))

    # Thin the background cloud: at 3000 points the overdraw hides the very
    # structure the figure is meant to show.
    rng = np.random.default_rng(0)
    n_background = min(1200, len(catalog))
    background = rng.choice(len(catalog), size=n_background, replace=False)

    for position, (data, labels, title) in enumerate([
        (scaled[:, [0, 1, 3]],
         ("danceability", "energy", "acousticness"),
         "Original feature space\n(3 of 9 features)"),
        (latent[:, :3],
         ("LF1", "LF2", "LF3"),
         "Latent feature space\n(SVD projection)"),
    ]):
        ax = fig.add_subplot(1, 2, position + 1, projection="3d")
        ax.scatter(data[background, 0], data[background, 1], data[background, 2],
                   c="#8A8F99", alpha=0.30, s=7, linewidths=0, depthshade=False)
        ax.scatter(data[seeds, 0], data[seeds, 1], data[seeds, 2],
                   c=CORAL, s=42, edgecolors="white", linewidths=0.6, depthshade=False)
        ax.set_xlabel(labels[0], fontsize=7, labelpad=-4)
        ax.set_ylabel(labels[1], fontsize=7, labelpad=-4)
        ax.set_zlabel(labels[2], fontsize=7, labelpad=-4)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_title(title, fontsize=9)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#8A8F99", label="catalogue"),
        plt.Line2D([], [], marker="o", ls="", color=CORAL, label="seed playlist"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    _save(fig, "fig_projection.png")


def figure_genre_space(catalog, model) -> None:
    """The catalogue in latent space, coloured by a label never shown to the model."""
    if "genre" not in catalog.frame.columns:
        return
    latent = model.transform(catalog.features)
    genres = catalog.frame["genre"].astype(str)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    palette = plt.get_cmap("tab10")
    for i, genre in enumerate(sorted(genres.unique())):
        mask = (genres == genre).to_numpy()
        ax.scatter(latent[mask, 0], latent[mask, 1], s=6, alpha=0.6,
                   color=palette(i % 10), label=genre, linewidths=0)
    ax.set(xlabel="LF1", ylabel="LF2",
           title="Latent space, coloured by genre\n(genre was never given to the model)")
    ax.legend(fontsize=6, ncol=2, markerscale=2, loc="best")
    fig.tight_layout()
    _save(fig, "fig_genre_space.png")


def figure_evaluation(results, tests) -> dict:
    """Accuracy and diversity side by side, with the significance verdict."""
    names = [r.name for r in results]
    significant = {t.name_a: t.significant for t in tests}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, metric, title in [
        (axes[0], "ndcg", "Accuracy (NDCG@10)"),
        (axes[1], "diversity", "Intra-list diversity"),
    ]:
        values = [r.metrics[metric] for r in results]
        colours = []
        for name in names:
            if name == "raw_cosine":
                colours.append(CORAL)
            elif name in ("random", "popularity"):
                colours.append(GREY)
            else:
                colours.append(BLUE)
        bars = ax.barh(names, values, color=colours)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.tick_params(labelsize=7)
        if metric == "ndcg":
            for bar, name in zip(bars, names):
                if name in significant and not significant[name]:
                    ax.text(bar.get_width() * 1.03, bar.get_y() + bar.get_height() / 2,
                            "n.s.", va="center", fontsize=6, color="#444")
    fig.suptitle("Coral = the no-SVD control; 'n.s.' = indistinguishable from it",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_evaluation.png")

    raw = next(r for r in results if r.name == "raw_cosine")

    # svd_k9 keeps every component, so its projection is a pure rotation and it
    # is provably identical to raw cosine. Report it separately as a
    # correctness check; "best" must mean the best *reduction*.
    full_rank = next((r for r in results if r.name == "svd_k9"), None)
    reduced = [
        r for r in results
        if r.name.startswith("svd") and r.name != "svd_k9"
    ]
    best = max(reduced, key=lambda r: r.metrics["ndcg"])
    best_test = next(t for t in tests if t.name_a == best.name)

    values = {
        "bestSvd": best.name.replace("_", r"\_"),
        "bestSvdNdcg": f"{best.metrics['ndcg']:.4f}",
        "rawNdcg": f"{raw.metrics['ndcg']:.4f}",
        "bestSvdDelta": f"{best_test.difference:+.4f}",
        "bestSvdCiLow": f"{best_test.ci_low:+.4f}",
        "bestSvdCiHigh": f"{best_test.ci_high:+.4f}",
        "bestSvdP": f"{best_test.p_value:.3f}",
        "bestSvdDiversity": f"{best.metrics['diversity']:.4f}",
        "rawDiversity": f"{raw.metrics['diversity']:.4f}",
        "bestSvdCoverage": f"{best.metrics['coverage']:.4f}",
        "rawCoverage": f"{raw.metrics['coverage']:.4f}",
    }
    if full_rank is not None:
        gap = abs(full_rank.metrics["ndcg"] - raw.metrics["ndcg"])
        values["fullRankNdcg"] = f"{full_rank.metrics['ndcg']:.4f}"
        values["fullRankGap"] = f"{gap:.2e}".replace("e-", "e{-}") if gap else "0"
    return values


def figure_confusion(catalog, model) -> dict:
    """Cluster-by-genre agreement: does the latent space rebuild the taxonomy?"""
    if "genre" not in catalog.frame.columns:
        return {}
    latent = model.transform(catalog.features)
    labels = catalog.frame["genre"].astype(str).tolist()
    result, agreement = compare_to_labels(latent, labels, random_state=0)
    table, cluster_keys, class_keys = confusion_table(result.labels, labels)

    # Order clusters by their dominant genre so the block structure is visible.
    order = np.argsort(np.argmax(table, axis=1), kind="stable")
    table = table[order]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    fractions = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(fractions, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_keys)))
    ax.set_xticklabels(class_keys, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(cluster_keys)))
    ax.set_yticklabels([f"C{i}" for i in range(len(cluster_keys))], fontsize=7)
    ax.set(xlabel="genre label", ylabel="latent cluster")
    ax.grid(False)
    for i in range(fractions.shape[0]):
        for j in range(fractions.shape[1]):
            if fractions[i, j] > 0.12:
                ax.text(j, i, f"{fractions[i, j]:.0%}", ha="center", va="center",
                        fontsize=6, color="white" if fractions[i, j] > 0.5 else "#333")
    fig.colorbar(image, ax=ax, shrink=0.85, label="share of cluster")
    ax.set_title(
        f"Latent clusters vs. genre  (ARI={agreement.adjusted_rand_index:.3f}, "
        f"NMI={agreement.normalized_mutual_information:.3f})", fontsize=9
    )
    fig.tight_layout()
    _save(fig, "fig_confusion.png")

    return {
        "ari": f"{agreement.adjusted_rand_index:.3f}",
        "nmi": f"{agreement.normalized_mutual_information:.3f}",
        "clusterPurity": f"{agreement.purity:.3f}",
        "silhouette": f"{agreement.silhouette:.3f}",
        "nGenres": str(agreement.n_reference_classes),
        "clusterVerdict": agreement.verdict(),
    }


# ---------------------------------------------------------------------------
# Generated LaTeX tables
# ---------------------------------------------------------------------------


def write_evaluation_table(results, tests) -> None:
    significance = {t.name_a: t for t in tests}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"System & HR@10 & R@10 & MRR & NDCG@10 & Div. & Cov. \\",
        r"\midrule",
    ]
    for r in results:
        name = r.name.replace("_", r"\_")
        marker = ""
        if r.name in significance and not significance[r.name].significant:
            marker = r"$^{\dagger}$"
        elif r.name == "raw_cosine":
            name = r"\textbf{" + name + "}"
        lines.append(
            f"{name}{marker} & {r.metrics['hit_rate']:.4f} & {r.metrics['recall']:.4f} & "
            f"{r.metrics['mrr']:.4f} & {r.metrics['ndcg']:.4f} & "
            f"{r.metrics['diversity']:.4f} & {r.metrics['coverage']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = FIGURES / "table_evaluation.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path.relative_to(_ROOT)}")


def write_values(values: dict) -> None:
    """Emit every quoted number as a LaTeX macro."""
    lines = [
        "% Generated by paper/make_figures.py -- do not edit by hand.",
        "% Every number quoted in the paper is defined here from a live run.",
    ]
    for key, value in sorted(values.items()):
        lines.append(rf"\newcommand{{\{key}}}{{{value}}}")
    path = FIGURES / "values.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None,
                        help="catalogue CSV; omit to use the synthetic catalogue")
    parser.add_argument("--songs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.data and args.data.exists():
        catalog = Catalog.from_csv(args.data)
        source = args.data.name
    else:
        catalog = make_synthetic_catalog(n_songs=args.songs, random_state=args.seed)
        source = "synthetic"
    print(f"Catalogue: {len(catalog)} tracks ({source}), "
          f"{catalog.n_duplicates_removed} duplicate rows collapsed")

    scaled, _ = fit_scaler(catalog.features, method="zscore")
    model = fit_latent_model(catalog.features, catalog.feature_names,
                             k="variance", random_state=args.seed)
    print(f"Model: {model.rank_selection}")

    # A coherent seed playlist: several tracks by the catalogue's busiest artist.
    by_artist: dict[str, list[int]] = {}
    for index, artist in enumerate(catalog.artists):
        by_artist.setdefault(artist, []).append(index)
    seeds = max(by_artist.values(), key=len)[:6]

    values: dict[str, str] = {
        "catalogSize": f"{len(catalog):,}".replace(",", r"{,}"),
        "duplicatesRemoved": f"{catalog.n_duplicates_removed:,}".replace(",", r"{,}"),
        "catalogSource": source,
        "modelK": str(model.k),
        "retainedVariance": f"{model.explained_variance().sum() * 100:.1f}",
    }

    print("Figures:")
    values |= figure_scaling(catalog, scaled)
    values |= figure_spectrum(scaled)
    values |= figure_backend_accuracy()
    figure_loadings(model)
    figure_projection(catalog, scaled, model, seeds)
    figure_genre_space(catalog, model)
    values |= figure_confusion(catalog, model)

    # Evaluation.
    configurations = {
        "svd_k2": dict(k=2), "svd_k3": dict(k=3), "svd_k5": dict(k=5),
        "svd_k7": dict(k=7), "svd_k9": dict(k=9),
    }
    models = {
        name: fit_latent_model(catalog.features, catalog.feature_names,
                               random_state=args.seed, **settings)
        for name, settings in configurations.items()
    }
    models["svd_k5_whiten"] = models["svd_k5"].with_whiten(True)

    rankers = build_standard_rankers(catalog, scaled, models, random_state=args.seed)
    groups = build_groups(catalog, group_by="artist", seed_size=3,
                          min_group_size=6, random_state=args.seed)
    results = compare_rankers(rankers, groups, catalog, scaled, k=10)
    baseline = next(r for r in results if r.name == "raw_cosine")
    tests = [
        paired_bootstrap_test(r, baseline, metric="ndcg", random_state=args.seed)
        for r in results if r.name != baseline.name
    ]

    values["nQueries"] = str(len(groups))
    values |= figure_evaluation(results, tests)
    write_evaluation_table(results, tests)
    write_values(values)

    (FIGURES / "results.json").write_text(
        json.dumps(
            {
                "catalog": {"size": len(catalog), "source": source},
                "model": {"k": model.k,
                          "retained_variance": float(model.explained_variance().sum())},
                "systems": [{"name": r.name, "metrics": r.metrics} for r in results],
                "tests": [{"system": t.name_a, "difference": t.difference,
                           "ci": [t.ci_low, t.ci_high], "p": t.p_value,
                           "significant": t.significant} for t in tests],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {(FIGURES / 'results.json').relative_to(_ROOT)}")

    print("\nSignificance vs. raw_cosine (NDCG@10):")
    for t in tests:
        print(f"  {t.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
