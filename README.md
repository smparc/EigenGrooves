# EigenGrooves — SVD & Dimensionality Reduction

> Discover music beyond genre labels — latent audio-feature analysis with a
> from-scratch singular value decomposition.

---

## Overview

Traditional music recommendation leans on genre tags and listening history.
This project decomposes songs into **latent audio dimensions** using SVD, then
recommends by similarity in that reduced space.

Everything numerical is implemented directly from the mathematics — QR,
symmetric eigensolvers, three SVD algorithms, cosine similarity, edit distance,
ranking metrics. NumPy provides array storage and elementwise operations;
`numpy.linalg`'s decompositions appear only in the test suite, as ground truth.

The project also ships an **evaluation harness** that benchmarks the latent
models against honest baselines, and reports the result whether or not it
flatters the premise. It currently does not — see
[Does the SVD earn its keep?](#does-the-svd-earn-its-keep).

---

## Quick start

Works on a fresh clone with no dataset and no network:

```bash
pip install -e .

python main.py                                    # recommend, with explanations
python main.py analyze  --synthetic               # inspect the latent space
python main.py evaluate --synthetic               # benchmark against baselines
```

Installed, the same commands are available as `eigengrooves`:

```bash
eigengrooves recommend --playlist "Kill Bill" "Saturn" --strategy mmr --explain
eigengrooves recommend --data path/to/songs.csv -n 20 --novelty 0.15
eigengrooves analyze --k gavish_donoho --save-model model.npz
eigengrooves evaluate --group-by genre --at-k 20
```

Sample output:

```
 1. Northern Chapel — The Circuit   [0.9768]
    └ both score high on LF2 (+0.62 tempo -0.61 speechiness); both score high on LF3 (+0.86 liveness)
```

### As a library

```python
from eigengrooves import make_synthetic_catalog, fit_latent_model, Recommender

catalog = make_synthetic_catalog()
model = fit_latent_model(catalog.features, catalog.feature_names, k="gavish_donoho")
recommender = Recommender(model, catalog)

seeds, _, unresolved = catalog.resolve_playlist(["Neon Fever", "Velvet Signal"])
for item in recommender.recommend(seeds, n=10, strategy="mmr", explain=True):
    print(item.title, "—", item.artist, "|", item.explanation.summary())
```

---

## How it works

### 1. Features

Nine Spotify audio features per track: danceability, energy, speechiness,
acousticness, instrumentalness, liveness, valence, loudness, tempo.

Scaling is not optional. Unscaled, `tempo` alone holds **95.3%** of total
variance, and the decomposition becomes a study of tempo and nothing else.
Z-score is the default; robust (median/IQR) scaling is available for the
heavily-skewed features.

### 2. Deduplication

Chart datasets carry one row per track *per chart week*. Collapsing to one row
per `(title, artist)` is what stops the recommender returning your own songs
back to you — and the collapsed row count becomes a popularity signal used for
novelty weighting.

### 3. Decomposition

$$A = U \Sigma V^\mathsf{T}$$

Three interchangeable backends:

| Backend | Algorithm | Use when |
|---|---|---|
| `jacobi` *(default)* | One-sided Jacobi | Always, unless you have a reason not to |
| `eigh` | Eigendecomposition of $A^\mathsf{T}A$ | The textbook derivation; kept for comparison |
| `randomized` | Halko–Martinsson–Tropp | $m \gg n$ and only the top $k$ are wanted |

**Why one-sided Jacobi is the default.** Forming $A^\mathsf{T}A$ squares the
condition number, and audio features are strongly correlated, so components
below $\sqrt{\varepsilon}\,\sigma_{\max}$ arrive as noise. Jacobi orthogonalises
the columns of $A$ directly and never forms that product. Measured on
correlated data (`notebooks/analysis.ipynb`, §4):

```
max relative error — jacobi: 3.27e-11
max relative error — eigh  : 5.13e-03
```

Note that *reconstruction* error does not reveal this — it is dominated by the
large singular values, and both backends reconstruct to ~1e-16. Relative
accuracy per singular value is the metric that discriminates.

Similarly, QR uses Householder reflections rather than Gram–Schmidt. On a
12×12 Hilbert matrix ($\kappa \approx 1.8\times10^{16}$), orthogonality error is
`1.8e-15` for Householder against `1.0e+00` for modified Gram–Schmidt.

### 4. Choosing $k$

Rather than a hardcoded 5:

- **`variance`** — smallest $k$ reaching a cumulative-variance threshold
- **`elbow`** — knee of the scree curve, by maximum distance from its endpoint chord
- **`gavish_donoho`** — the optimal hard threshold, $4/\sqrt{3}$ ([Gavish &
  Donoho, 2014](https://doi.org/10.1109/TIT.2014.2323359)); needs no threshold
  at all, since the matrix aspect ratio determines everything

### 5. Recommendation

| Strategy | Behaviour |
|---|---|
| `overall_top` | Globally highest-scoring candidates |
| `one_per_song` | One result per seed, cycling for larger $n$ |
| `centroid` | Query by the playlist's mean latent vector |
| `mmr` | Maximal Marginal Relevance — explicit relevance/diversity trade-off |

With per-artist caps, popularity de-biasing, Rocchio negative feedback
(`--avoid`), three seed-aggregation modes (`max`, `mean`, `borda`), and
optional whitening to stop LF1 dominating the cosine.

**Explanations.** Cosine similarity between unit latent vectors is
$\sum_i q_i r_i$, so each term is exactly how much component $i$ contributed.
Pair that with the component's loadings and every recommendation explains
itself.

---

## Does the SVD earn its keep?

The premise is that projecting to a latent subspace beats using the features
directly. That is a hypothesis, so the harness tests it — with a **paired
bootstrap**, because all systems see identical queries and a difference in
means is not by itself evidence. Held-out-artist protocol, 219 queries:

```
system         hit_rate@10  recall@10  mrr     ndcg@10  diversity  coverage   vs raw_cosine
-------------  -----------  ---------  ------  -------  ---------  --------   --------------------
raw_cosine     0.3242       0.0351     0.0865  0.0371   0.2125     0.4698     (control)
svd_auto(k=9)  0.3242       0.0351     0.0865  0.0371   0.2125     0.4698     Δ=0.0000  identical
svd_k7         0.3059       0.0322     0.0793  0.0340   0.2331     0.4875     p_adj=0.855  n.s.
svd_k5_whiten  0.2009       0.0224     0.0559  0.0237   0.3208     0.5032     p_adj=0.012  worse
svd_k5         0.2283       0.0241     0.0508  0.0236   0.2903     0.5052     p_adj=0.008  worse
svd_k3         0.1370       0.0153     0.0412  0.0169   0.3904     0.5128     p_adj=0.001  worse
svd_k2         0.0959       0.0086     0.0338  0.0120   0.4973     0.5189     p_adj=0.001  worse
random         0.0457       0.0057     0.0154  0.0056   1.0046     0.5202     p_adj=0.001  worse
popularity     0.0411       0.0033     0.0088  0.0036   1.0131     0.0037     p_adj=0.001  worse
```

Reproduce with `eigengrooves evaluate --synthetic`. The default sweep reports
`svd_auto` at the rank the selector picks, which is 7 — the full-rank row above
needs `eigengrooves evaluate --synthetic --k 9`.

**The p-values are Holm-corrected**, because every row is tested against the
same control and the eight comparisons are therefore one family. At an
uncorrected 5% threshold a family that size throws a spurious "significant"
roughly a third of the time — and the entire point of this harness is that the
SVD's value is a hypothesis under test rather than a premise being illustrated.
Correcting made no difference to any conclusion here (raw p = 0.427, 0.004,
0.002, <0.001…), which is the outcome worth having: the finding survives the
stricter test rather than depending on the looser one. `ComparisonTest.significant`
reads off the adjusted value only, so an uncorrected test reports as not
significant — in isolation it is not yet an answer.

**The honest reading**, in three parts:

*The pipeline is provably correct.* At full rank (`--k 9`) the SVD model
reproduces `raw_cosine` to every digit on every metric. That is the expected identity: at full rank the
projection is an orthogonal rotation, and cosine similarity is rotation-
invariant. Any bug in scaling, decomposition or ranking would break it. It also
sharpens the question — the SVD can only do something *through truncation*.

*Truncating to k=7 costs nothing measurable.* Δ = −0.0031, 95% CI
[−0.0105, +0.0048], p = 0.427 (Holm-adjusted 0.855). The interval contains zero.
Meanwhile diversity rises 0.2125 → 0.2331 and coverage 0.4698 → 0.4875. Two
dimensions are free.

*Below that, reduction is significantly worse.* Every k ≤ 5 comparison survives
Holm correction at p_adj < 0.05, and accuracy falls monotonically with k. The
original project's hardcoded k=5 sits on the wrong side of that line.

So dimensionality reduction here is a **diversity technique, not an accuracy
technique**. Every latent model still beats random and popularity by a wide
margin, so the features carry real signal.

Two caveats that genuinely matter:

- **The protocol is biased against this project's own goal.** It rewards
  recovering known stylistic neighbours, so the cross-genre discovery the
  system is designed for scores as a miss by construction.
- **These numbers are from the synthetic catalogue**, whose generative
  structure is known and influences the outcome. Re-run against a real dataset
  before treating any of it as settled.

Which is the point of shipping the harness: the claim is now falsifiable, and
changing `k`, the scaling, or the aggregation produces a number rather than an
opinion.

---

## Does audio-feature classification reproduce genre?

The original research question, finally measured. `eigengrooves cluster
--synthetic` partitions the latent space with from-scratch k-means and compares
the partition to genre labels using chance-corrected metrics:

```
Adjusted Rand Index    : 0.4338   (0 = chance, 1 = identical)
Normalised Mutual Info : 0.5716
Silhouette             : 0.1784
```

Partial agreement — the taxonomy is neither recovered nor unrelated. The
confusion table is where the finding actually is:

| Genre | Outcome |
|---|---|
| hip-hop (88%), live (87%), EDM (78%) | **cleanly separable** — each owns a cluster |
| ambient + classical | **merged** — split across two clusters that are both mixtures |
| pop, rock, R&B | **dissolved** — no cluster of their own |

Audio features reorganise music by **production character**, not genre
ancestry. Where a genre label encodes production (speechiness, audience noise,
synthetic timbre) it survives; where it encodes history and marketing, it
dissolves.

Same caveat as above: the specific merges partly reflect the synthetic
generator's hand-written genre profiles. The method transfers; re-measure on a
real corpus before relying on the particulars.

---

## Getting a real dataset

There is none bundled, and none can be downloaded automatically. The dataset
this project references
([Orlandi](https://github.com/JulianoOrlandi/Spotify_Top_Songs_and_Audio_Features))
ships a *builder notebook*, not a CSV, and building it requires chart exports
from a companion scraper plus Spotify API credentials.

```bash
python scripts/fetch_data.py                  # explains the options
python scripts/fetch_data.py --url <csv-url>  # install a CSV you have located
```

Any CSV with `track_name`, `artist_names` and the nine audio features will
load; common column aliases are accepted. Otherwise `--synthetic` works
everywhere.

> Spotify restricted the `audio-features` endpoint for newly-created API
> applications in late 2024. Confirm you still have access before investing
> effort in rebuilding the dataset.

---

## Project structure

```
src/eigengrooves/
├── linalg/
│   ├── qr.py            # Householder reflections, modified Gram-Schmidt
│   ├── eigen.py         # cyclic Jacobi + shifted QR with deflation
│   ├── jacobi_svd.py    # one-sided Jacobi SVD
│   ├── randomized.py    # Halko-Martinsson-Tropp
│   └── svd.py           # backend dispatch, sign canonicalisation
├── rank.py              # variance / elbow / Gavish-Donoho
├── normalization.py     # z-score and robust scalers
├── catalog.py           # loading, validation, deduplication
├── synthetic.py         # generated catalogue with genre structure
├── model.py             # fit / transform / persist
├── similarity.py        # vectorised cosine
├── recommend.py         # four strategies, caps, novelty, negative feedback
├── explain.py           # per-component similarity attribution
├── metrics.py           # accuracy + beyond-accuracy metrics
├── baselines.py         # random, popularity, raw-cosine controls
├── evaluate.py          # held-out protocol and comparison
├── matching.py          # Levenshtein, token-set, title normalisation
├── console.py           # encoding-safe output
└── cli.py               # recommend / analyze / evaluate / cluster / fetch-data
tests/                   # 355 tests
notebooks/analysis.ipynb # visual walkthrough, runs without a dataset
paper/paper.pdf          # the write-up; rebuild with `python paper/build.py`
```

## Paper

[`paper/paper.pdf`](paper/paper.pdf) is the full write-up: method, the
evaluation and its negative result, the genre-clustering analysis, and a
section documenting the defects in the original pipeline.

```bash
python paper/build.py                              # synthetic catalogue
python paper/build.py --data data/spotify_songs.csv
```

Every figure, table and quoted number is regenerated from a live run of the
package, so the paper cannot drift from the code.

---

## What changed in v2

Each of these was reproduced before it was fixed, and each has a regression
test in `tests/test_regressions.py`.

| | Defect | Evidence |
|---|---|---|
| **P0** | `python main.py` crashed on Windows. `stdout` is cp1252; the code printed `✓` unconditionally → `UnicodeEncodeError`. The entry point was unrunnable. | `test_cli_runs_under_a_legacy_code_page` |
| **P0** | The system recommended input songs back to themselves at similarity `1.000000`. Exclusion was by *row*, but chart data carries many rows per track. | `test_a_song_is_never_recommended_to_itself` |
| **P0** | Duplicate flooding: 10 recommendations containing 4 unique titles. | `test_recommendations_contain_no_duplicate_tracks` |
| **P0** | The repo could not run after cloning — no data, no fetch path, no fallback. | `--synthetic` everywhere |
| **P1** | "Variance explained: 100.0%" on every run — the truncated spectrum was normalised against itself. True figure for k=5 of 9 was **58.7%**. | `test_model_reports_variance_against_the_full_spectrum` |
| **P1** | `src/` was not importable. `__init__.py` used relative imports, submodules used absolute → `ModuleNotFoundError: No module named 'svd'`. | `test_package_imports_cleanly` |
| **P1** | The eigensolver's convergence test compared an *absolute* off-diagonal sum against `1e-10`. At 5000 rows it burned all 1000 iterations and still exited with off-diagonal mass `8e-3`. | `test_convergence_tolerance_is_scale_invariant` |
| **P2** | Forming $A^\mathsf{T}A$ destroyed small singular values on correlated features. | `test_jacobi_preserves_small_singular_values` |
| **P2** | Per-query similarity ran a Python loop over every song: 23 ms vs 0.31 ms vectorised at 10k songs (**74×**). | `similarity.py` |

Plus: fuzzy title matching, `--json` output, model persistence, sign
canonicalisation, per-recommendation explanations, three rank-selection
strategies, two extra SVD backends, the evaluation harness, and CI across three
operating systems — including a Windows job that runs the suite under cp1252,
so the encoding regression cannot pass CI on a UTF-8 runner and still break on
a real desktop.

---

## Development

```bash
pip install -e ".[dev]"

pytest -q                                    # 355 tests
pytest -q --cov=eigengrooves --cov-report=term-missing
ruff check .
mypy
```

---

## References

- Gavish, M., & Donoho, D. L. (2014). *The Optimal Hard Threshold for Singular
  Values is 4/√3*. IEEE Trans. Information Theory, 60(8), 5040–5053.
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). *Finding Structure with
  Randomness*. SIAM Review, 53(2), 217–288.
- Carbonell, J., & Goldstein, J. (1998). *The Use of MMR, Diversity-Based
  Reranking for Reordering Documents and Producing Summaries*. SIGIR '98.
- Demmel, J., & Veselić, K. (1992). *Jacobi's Method is More Accurate than QR*.
  SIAM J. Matrix Analysis and Applications, 13(4), 1204–1245.

## License

MIT — see [LICENSE](LICENSE).
