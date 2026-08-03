# Paper

Revised write-up of the project: `build/main.pdf`.

## Building

```bash
python paper/build.py                              # synthetic catalogue
python paper/build.py --data data/spotify_songs.csv
python paper/build.py --skip-figures               # LaTeX only
```

Requires a TeX distribution (MiKTeX, TeX Live) plus `matplotlib`
(`pip install -e ".[viz]"`). Without TeX, `make_figures.py` still runs and the
figures land in `figures/`.

## Why nothing here is typed by hand

`make_figures.py` regenerates every figure, the results table
(`figures/table_evaluation.tex`) and every number quoted in the prose
(`figures/values.tex`, a set of LaTeX macros) from a live run of the package.
`main.tex` references those macros rather than literals.

The consequence: the paper cannot silently disagree with the code. Change `k`,
the scaling, or the ranking, re-run the build, and the prose updates with the
figures — or fails loudly. Point it at a different catalogue and the whole
document re-derives.

## Files

```
main.tex              the paper
build.py              regenerate figures, then compile twice
make_figures.py       all figures, the results table, and values.tex
figures/
├── cover_genres.png        carried over from the original paper
├── dataset_table.png       carried over from the original paper
├── orig_projection_3d.png  the original Figure 2, kept for reference
├── fig_*.png               regenerated from the current model
├── table_evaluation.tex    generated
├── values.tex              generated macros for every quoted number
└── results.json            the same numbers, machine-readable
```

## Relationship to the original

The original (March 2025) is preserved in intent and structure — same research
question, same cover image and dataset figure, same original-vs-latent-space
visualisation. What changed:

- The central claim is now tested rather than asserted, with baselines and
  paired significance tests. The result is negative.
- Section 8 answers the research question the original abstract posed but never
  measured.
- Section 9 documents four defects in the original pipeline, one of which is
  visible in the original paper's own published results.
