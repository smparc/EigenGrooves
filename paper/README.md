# Paper

The write-up for this project: **[`paper.pdf`](paper.pdf)**.

`build/` holds LaTeX intermediates and is gitignored; `paper.pdf` is the
published copy and is committed.

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
paper.pdf             the built paper (committed)
main.tex              the source
build.py              regenerate figures, compile twice, publish paper.pdf
make_figures.py       all figures, the results table, and values.tex
build/                LaTeX intermediates (gitignored)
figures/
├── cover_genres.png        title-page image
├── dataset_table.png       Figure 1, illustrating the source data format
├── fig_*.png               generated from the current model
├── table_evaluation.tex    generated
├── values.tex              generated macros for every quoted number
└── results.json            the same numbers, machine-readable
```
