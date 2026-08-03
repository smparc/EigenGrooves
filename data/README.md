# Data

**Nothing needs to go here.** Every entry point runs against a generated
catalogue with `--synthetic`:

```bash
python main.py                        # falls back automatically
eigengrooves recommend --synthetic
eigengrooves evaluate  --synthetic
```

## Using a real catalogue

Drop any CSV here (or pass `--data path/to/file.csv`) with these columns:

**Required**
- `track_name` — aliases accepted: `track`, `song`, `title`
- `artist_names` — aliases accepted: `artist`, `artists`, `artist_name`
- `danceability`, `energy`, `speechiness`, `acousticness`,
  `instrumentalness`, `liveness`, `valence`, `loudness`, `tempo`

**Optional, used if present**
- `genre` — enables `evaluate --group-by genre`
- `popularity` or `streams` — used for novelty weighting; otherwise the count
  of collapsed duplicate rows serves as a popularity proxy

Column names are lowercased and space-stripped on load, so `Track Name` works.

```bash
python scripts/fetch_data.py                  # explains the options
python scripts/fetch_data.py --url <csv-url>  # download and validate a CSV
```

The downloader writes to a temporary path and only installs the file after its
header validates, so a truncated transfer cannot leave something that later
looks like a parsing bug.

## About the referenced dataset

The project cites [Orlandi, *Spotify Top Songs and Audio
Features*](https://github.com/JulianoOrlandi/Spotify_Top_Songs_and_Audio_Features).
That repository does **not** host a CSV — it hosts a notebook that builds one,
which needs weekly chart exports from a companion scraper project plus personal
Spotify API credentials. There is no URL to download.

Note also that Spotify restricted the `audio-features` endpoint for
newly-created API applications in late 2024. Older credentials may still work,
but a fresh application likely cannot retrieve the nine features this project
is built on. Verify current API access before committing effort to rebuilding
it.

## A note on duplicates

Chart datasets carry one row per track *per chart week*. `Catalog` collapses
these to one row per `(title, artist)` on load, aggregating features by median.

This is load-bearing, not cosmetic: without it the recommender returns the
query track back to itself at similarity `1.000000`, because excluding a
track's matched *row* leaves its other rows eligible. Note that deduplicating
by Spotify `uri` — which the upstream builder does — is **not** sufficient: the
same song commonly holds several URIs across singles, albums, remasters and
regional releases.

Contents of this directory other than this file are gitignored.
