# Loop'd Loop — SVD & Dimensionality Reduction

> Discover music beyond genre labels — powered by latent feature analysis using Singular Value Decomposition.

---

## Overview

Traditional music recommendation leans heavily on genre tags and listening history. This project takes a different approach: it decomposes songs into **latent audio features** using SVD, then recommends music based on deeper mathematical similarity rather than surface-level metadata.

Given an input playlist, the system projects every song into a reduced-dimensional latent space and surfaces tracks that are most similar in terms of actual sonic character — energy, rhythm, vocal texture, and more.

---

## Features

- **SVD-Based Feature Projection** — Decomposes a song feature matrix into latent components that capture underlying musical patterns
- **Z-Score Normalization** — Ensures all audio features contribute equally to similarity calculations
- **Two Recommendation Strategies**
  - *One-Per-Song*: Maximizes diversity by sourcing one recommendation per input track
  - *Overall Top*: Maximizes coherence by returning the globally most-similar songs
- **Cosine Similarity Ranking** — Compares songs in latent space for musically meaningful matching
- **Latent Feature Interpretation** — Extracts human-readable dimensions like "energy-loudness" or "vocal focus"

---

## How It Works

### 1. Feature Extraction
Songs are represented as vectors across 9 Spotify audio features:

| Feature | Description |
|---|---|
| Danceability | Rhythmic suitability for dancing |
| Energy | Perceptual intensity and activity |
| Speechiness | Presence of spoken words |
| Acousticness | Confidence the track is acoustic |
| Instrumentalness | Predicts absence of vocals |
| Liveness | Presence of a live audience |
| Valence | Musical positivity |
| Loudness | Overall loudness in dB |
| Tempo | Estimated beats per minute |

### 2. SVD Decomposition
The feature matrix **A** is decomposed as:

$$A = U \Sigma V^T$$

- **U** — Left singular vectors (song representations)
- **Σ** — Diagonal matrix of singular values (variance explained per component)
- **V** — Right singular vectors (feature combination weights)

### 3. Dimensionality Reduction
Songs are projected into a **k = 5** dimensional latent space:

$$B_k = B \cdot V_k$$

This is mathematically equivalent to projecting each song vector onto the orthonormal basis defined by the top-k right singular vectors.

### 4. Similarity & Recommendation
Cosine similarity is computed between all songs in the latent space. The top matches are returned according to the chosen recommendation strategy.

---

## Example

**Input Playlist:** 21 SZA-centric tracks (e.g., *Kill Bill*, *Saturn*, *I Hate U*)

**Sample Latent Features Discovered:**

| Latent Feature | Interpretation |
|---|---|
| LF1 | Energy-Loudness dimension (high energy, loud, low acousticness) |
| LF2 | Melodic focus (low danceability, low speechiness) |
| LF3 | Rhythmic-vocal delivery (high tempo + speechiness) |
| LF4 | Studio vs. Live feel |
| LF5 | Vocal-forward character (strongly penalizes instrumentalness) |

**Sample Recommendations (Overall Top method):**
- *when the party's over* — Billie Eilish
- *Hold On* — Adele
- *pete davidson* — Ariana Grande
- *City of Stars* — Ryan Gosling, Emma Stone
- *It's Nice to Have a Friend* — Taylor Swift

The system surfaced cross-genre matches unified by shared latent sonic characteristics, rather than simply returning more SZA songs.

---

## Tech Stack

- **Language:** Python
- **Dataset:** [Spotify Top Songs & Audio Features](https://github.com/JulianoOrlandi/Spotify_Top_Songs_and_Audio_Features)
- **Core Math:** Custom SVD, projection, and cosine similarity implementations (minimal library use)
- **Libraries:** NumPy (limited), Matplotlib (visualization)

---

## Project Structure

```
music-recommender/
├── data/
│   └── spotify_songs.csv
├── src/
│   ├── normalization.py      # Z-score normalization
│   ├── svd.py                # Custom SVD implementation
│   ├── projection.py         # Feature projection & similarity
│   └── recommend.py          # One-per-song & overall top methods
├── notebooks/
│   └── analysis.ipynb        # Visualization & exploration
└── README.md
```

---

## References

- Orlandi, J. *Spotify Top Songs and Audio Features*. GitHub, 2024.
- Kim, G. B. *SVD Mathematical Definitions*. Stanford Statistics Department.
