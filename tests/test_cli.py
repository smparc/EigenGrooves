"""End-to-end tests for the command-line interface."""

from __future__ import annotations

import json

import pytest

from eigengrooves.cli import main

SMALL = ["--synthetic", "--synthetic-songs", "400"]


def run(capsys, argv: list[str]) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


def test_recommend_runs_on_a_fresh_clone(capsys):
    """The headline fix: no dataset, no configuration, still produces output."""
    code, out = run(capsys, ["recommend", *SMALL, "-n", "5"])
    assert code == 0
    assert "recommendations" in out


@pytest.mark.parametrize("strategy", ["overall_top", "one_per_song", "mmr", "centroid"])
def test_every_strategy_runs(capsys, strategy):
    code, out = run(capsys, ["recommend", *SMALL, "-n", "5", "--strategy", strategy])
    assert code == 0
    assert strategy in out


def test_strategy_all_runs_every_strategy(capsys):
    code, out = run(capsys, ["recommend", *SMALL, "-n", "3", "--strategy", "all"])
    assert code == 0
    for strategy in ("overall_top", "one_per_song", "mmr", "centroid"):
        assert strategy in out


def test_json_output_is_valid_and_complete(capsys):
    code, out = run(capsys, ["recommend", *SMALL, "-n", "4", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["catalog_size"] > 0
    assert payload["model"]["k"] >= 1
    recommendations = payload["results"]["mmr"]["recommendations"]
    assert len(recommendations) == 4
    assert "explanation" in recommendations[0]


def test_explain_flag_shows_latent_reasoning(capsys):
    code, out = run(capsys, ["recommend", *SMALL, "-n", "3", "--explain"])
    assert code == 0
    assert "LF" in out


def test_analyze_reports_honest_variance(capsys):
    code, out = run(capsys, ["analyze", *SMALL, "--json"])
    assert code == 0
    payload = json.loads(out)
    retained = payload["cumulative_variance_retained"]
    assert 0.0 < retained <= 1.0
    if payload["k"] < len(payload["full_spectrum"]):
        assert retained < 1.0, "a truncated model cannot retain 100% of variance"


def test_analyze_scree_output(capsys):
    code, out = run(capsys, ["analyze", *SMALL])
    assert code == 0
    assert "Scree" in out
    assert "selected k" in out


def test_analyze_saves_a_model(capsys, tmp_path):
    path = tmp_path / "model.npz"
    code, _ = run(capsys, ["analyze", *SMALL, "--save-model", str(path)])
    assert code == 0
    assert path.exists()

    from eigengrooves.model import LatentModel

    assert LatentModel.load(path).k >= 1


def test_evaluate_produces_a_comparison_table(capsys):
    code, out = run(
        capsys,
        ["evaluate", *SMALL, "--max-groups", "25", "--min-group-size", "5"],
    )
    assert code == 0
    for expected in ("random", "popularity", "raw_cosine", "ndcg@10"):
        assert expected in out


def test_evaluate_json(capsys):
    code, out = run(
        capsys,
        ["evaluate", *SMALL, "--max-groups", "20", "--min-group-size", "5", "--json"],
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["protocol"]["n_queries"] > 0
    names = {s["name"] for s in payload["systems"]}
    assert {"random", "popularity", "raw_cosine"} <= names


@pytest.mark.parametrize("k", ["3", "variance", "elbow", "gavish_donoho"])
def test_rank_strategies_from_the_cli(capsys, k):
    code, out = run(capsys, ["analyze", *SMALL, "--k", k, "--json"])
    assert code == 0
    assert json.loads(out)["k"] >= 1


@pytest.mark.parametrize("backend", ["jacobi", "eigh", "randomized"])
def test_backends_from_the_cli(capsys, backend):
    code, _ = run(capsys, ["analyze", *SMALL, "--k", "4", "--backend", backend])
    assert code == 0


def test_whiten_flag(capsys):
    code, out = run(capsys, ["recommend", *SMALL, "-n", "3", "--whiten"])
    assert code == 0
    assert "Whitening" in out


def test_novelty_and_artist_cap_flags(capsys):
    code, _ = run(
        capsys,
        ["recommend", *SMALL, "-n", "5", "--novelty", "0.2", "--max-per-artist", "1"],
    )
    assert code == 0


def test_avoid_flag(capsys):
    from eigengrooves import make_synthetic_catalog

    catalog = make_synthetic_catalog(n_songs=400, random_state=0)
    code, _ = run(
        capsys,
        ["recommend", *SMALL, "-n", "5", "--avoid", catalog.titles[0]],
    )
    assert code == 0


def test_unresolvable_playlist_exits_nonzero(capsys):
    code, out = run(
        capsys,
        ["recommend", *SMALL, "--playlist", "zzzz nonexistent qqqq", "--exact"],
    )
    assert code == 1
    assert "Not found" in out or "No playlist" in out


def test_missing_dataset_exits_with_actionable_message(capsys, tmp_path):
    code = main(["recommend", "--data", str(tmp_path / "absent.csv")])
    assert code == 2
    assert "--synthetic" in capsys.readouterr().err


def test_fetch_data_without_a_url_explains_the_options(capsys, tmp_path):
    code, out = run(capsys, ["fetch-data", "--data", str(tmp_path / "songs.csv")])
    assert code == 1
    assert "--synthetic" in out
    assert "builder" in out.lower() or "notebook" in out.lower()


def test_no_dedup_flag_reproduces_the_v1_behaviour(capsys, tmp_path):
    """--no-dedup exists to demonstrate the old bug; it must warn loudly."""
    from eigengrooves import make_synthetic_frame

    path = tmp_path / "songs.csv"
    make_synthetic_frame(n_songs=200, n_artists=20, duplicate_rate=0.6,
                         random_state=1).to_csv(path, index=False)

    code, out = run(capsys, ["recommend", "--data", str(path), "-n", "10", "--no-dedup"])
    assert code == 0
    assert "Deduplication disabled" in out


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
