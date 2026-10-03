"""Tests for benchmark aggregate reporting."""

from __future__ import annotations

import json

from benchmarks.reporter import build_aggregate_summary, write_ranked_report
from metrics.collector import MetricsStore


def _store(run_name: str, peaks: list[float]) -> MetricsStore:
    store = MetricsStore(run_name=run_name)
    for idx, peak in enumerate(peaks):
        store.record_generation(
            generation=idx,
            best_fitness=peak,
            best_so_far=peak,
            avg_fitness=peak,
            worst_fitness=peak,
            wall_clock_ms=100,
        )
    return store


def test_build_aggregate_summary_ranks_by_median_peak(tmp_path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "best_fitness_peak": 0.9,
                "wall_clock_total_s": 531.9,
                "jobs_submitted": 1440,
                "jobs_returned": 1440,
                "jobs_per_sec": 46.0,
                "raw_jobs_per_sec": 2.7,
            }
        ),
        encoding="utf-8",
    )

    results = {
        "Algo A": {
            42: _store("a42", [0.7, 0.8]),
            43: _store("a43", [0.75, 0.82]),
        },
        "Algo B": {
            42: _store("b42", [0.6, 0.65]),
            43: _store("b43", [0.62, 0.66]),
        },
    }
    summary = build_aggregate_summary(
        results=results,
        config={"seeds": [42, 43]},
        reference_summary_path=str(reference_path),
    )

    ranked = summary["ranked_algorithms"]
    assert ranked[0]["name"] == "Algo A"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["name"] == "Algo B"
    assert ranked[1]["rank"] == 2
    assert summary["reference_row"] is not None
    assert summary["reference_row"]["reference_only"] is True
    assert summary["reference_row"]["jobs_per_sec"] is None
    assert summary["reference_row"]["legacy_modeled_jobs_per_sec"] == 46.0


def test_reference_report_separates_observed_and_legacy_modeled_rates(tmp_path):
    path=tmp_path/'reference.json'
    path.write_text(json.dumps({'metric_schema_version':2,'completed':300,'jobs_submitted':320,
        'jobs_returned':319,'jobs_per_sec':6.,'legacy_modeled_jobs_per_sec':46.099,
        'wall_clock_s':50.,'wall_clock_total_s':100.}))
    summary=build_aggregate_summary(results={},config={},reference_summary_path=str(path))
    row=summary['reference_row']
    assert row['jobs_per_sec']==3. and row['completed']==300
    report=tmp_path/'report.md'
    write_ranked_report(summary=summary,output_path=report)
    text=report.read_text()
    assert 'Completed/s (total)' in text and 'Legacy modeled jobs/s' in text
    assert '3.000000' in text and '46.099000' in text


def test_reference_missing_or_invalid_clock_remains_unknown(tmp_path):
    for clock in (None,'invalid','missing'):
        data={'completed':300,'jobs_per_sec':46.099}
        if clock!='missing':
            data['wall_clock_total_s']=clock
        source=tmp_path/'reference.json'
        source.write_text(json.dumps(data))
        summary=build_aggregate_summary(results={},config={},reference_summary_path=str(source))
        assert summary['reference_row']['wall_clock_total_s'] is None
        assert summary['reference_row']['jobs_per_sec'] is None
        report=tmp_path/'report.md'
        write_ranked_report(summary=summary,output_path=report)
        assert '| unknown |' in report.read_text()


def test_write_ranked_report_creates_markdown(tmp_path):
    summary = {
        "config": {
            "space_mode": "stress",
            "dataset": "mnist",
            "seeds": [42],
            "max_wall_clock_s": 10.0,
            "budget": 0,
        },
        "ranked_algorithms": [
            {
                "rank": 1,
                "name": "Algo A",
                "median_best_fitness_peak": 0.8,
                "mean_best_fitness_peak": 0.8,
                "median_evals_completed": 10.0,
                "mean_evals_completed": 10.0,
                "median_wall_clock_s": 9.9,
                "mean_wall_clock_s": 9.9,
                "seed_runs": [
                    {
                        "seed": 42,
                        "evals_completed": 10,
                        "best_fitness_peak": 0.8,
                        "final_best_fitness": 0.8,
                        "wall_clock_s": 9.9,
                    }
                ],
            }
        ],
        "reference_row": None,
    }
    output = tmp_path / "ranked_report.md"
    write_ranked_report(summary=summary, output_path=output)
    text = output.read_text(encoding="utf-8")
    assert "Ranked Local Algorithms" in text
    assert "Algo A" in text
