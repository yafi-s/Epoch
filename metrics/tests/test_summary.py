"""Cross-schema rates are observed successes over one total run clock."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from metrics.summary import observed_throughput,legacy_modeled_throughput
from scripts.rank_modal_runs import _load_summary,_rank,_write_markdown


class SummaryTests(unittest.TestCase):
    def test_schema_versions_share_successful_total_clock(self):
        historical={'jobs_per_sec':46.099,'jobs_per_sec_total':45.,'completed':270,'jobs_submitted':320,'wall_clock_total_s':100.}
        modern={'metric_schema_version':2,'jobs_per_sec':6.,'completed':300,'wall_clock_s':50.,'wall_clock_total_s':100.,'legacy_modeled_jobs_per_sec':60.}
        self.assertEqual(observed_throughput(historical).jobs_per_sec,2.7)
        self.assertEqual(observed_throughput(modern).jobs_per_sec,3.)
        self.assertEqual(legacy_modeled_throughput(historical),46.099)
        self.assertEqual(legacy_modeled_throughput(modern),60.)

    def test_missing_counts_are_unknown_and_failures_are_not_successes(self):
        for summary in ({'jobs_returned':320,'jobs_submitted':320,'wall_clock_total_s':100.,'jobs_per_sec':46.099},
                        {'completed':300,'wall_clock_s':100.},
                        {'completed':float('nan'),'wall_clock_total_s':100.},
                        {'completed':300,'wall_clock_total_s':0.},
                        {'completed':-1,'wall_clock_total_s':100.}):
            self.assertIsNone(observed_throughput(summary))
        observed=observed_throughput({'completed':300,'jobs_returned':319,'jobs_submitted':320,'wall_clock_total_s':100.})
        self.assertEqual((observed.completed,observed.jobs_per_sec),(300,3.))
        self.assertEqual(observed_throughput({'completed':0,'wall_clock_total_s':100.}).jobs_per_sec,0.)

    def test_mixed_schema_ranking_uses_observed_rates_and_skips_unknown(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            summaries={'legacy':{'completed':270,'wall_clock_total_s':100.,'jobs_per_sec':46.099},
                       'modern':{'metric_schema_version':2,'completed':300,'wall_clock_total_s':100.,'wall_clock_s':50.,'jobs_per_sec':6.},
                       'unknown':{'jobs_submitted':500,'jobs_per_sec':100.}}
            runs=[]
            for name,data in summaries.items():
                path=root/(name+'_summary.json')
                path.write_text(json.dumps(data))
                with contextlib.redirect_stdout(io.StringIO()):
                    run=_load_summary(path)
                if run is not None:
                    runs.append(run)
            ranked=_rank(runs)
            self.assertEqual([r.run_id for r in ranked],['modern','legacy'])
            self.assertEqual([r.jobs_per_sec for r in ranked],[3.,2.7])
            report=root/'ranking.md'
            _write_markdown(report,ranked)
            text=report.read_text()
            self.assertIn('Completed/s (total)',text)
            self.assertNotIn('46.099',text)

if __name__=='__main__':
    unittest.main()
