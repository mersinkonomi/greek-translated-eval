"""CPU-only regression tests for mixed generation / original TruthfulQA MC2."""

import json
from html.parser import HTMLParser
from pathlib import Path
import unittest
from unittest.mock import patch

from greek_translated import report


ROOT = Path("/tmp/greek-translated-report-fixture")


def spec(mc2=True, status="completed"):
    return dict(index=0, model="Qwen/Qwen3.5-4B", benchmark="truthfulqa" if mc2 else "mmlu",
                label="TruthfulQA Greek MC2" if mc2 else "MMLU Greek", shots=0 if mc2 else 5,
                task="greektruthfulqa_mc2" if mc2 else "greek_gen_mmlu", expected_samples=2,
                status=status, output_type="multiple_choice" if mc2 else "generate_until",
                metric="acc,none" if mc2 else report.METRIC, results_dir="model",
                raw_likelihood_path="model/raw_likelihoods.jsonl" if mc2 else None,
                raw_generation_path=None if mc2 else "model/raw_generations.jsonl")


def sample_rows(mc2=True, values=None):
    values = values or ([0.25, 0.75] if mc2 else [1, 0])
    rows = []
    for index, value in enumerate(values):
        if mc2:
            rows.append(dict(doc_id=index, doc=dict(uid=str(index), question=f"Question {index}",
                                                   mc2_targets=dict(choices=["True option", "False option"], labels=[1, 0])),
                             target="0", filtered_resps=[["-2.0", "False"], ["-1.0", "False"]], acc=value))
        else:
            rows.append(dict(doc_id=index, doc=dict(uid=str(index), question=f"Question {index}",
                                                   choices=["One", "Two"], target="A", subject="mathematics"),
                             target="A", filtered_resps=["A" if value == 1 else "[invalid]"],
                             resps=[["Reasoning\nΤελική απάντηση: \\boxed{A}" if value == 1 else "unfinished"]], exact_match=value))
    return rows


def scored(mc2=True, values=None):
    with patch.object(report, "jsonl_rows", return_value=enumerate(sample_rows(mc2, values), 1)):
        return report.read_samples([ROOT / "samples.jsonl"], [], spec(mc2))


def assemble_fixture(mc2=True, values=None, aggregate=0.5, status="completed"):
    item = spec(mc2, status)
    manifest = dict(run_id="fixture", runs=[item], suite={"benchmarks": []})
    results = dict(results={item["task"]: {item["metric"]: aggregate, report.stderr_name(item): 0.1}},
                   **{"n-samples": {item["task"]: {"original": 2, "effective": 2}}}, config={"limit": None})
    samples = scored(mc2, values)
    with patch.object(report, "read_json", return_value=manifest), \
            patch.object(Path, "is_file", return_value=False), \
            patch.object(report, "discover_result", return_value=(ROOT / "results.json", results, [ROOT / "samples.jsonl"])), \
            patch.object(report, "read_samples", return_value=samples):
        return report.assemble(ROOT, ROOT / "manifest.json")


class PayloadParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.payload = ""
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts += 1
            self.in_data = dict(attrs).get("id") == "report-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.payload += data


class ReportTests(unittest.TestCase):
    def test_native_title_and_pilot_are_not_translated_full_results(self):
        manifest, runs, warnings = assemble_fixture(mc2=False)
        runs[0]["item"].update(benchmark="native_greekmmlu_0shot", label="Native GreekMMLU", shots=0)
        manifest["pilot"] = True
        page = report.render(ROOT, ROOT / "manifest.json", ROOT / "report.html", manifest, runs, warnings)
        self.assertIn("PILOT ONLY · Native GreekMMLU", page)
        self.assertIn("not full benchmark results", page)
        self.assertIn("native dascim/GreekMMLU", page)
        self.assertNotIn("These are translated Greek datasets, not native GreekMMLU", page)

    def test_mc2_fractional_scores_are_not_binary(self):
        result = scored()
        self.assertEqual(result["scored"], 2)
        self.assertEqual(result["score_sum"], 1)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["invalid"], 0)
        for sample in result["examples"]:
            self.assertIsNone(sample["correct"])
            self.assertIsNone(sample["prediction"])
            self.assertEqual(sample["response"], "")
            self.assertEqual(sample["choice_labels"], [1, 0])

    def test_mc2_nonfinite_or_out_of_range_scores_rejected(self):
        result = scored(values=[float("nan"), float("inf"), -0.1, 1.1])
        self.assertEqual(result["unique"], 4)
        self.assertEqual(result["scored"], 0)
        self.assertEqual(result["score_sum"], 0)

    def test_completed_mc2_mean_validates(self):
        _, runs, warnings = assemble_fixture()
        self.assertEqual(warnings, [])
        self.assertTrue(runs[0]["completed"])
        self.assertEqual(runs[0]["accuracy"], 0.5)
        cell = report.score_cell(runs[0])
        self.assertIn("MC2 mean probability mass", cell)
        self.assertNotIn("correct", cell)

    def test_mc2_missing_metrics_and_mean_mismatch_fail_validation(self):
        for values, aggregate in (([float("nan"), 0.75], 0.5), ([0.25, 0.75], 0.7)):
            with self.subTest(values=values, aggregate=aggregate):
                _, runs, _ = assemble_fixture(values=values, aggregate=aggregate)
                self.assertEqual(runs[0]["state"], "validation error")
                self.assertIsNone(runs[0]["accuracy"])

    def test_pending_mc2_does_not_show_zero_or_generation_progress(self):
        _, runs, _ = assemble_fixture(status="running")
        cell = report.score_cell(runs[0])
        self.assertIsNone(runs[0]["accuracy"])
        self.assertIn("option likelihoods saved", cell)
        self.assertNotIn("generations", cell)
        self.assertNotIn("0.00%", cell)

    def test_likelihood_telemetry_deduplicates_options_not_questions(self):
        rows = [dict(request_hash="a", uid="q1", task_name="greektruthfulqa_mc2", option_index=0,
                     context="context", continuation="true", continuation_token_ids=[1, 2],
                     continuation_token_logprobs=[-0.5, -1.0], total_loglikelihood=-1.5),
                dict(request_hash="b", uid="q1", option_index=1, task_name="greektruthfulqa_mc2",
                     continuation_token_ids=[3], total_loglikelihood=-2.0)]
        rows.append(rows[0])
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(report, "jsonl_rows", return_value=enumerate(rows, 1)):
            result = report.read_likelihood_telemetry(ROOT / "raw_likelihoods.jsonl", [])
        self.assertEqual(result["unique"], 2)
        self.assertEqual(result["question_ids"], 1)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["likelihood_tokens"], 3)
        self.assertEqual(result["finite_loglikelihoods"], 2)
        self.assertEqual(result["token_total"], 0)
        self.assertEqual(result["final_known"], 0)
        self.assertEqual(result["examples"][0]["source"], "raw-likelihood")

    def test_generative_regression(self):
        _, runs, warnings = assemble_fixture(mc2=False)
        self.assertEqual(warnings, [])
        self.assertTrue(runs[0]["completed"])
        self.assertEqual(runs[0]["samples"]["correct"], 1)
        self.assertEqual(runs[0]["samples"]["invalid"], 1)
        self.assertIn("1 / 2 correct", report.score_cell(runs[0]))

    def test_mixed_html_separates_protocols_and_embeds_mc2_safely(self):
        manifest, mc2_runs, warnings = assemble_fixture()
        gen_manifest, gen_runs, _ = assemble_fixture(mc2=False)
        manifest["runs"].extend(gen_manifest["runs"])
        samples = mc2_runs[0]["samples"]["examples"]
        samples[0]["question"] = "</script><img src=x onerror=alert(1)>"
        page = report.render(ROOT, ROOT / "manifest.json", ROOT / "report.html", manifest, mc2_runs + gen_runs, warnings)
        self.assertIn("MIXED PROTOCOL", page)
        self.assertIn("ORIGINAL MC2 · LIKELIHOOD", page)
        self.assertIn("GENERATIVE · EXACT MATCH", page)
        self.assertIn("original 6 fixed priming", page)
        self.assertNotIn("MC1", page)
        self.assertNotIn("never from likelihoods", page)
        self.assertNotIn("</script><img", page)
        table = page.split("<h2>Generation completion and answer format</h2>")[1].split("<h2>TruthfulQA MC2 likelihood diagnostics</h2>")[0]
        self.assertNotIn("TruthfulQA Greek MC2", table)
        parser = PayloadParser()
        parser.feed(page)
        self.assertEqual(parser.scripts, 2)
        payload = json.loads(parser.payload)
        mc2_samples = [sample for sample in payload["samples"] if sample.get("scoring_mode") == "mc2"]
        self.assertEqual(len(mc2_samples), 2)
        self.assertEqual(mc2_samples[0]["question"], samples[0]["question"])

    def test_default_generative_mode_and_bounded_previews(self):
        self.assertFalse(report.is_mc2({}))
        self.assertEqual(report.metric_name({}), report.METRIC)
        self.assertLessEqual(len(report.preview("x" * 100_000)), report.PREVIEW_LIMIT)


if __name__ == "__main__":
    unittest.main()
