import os
import json
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

from greek_translated.protocol import INVALID, answer_format, build_prompt, extract_final_answer, extract_final_answers, reasoning_is_closed
from greek_translated.prepare_data import HERE, PROJECT, TRUTHFULQA_DIR, _load, attach_prompts, canonicalize, load_truthfulqa_mc2_task, truthfulqa_original_hashes


class ExtractionTests(unittest.TestCase):
    def test_box_variants(self):
        examples = [r"Τελική απάντηση: \boxed{C}", r"\boxed{C}", r"Τελική απάντηση: $\boxed{\text{C}}$", r"Τελική απάντηση: \[\boxed{\mathrm{C}}\]", r"Τελική απάντηση: \boxed{Γ}", r"Τελική απάντηση: \boxed{\Gamma}"]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(extract_final_answer(text, 4), "C")

    def test_unboxed_requires_marker(self):
        self.assertEqual(extract_final_answer("Τελική απάντηση: D", 4), "D")
        self.assertEqual(answer_format("Τελική απάντηση: D", 4), "marked-letter")
        self.assertEqual(extract_final_answer("D", 4), INVALID)
        self.assertEqual(extract_final_answer("I think D", 4), INVALID)

    def test_thinking(self):
        self.assertEqual(extract_final_answer("analysis\nΤελική απάντηση: \\boxed{A}", 4, True), INVALID)
        self.assertEqual(extract_final_answer("analysis\n</think>\nΤελική απάντηση: \\boxed{A}", 4, True), "A")
        self.assertEqual(extract_final_answer("<think>\nΤελική απάντηση: \\boxed{A}", 4), INVALID)
        self.assertEqual(extract_final_answer("<think>\nΤελική απάντηση: \\boxed{A}\n</think>", 4), INVALID)
        self.assertTrue(reasoning_is_closed("abc</ifm|think>"))
        self.assertFalse(reasoning_is_closed("abc"))

    def test_terminal_only_and_range(self):
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{A}\nnot final", 4), INVALID)
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{J}", 10), "J")
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{M}", 13), "M")
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{M}", 4), INVALID)
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{A,B}", 4), INVALID)

    def test_new_question(self):
        text = "Τελική απάντηση: \\boxed{A}\n\nΕρώτηση: νέα\nΤελική απάντηση: \\boxed{B}"
        self.assertEqual(extract_final_answer(text, 4), "A")
        text = "Ερώτηση: quoted inside thinking\n</think>\nΤελική απάντηση: \\boxed{C}"
        self.assertEqual(extract_final_answer(text, 4, True), "C")

    def test_filter_native_env(self):
        with patch.dict(os.environ, {"GREEK_TRANSLATED_REQUIRE_THINKING_CLOSE": "1"}):
            self.assertEqual(extract_final_answers([[r"\boxed{A}"]], [{"choices": ["a", "b"]}]), [[INVALID]])


class PromptTests(unittest.TestCase):
    def test_current_gold_independent(self):
        doc = {"question": "QUESTION_SENTINEL", "choices": ["x", "y", "z", "w"], "target": "A", "cot_content": "GOLD_RATIONALE_SENTINEL"}
        before = build_prompt(doc, [])
        self.assertEqual(before, build_prompt(dict(doc, target="D", cot_content="OTHER_RATIONALE"), []))
        self.assertNotIn("GOLD_RATIONALE_SENTINEL", before)
        self.assertIn("QUESTION_SENTINEL", before)

    def test_demonstration_targets_only(self):
        doc = {"question": "current", "choices": ["x", "y"], "target": "A"}
        example = {"question": "example", "choices": ["x", "y"], "target": "B"}
        prompt = build_prompt(doc, [example])
        self.assertIn(r"Τελική απάντηση: \boxed{B}", prompt)
        self.assertNotIn(r"Τελική απάντηση: \boxed{A}", prompt)

    def test_truthfulqa_source_rows_and_choice_order_preserved(self):
        rows = [{"question": f" Question {i} ", "question_en": "original", "mc1_targets": {"choices": ["true", "false", "false2"], "labels": [1, 0, 0]}, "mc2_targets": {"choices": ["true1", "true2", "false"], "labels": [1, 1, 0]}} for i in range(5)]
        docs = canonicalize("truthfulqa", "ilsp/truthful_qa_greek", "multiple_choice", "train", rows)
        for row, doc in zip(rows, docs):
            self.assertEqual({k:v for k,v in doc.items() if k != "uid"}, row)
            self.assertNotIn("prompt", doc)
            self.assertNotIn("option_permutation", doc)
        with self.assertRaises(ValueError):
            attach_prompts("truthfulqa", docs, [], 0)

    def test_belebele_passage_exclusion(self):
        docs = [{"uid": str(i), "question": str(i), "passage": str(i // 2), "source_link": str(i // 2), "choices": ["x", "y"], "target": "A"} for i in range(10)]
        attach_prompts("belebele", docs, docs, 5)
        for doc in docs:
            self.assertEqual(len(doc["demonstration_ids"]), 5)
            for uid in doc["demonstration_ids"]:
                example = next(x for x in docs if x["uid"] == uid)
                self.assertNotEqual(example["passage"], doc["passage"])

    def test_arc_numeric_keys_and_five_choices(self):
        row = {"id": "x", "question": "q", "choices": {"label": ["1", "2", "3", "4", "5"], "text": list("abcde")}, "answerKey": "5"}
        self.assertEqual(canonicalize("arc_easy", "ilsp/arc_greek", "ARC-Easy", "test", [row])[0]["target"], "E")


INTEGRATION_SUITE = Path(os.environ["GREEK_EVAL_TEST_SUITE"]).resolve() if os.environ.get("GREEK_EVAL_TEST_SUITE") else None


@unittest.skipUnless(INTEGRATION_SUITE and INTEGRATION_SUITE.is_file(),
                     "Set GREEK_EVAL_TEST_SUITE to a prepared translated suite for full cached-data integration checks")
class OriginalTruthfulQAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(INTEGRATION_SUITE.read_text())
        cls.spec = next(x for x in cls.suite["benchmarks"] if x["id"] == "truthfulqa")
        cls.task = load_truthfulqa_mc2_task(data_path=cls.spec["data_path"],
            cache_dir=INTEGRATION_SUITE.parent / "cache/test_task_loader",
            expected_hashes=cls.spec["original_task_hashes"])
        cache = Path(os.environ.get("GREEK_EVAL_TEST_CACHE", str(INTEGRATION_SUITE.parent / "cache/datasets")))
        cls.sources, _ = _load("ilsp/truthful_qa_greek", "multiple_choice", cache, offline=True)

    def test_original_files_match_git(self):
        for path in truthfulqa_original_hashes():
            original = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=PROJECT)
            self.assertEqual((PROJECT / path).read_bytes(), original)

    def test_original_configuration(self):
        from lm_eval.utils import load_yaml_config
        config = load_yaml_config(str(TRUTHFULQA_DIR / "greektruthfulqa_mc2.yaml"))
        self.assertEqual(self.task.config.task, "greektruthfulqa_mc2")
        self.assertEqual(self.task.OUTPUT_TYPE, "multiple_choice")
        self.assertEqual(self.task.config.num_fewshot, 0)
        self.assertEqual(self.task.config.doc_to_text, config["doc_to_text"])
        self.assertEqual(self.task.config.doc_to_choice, config["doc_to_choice"])
        self.assertEqual(self.task.config.doc_to_target, config["doc_to_target"])
        self.assertEqual(self.task.config.metric_list, config["metric_list"])
        self.assertEqual([f.name for f in self.task._filters], ["none"])

    def test_all_source_rows_and_prompts_preserved(self):
        from lm_eval.utils import apply_template
        self.assertEqual(len(self.task.eval_docs), 817)
        for source, doc in zip(self.sources["train"], self.task.eval_docs):
            self.assertEqual({k:v for k,v in doc.items() if k != "uid"}, source)
            self.assertEqual(self.task.doc_to_choice(doc), source["mc2_targets"]["choices"])
            prompt = self.task.doc_to_text(doc)
            self.assertEqual(prompt, apply_template(self.task.config.doc_to_text, source))
            self.assertEqual(len(re.findall(r"(?m)^Q:", prompt)), 7)
            self.assertNotIn("\\boxed", prompt)

    def test_probability_mass_not_generated_letter_accuracy(self):
        import numpy as np
        for doc in list(self.task.eval_docs)[:10]:
            labels = np.array(doc["mc2_targets"]["labels"])
            ll = np.linspace(-1.0, -3.0, len(labels))
            expected = float(np.sum(np.exp(ll)[labels == 1]) / np.sum(np.exp(ll)))
            actual = self.task.process_results(doc, [(float(x), False) for x in ll])["acc"]
            self.assertAlmostEqual(actual, expected)
            self.assertGreater(actual, 0)
            self.assertLess(actual, 1)

    def test_suite_declares_unchanged_mc2_exception(self):
        self.assertEqual(self.spec["task"], "greektruthfulqa_mc2")
        self.assertEqual(self.spec["metric_key"], "acc,none")
        self.assertFalse(self.spec["binary_metric"])
        self.assertFalse(self.spec["apply_chat_template"])
        self.assertEqual(self.spec["fixed_priming_examples"], 6)
        self.assertEqual(len(list((INTEGRATION_SUITE.parent / "tasks").glob("*.yaml"))), 7)


if __name__ == "__main__":
    unittest.main()
