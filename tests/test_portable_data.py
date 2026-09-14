"""CPU-only portable data/protocol checks. Never contact the Hub."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
import re
from unittest.mock import patch

from greek_translated import prepare_data as data


class PortableDataTests(unittest.TestCase):
    def test_pins_match_all_eight_benchmarks_and_counts(self):
        pins = data.dataset_pins()
        self.assertEqual(len(pins["benchmarks"]), 8)
        self.assertEqual(sum(spec[6] for spec in data.SPECS), 55401)
        self.assertEqual(pins["expected_samples_per_model"], 55401)
        self.assertEqual([spec[5] for spec in data.SPECS], [5, 0, 5, 25, 25, 10, 0, 5])
        self.assertNotIn("/shared/", data.PINS_PATH.read_text())
        self.assertNotIn("cache_path", data.PINS_PATH.read_text())

    def test_mmlu_configs_come_from_upstream_yaml(self):
        subjects = data.mmlu_subjects()
        self.assertEqual(len(subjects), 57)
        self.assertEqual(subjects[0], "abstract_algebra")
        self.assertEqual(subjects[-1], "world_religions")

    def test_parallel_mmlu_loading_preserves_subject_order(self):
        subjects = ["anatomy", "abstract_algebra", "world_religions"]
        with patch.object(data, "_load", side_effect=lambda repo, subject, cache, offline: ({"subject": subject}, {})):
            output = list(data._mmlu_datasets("ilsp/mmlu_greek", subjects, Path("cache"), download_workers=3))
        self.assertEqual([subject for subject, _ in output], subjects)
        self.assertEqual([dataset["subject"] for _, (dataset, _) in output], subjects)

    def test_load_always_passes_pinned_revision_and_selected_cache(self):
        calls = []
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.config = types.SimpleNamespace(HF_DATASETS_OFFLINE=False, HF_HUB_OFFLINE=False)
        fake_datasets.DownloadConfig = lambda **kwargs: kwargs
        fake_datasets.load_dataset = lambda *args, **kwargs: calls.append((args, kwargs)) or {"test": []}
        with patch.dict("sys.modules", {"datasets": fake_datasets}):
            result, provenance = data._load("ilsp/mmlu_greek", "anatomy", Path("my-cache"))
            data._load("ilsp/mmlu_greek", "anatomy", Path("my-cache"), offline=True)
        self.assertEqual(result, {"test": []})
        args, kwargs = calls[0]
        self.assertEqual(args, ("ilsp/mmlu_greek", "anatomy"))
        self.assertEqual(kwargs, {"revision": "5185752b1015abb6bb96c6ed7e860b92a194484a",
                                  "cache_dir": "my-cache"})
        self.assertEqual(calls[1][1]["download_config"], {"local_files_only": True})
        self.assertEqual(provenance["revision"], kwargs["revision"])
        self.assertNotIn("cache_path", provenance)
        self.assertFalse(fake_datasets.config.HF_DATASETS_OFFLINE)
        self.assertFalse(fake_datasets.config.HF_HUB_OFFLINE)

    def test_offline_mode_disables_metadata_requests_and_restores_caller(self):
        from huggingface_hub import constants
        previous_hub_offline = constants.HF_HUB_OFFLINE
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.config = types.SimpleNamespace(HF_DATASETS_OFFLINE=False, HF_HUB_OFFLINE=False)
        with patch.dict("sys.modules", {"datasets": fake_datasets}), \
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"}):
            with self.assertRaisesRegex(RuntimeError, "cache miss"), data.offline_mode(True):
                self.assertTrue(fake_datasets.config.HF_HUB_OFFLINE)
                self.assertTrue(constants.HF_HUB_OFFLINE)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                raise RuntimeError("cache miss")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertFalse(fake_datasets.config.HF_HUB_OFFLINE)
            self.assertEqual(constants.HF_HUB_OFFLINE, previous_hub_offline)

    def test_unknown_dataset_never_resolves_mutable_main(self):
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.DownloadConfig = lambda **kwargs: kwargs
        fake_datasets.load_dataset = lambda *args, **kwargs: self.fail("must not load unpinned input")
        with patch.dict("sys.modules", {"datasets": fake_datasets}):
            with self.assertRaisesRegex(ValueError, "No unique pinned dataset revision"):
                data._load("unknown/dataset", "default", Path("cache"))

    def test_cache_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(data._cache_directory(root), root / "cache/datasets")
            with patch.dict(os.environ, {"HF_HOME": str(root / "hf")}, clear=True):
                self.assertEqual(data._cache_directory(root), root / "hf/datasets")
            with patch.dict(os.environ, {"HF_HOME": str(root / "hf"),
                                         "HF_DATASETS_CACHE": str(root / "env-cache")}, clear=True):
                self.assertEqual(data._cache_directory(root), root / "env-cache")
                self.assertEqual(data._cache_directory(root, root / "explicit"), root / "explicit")

    def test_yaml_respects_output_dir_and_imports_portably(self):
        with tempfile.TemporaryDirectory(prefix="greek suite ") as directory:
            root = Path(directory)
            task, yaml_path = data._write_yaml("mmlu", root / "data/mmlu.jsonl", 5, root)
            self.assertEqual(task, "greek_gen_mmlu")
            self.assertEqual(yaml_path.parent, root / "tasks")
            text = yaml_path.read_text()
            self.assertIn("output_type: generate_until", text)
            self.assertIn("actual_num_fewshot: 5", text)
            self.assertIn("num_fewshot: 0", text)
            self.assertIn("doc_to_text: !function protocol.doc_to_text", text)
            self.assertNotIn(str(data.PROJECT), text)
            shim = yaml_path.parent / "protocol.py"
            self.assertEqual(shim.read_text(),
                             "from greek_translated.protocol import doc_to_text, doc_to_target, extract_final_answers\n")
            namespace = {}
            exec(compile(shim.read_text(), str(shim), "exec"), namespace)
            self.assertEqual(namespace["doc_to_text"]({"prompt": "Greek prompt"}), "Greek prompt")
            self.assertEqual(namespace["extract_final_answers"](
                [["Τελική απάντηση: \\boxed{B}"]], [{"choices": ["one", "two"]}]), [["B"]])

    def test_task_hashes_are_stable_between_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes = []
            raw_hashes = []
            for name in ("first-machine", "second machine"):
                root = Path(directory) / name
                data_path = root / "data/mmlu.jsonl"
                _, yaml_path = data._write_yaml("mmlu", data_path, 5, root)
                hashes.append(data.portable_task_hash(yaml_path, data_path))
                raw_hashes.append(data._sha(yaml_path))
            self.assertEqual(hashes[0], hashes[1])
            self.assertNotEqual(raw_hashes[0], raw_hashes[1])

    def test_truthfulqa_source_hash_keys_are_checkout_relative(self):
        hashes = data.truthfulqa_original_hashes()
        self.assertEqual(hashes, {
            "lm_eval/tasks/greektruthfulqa/greektruthfulqa_mc1.yaml":
                "aa23dee6f57b3ec5471d66dcc3804534bf76b4b316e5076aa08d36e28dc1e1a2",
            "lm_eval/tasks/greektruthfulqa/greektruthfulqa_mc2.yaml":
                "c924d1d13ac0cb8ad6aec000299e2fee070c285f7146f1d8190e0001c49c4a67",
            "lm_eval/tasks/greektruthfulqa/utils.py":
                "d3c827d1e2cd442a1e4738b527a18effd48cbb2aed478c12a652311b5ea4dbc8",
        })
        self.assertTrue(all(not Path(key).is_absolute() for key in hashes))

    def test_truthfulqa_loader_changes_only_dataset_transport(self):
        processor = object()
        original = {"task": "greektruthfulqa_mc2", "output_type": "multiple_choice",
                    "doc_to_text": "The original six fixed QA demonstrations",
                    "doc_to_choice": "{{mc2_targets.choices}}", "process_results": processor,
                    "dataset_path": "ilsp/truthful_qa_greek", "dataset_name": "multiple_choice",
                    "metric_list": [{"metric": "acc", "aggregation": "mean"}], "num_fewshot": 0}
        task_module = types.ModuleType("lm_eval.api.task")
        task_module.ConfigurableTask = lambda config: config
        utils_module = types.ModuleType("lm_eval.utils")
        utils_module.load_yaml_config = lambda path: dict(original)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {
            "lm_eval.api.task": task_module, "lm_eval.utils": utils_module,
        }):
            source = Path(directory) / "truthfulqa.jsonl"
            cache = Path(directory) / "cache"
            config = data.load_truthfulqa_mc2_task(source, cache, data.truthfulqa_original_hashes())
        for key in original.keys() - {"dataset_path", "dataset_name", "dataset_kwargs"}:
            self.assertEqual(config[key], original[key])
        self.assertIs(config["process_results"], processor)
        self.assertEqual(config["dataset_path"], "json")
        self.assertIsNone(config["dataset_name"])
        self.assertEqual(config["dataset_kwargs"], {
            "data_files": {"train": str(source)}, "cache_dir": str(cache)})

    def test_actual_truthfulqa_task_preserves_prompt_order_and_mc2_scoring(self):
        # Exercise the real ConfigurableTask and original YAML/function loader
        # on invented rows only; no public dataset or GPU is needed.
        from lm_eval.utils import apply_template, load_yaml_config
        source_row = {
            "question": "Ποια από τις επιλογές είναι σωστή;",
            "mc1_targets": {"choices": ["Σωστή", "Λάθος"], "labels": [1, 0]},
            "mc2_targets": {"choices": ["Λάθος πρώτη", "Σωστή δεύτερη", "Σωστή τρίτη"],
                            "labels": [0, 1, 1]},
        }
        row = data.canonicalize("truthfulqa", "ilsp/truthful_qa_greek", "multiple_choice",
                                "train", [source_row])[0]
        with tempfile.TemporaryDirectory(prefix="truthfulqa cache ") as directory:
            root = Path(directory)
            source_path = root / "truthfulqa.jsonl"
            source_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            hashes = data.truthfulqa_original_hashes()
            with data.offline_mode(True):
                task = data.load_truthfulqa_mc2_task(source_path, root / "cache", hashes)
            original = load_yaml_config(str(data.TRUTHFULQA_DIR / "greektruthfulqa_mc2.yaml"))
            self.assertEqual(len(task.eval_docs), 1)
            doc = task.eval_docs[0]
            self.assertEqual({key: value for key, value in doc.items() if key != "uid"}, source_row)
            self.assertEqual(task.config.task, "greektruthfulqa_mc2")
            self.assertEqual(task.OUTPUT_TYPE, "multiple_choice")
            self.assertEqual(task.config.num_fewshot, 0)
            self.assertEqual([item.name for item in task._filters], ["none"])
            self.assertEqual(task.doc_to_choice(doc), source_row["mc2_targets"]["choices"])
            prompt = task.doc_to_text(doc)
            self.assertEqual(prompt, apply_template(original["doc_to_text"], source_row))
            self.assertEqual(len(re.findall(r"(?m)^Q:", prompt)), 7)
            self.assertNotIn("\\boxed", prompt)
            likelihoods = [-1.0, -2.0, -3.0]
            expected = sum(math.exp(value) for value in likelihoods[1:]) / sum(
                math.exp(value) for value in likelihoods)
            actual = task.process_results(doc, [(value, False) for value in likelihoods])["acc"]
            self.assertAlmostEqual(actual, expected)
            self.assertEqual(data.truthfulqa_original_hashes(), hashes)

    def test_mmlu_pro_uses_answer_letter_without_gold_reasoning(self):
        row = {"question": "Ερώτηση", "options": ["Πρώτο", "Δεύτερο", "Τρίτο"],
               "answer": "C", "answer_index": 1, "category": "math", "question_id": 3983,
               "cot_content": "SECRET GOLD REASONING"}
        docs = data.canonicalize("mmlu_pro", "ilsp/MMLU-Pro_greek", "default", "test", [row])
        data.attach_prompts("mmlu_pro", docs, [], 0)
        self.assertEqual(docs[0]["target"], "C")
        self.assertTrue(docs[0]["source_answer_index_mismatch"])
        self.assertNotIn("SECRET GOLD REASONING", docs[0]["prompt"])
        self.assertNotIn("cot_content", docs[0])

    def test_belebele_excludes_same_passage_and_source_link(self):
        def doc(index, passage, link):
            result = data._base("facebook/belebele", "ell_Grek", "test", index,
                                "Ερώτηση", ["Απάντηση 1", "Απάντηση 2"], 0, passage=passage)
            result["source_link"] = link
            return result
        question = doc(0, "passage0", "link0")
        pool = [question, doc(1, "passage0", "link1"), doc(2, "passage2", "link0")]
        pool.extend(doc(index, f"passage{index}", f"link{index}") for index in range(3, 8))
        data.attach_prompts("belebele", [question], pool, 5)
        self.assertEqual(question["demonstration_ids"], [item["uid"] for item in pool[3:]])

    def test_preparation_rejects_wrong_normalized_digest_before_replacing_data(self):
        row = {"question": "Ερώτηση", "options": ["one", "two"], "answer": "A",
               "answer_index": 0, "category": "math", "question_id": 1}
        spec = ("mmlu_pro", "MMLU-Pro Greek", "ilsp/MMLU-Pro_greek", "default", "test", 0, 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "SPECS", [spec]), \
                patch.object(data, "dataset_pins", return_value={"benchmarks": {"mmlu_pro": {"data_sha256": "0" * 64}}}), \
                patch.object(data, "_load", return_value=({"test": [row]}, {"revision": "1" * 40})):
            root = Path(directory)
            (root / "data").mkdir()
            final_path = root / "data/mmlu_pro.jsonl"
            final_path.write_text("previous verified artifact\n")
            with self.assertRaisesRegex(ValueError, "normalized input SHA256 differs"):
                data.prepare_all(root)
            self.assertEqual(final_path.read_text(), "previous verified artifact\n")
            self.assertFalse((root / "suite.json").exists())

    def test_prepared_manifest_uses_runtime_paths_and_portable_provenance(self):
        row = {"question": "Ερώτηση", "options": ["one", "two"], "answer": "A",
               "answer_index": 0, "category": "math", "question_id": 1}
        spec = ("mmlu_pro", "MMLU-Pro Greek", "ilsp/MMLU-Pro_greek", "default", "test", 0, 1)
        expected_docs = data.canonicalize("mmlu_pro", spec[2], spec[3], spec[4], [row])
        data.attach_prompts("mmlu_pro", expected_docs, [], 0)
        serialized = "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in expected_docs)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "SPECS", [spec]), \
                patch.object(data, "dataset_pins", return_value={"benchmarks": {"mmlu_pro": {"data_sha256": digest}}}), \
                patch.object(data, "_load", return_value=({"test": [row]}, {"revision": "1" * 40})):
            root = Path(directory)
            suite = data.prepare_all(root)
            saved = json.loads((root / "suite.json").read_text())
            self.assertEqual(saved, json.loads(json.dumps(suite)))
            self.assertEqual(suite["expected_samples_per_model"], 1)
            self.assertEqual(suite["benchmarks"][0]["data_sha256"], digest)
            self.assertEqual(suite["benchmarks"][0]["yaml_path"],
                             str(root / "tasks/greek_gen_mmlu_pro.yaml"))
            self.assertEqual(suite["tasks_dir"], str(root / "tasks"))
            self.assertEqual(suite["benchmarks"][0]["metric_key"], "exact_match,final-answer")
            self.assertTrue(all(not Path(path).is_absolute() for path in suite["source_hashes"]))


if __name__ == "__main__":
    unittest.main()
