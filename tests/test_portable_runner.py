"""CPU-only checks for portable provenance, profiles, scheduling and validation."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from greek_translated import local
from greek_translated.validation import validate


class PortableRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"llama"}')
        (self.model / "tokenizer_config.json").write_text('{"bos_token_id":0}')
        (self.model / "model.safetensors").write_bytes(b"weight-fixture-not-a-real-model")
        self.config = self.root / "models.yaml"
        self.write_config()

    def write_config(self, profile="generic_base", **extra):
        value = {"models": [dict(name="My local model", path=str(self.model), profile=profile, **extra)]}
        self.config.write_text(json.dumps(value))
        return self.config

    def suite(self):
        directory = self.root / "suite"
        (directory / "tasks").mkdir(parents=True)
        docs = [dict(uid=f"q{i}", target="A", choices=["ένα", "δύο"], prompt="Ερώτηση", question="Ποιο;") for i in range(3)]
        data = directory / "data.jsonl"
        data.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
        task = "greek_gen_fixture"
        (directory / "tasks" / f"{task}.yaml").write_text(f"task: {task}\n")
        suite = dict(benchmarks=[dict(id="native_greekmmlu_0shot", task=task,
            data_path=str(data), data_sha256=local.digest(data), expected_samples=3,
            label="Native GreekMMLU 0-shot", shots=0, notes="CPU fixture")])
        path = directory / "suite.json"
        path.write_text(json.dumps(suite))
        return path

    def prepare(self, suite, name="run", pilot=False):
        args = argparse.Namespace(config=self.config, suite=suite, run_dir=self.root / name, pilot=pilot)
        local.prepare(args)
        return local.load_manifest(args.run_dir)

    def artifacts(self, manifest, answer="A"):
        item = manifest["runs"][0]
        output = Path(item["results_dir"])
        output.mkdir(parents=True, exist_ok=True)
        n = item["expected_samples"]
        response = "Τελική απάντηση: \\boxed{" + answer + "}"
        raw = []
        samples = []
        for i in range(n):
            doc = dict(uid=f"q{i}", target="A", choices=["ένα", "δύο"], prompt="Ερώτηση", question="Ποιο;")
            raw.append(dict(task_name=item["task"], uid=doc["uid"], request_hash=f"hash{i}",
                raw_response=response, token_ids=[1], generated_tokens=1, finish_reason="stop"))
            samples.append(dict(doc=doc, resps=[[response]], filtered_resps=[answer],
                                filter="final-answer", exact_match=int(answer == "A")))
        Path(item["raw_generation_path"]).write_text("".join(json.dumps(row) + "\n" for row in raw))
        stamp = "2026-01-01T00-00-00"
        aggregate = dict(configs={item["task"]: {}},
            results={item["task"]: {item["metric"]: float(answer == "A")}},
            config={"limit": 2 if manifest["pilot"] else None},
            **{"n-samples": {item["task"]: {"effective": n, "original": 3}}})
        path = output / f"results_{stamp}.json"
        path.write_text(json.dumps(aggregate))
        sample_path = output / f"samples_{item['task']}_{stamp}.jsonl"
        sample_path.write_text("".join(json.dumps(row) + "\n" for row in samples))
        local.atomic_json(Path(manifest["run_dir"]) / "status/0.json", dict(status="completed", result_path=str(path)))
        return path, sample_path

    def test_local_paths_and_explicit_profiles(self):
        config = local.load_config(self.config)
        model = config["models"][0]
        self.assertFalse(model["apply_chat_template"])
        self.assertFalse(model["trust_remote_code"])
        self.assertEqual(model["tokenizer_path"], str(self.model))
        self.write_config(profile="guess")
        with self.assertRaisesRegex(ValueError, "explicit model profile"):
            local.load_config(self.config)
        self.config.write_text(json.dumps({"models": [{"name": "remote", "path": "owner/model", "profile": "generic_base"}]}))
        with self.assertRaises(FileNotFoundError):
            local.load_config(self.config)

    def test_profile_presets(self):
        for profile, chat, thinking, top_k, presence in [
                ("qwen_instruct", True, True, 20, 1.5), ("qwen_base", False, False, 20, 0),
                ("generic_chat", True, False, -1, 0), ("generic_base", False, False, -1, 0)]:
            with self.subTest(profile=profile):
                self.write_config(profile)
                model = local.load_config(self.config)["models"][0]
                self.assertEqual(model["apply_chat_template"], chat)
                self.assertEqual(model["require_native_thinking_close"], thinking)
                self.assertEqual(model["generation"]["top_k"], top_k)
                self.assertEqual(model["generation"]["presence_penalty"], presence)
                self.assertFalse(model["add_bos_token"])

    def test_k2_base_requires_consent_and_native_bos(self):
        (self.model / "config.json").write_text(json.dumps(dict(model_type="k2_horizon", num_hidden_layers=36, layernorm_num_groups=2)))
        self.write_config("k2_base")
        with self.assertRaisesRegex(ValueError, "explicit trust_remote_code"):
            local.load_config(self.config)
        self.write_config("k2_base", trust_remote_code=True, revision="mid_4")
        model = local.load_config(self.config)["models"][0]
        self.assertTrue(model["add_bos_token"])
        self.assertEqual(model["model_impl"], "transformers")
        self.assertFalse(model["apply_chat_template"])
        self.assertEqual(model["generation"]["top_k"], -1)

    def test_wrong_flags_and_unknown_settings_rejected(self):
        self.write_config(trust_remote_code="false")
        with self.assertRaisesRegex(ValueError, "boolean"):
            local.load_config(self.config)
        self.write_config(chat_template_args={"enable_thinking": True})
        with self.assertRaisesRegex(ValueError, "Base profiles"):
            local.load_config(self.config)
        self.write_config(generation={"max_gen_toks": 4})
        with self.assertRaisesRegex(ValueError, "belongs in engine"):
            local.load_config(self.config)

    def test_generic_base_can_preserve_native_bos(self):
        self.write_config(add_bos_token=True)
        self.assertTrue(local.load_config(self.config)["models"][0]["add_bos_token"])
        self.write_config("qwen_base", add_bos_token=True)
        with self.assertRaisesRegex(ValueError, "Only k2_base and generic_base"):
            local.load_config(self.config)

    def test_gpu_indices_cannot_escape_parent_allocation(self):
        self.assertEqual(local.devices_for_children("0,1", {"CUDA_VISIBLE_DEVICES": "3,7"}), ["3", "7"])
        self.assertEqual(local.devices_for_children("1", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}), ["GPU-b"])
        self.assertEqual(local.devices_for_children("2", {}), ["2"])
        for mask, request in [("3,7", "2"), ("", "0"), ("-1", "0"), ("3,7", "1,1")]:
            with self.subTest(mask=mask, request=request), self.assertRaises(ValueError):
                local.devices_for_children(request, {"CUDA_VISIBLE_DEVICES": mask})

    def test_actual_weight_hash_detects_content_change(self):
        identity = local.checkpoint_identity(self.model)
        path = self.model / "model.safetensors"
        self.assertEqual(identity[path.name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        local.check_checkpoint(self.model, identity, strong=True)
        path.write_bytes(b"different weights")
        with self.assertRaisesRegex(RuntimeError, "metadata changed"):
            local.check_checkpoint(self.model, identity, strong=True)

    def test_prepare_dynamic_suite_and_no_answer_seed_leakage(self):
        suite = self.suite()
        first = self.prepare(suite, "first")
        self.assertEqual(len(first["runs"]), 1)
        self.assertEqual(first["runs"][0]["expected_samples"], 3)
        with self.assertRaises(FileExistsError):
            self.prepare(suite, "first")
        data_path = Path(first["suite"]["benchmarks"][0]["data_path"])
        data_path.write_text(data_path.read_text().replace('"target": "A"', '"target": "B"'))
        content = json.loads(suite.read_text())
        content["benchmarks"][0]["data_sha256"] = local.digest(data_path)
        suite.write_text(json.dumps(content))
        second = self.prepare(suite, "second")
        self.assertNotEqual(first["identity_sha256"], second["identity_sha256"])
        self.assertEqual(first["inference_identity_sha256"], second["inference_identity_sha256"])

    def test_validation_and_complete_report(self):
        manifest = self.prepare(self.suite())
        path, _ = self.artifacts(manifest)
        self.assertEqual(validate(manifest["runs"][0], manifest), (path, 3))
        local.refresh_report(Path(manifest["run_dir"]), require_complete=True)
        summary = json.loads((Path(manifest["run_dir"]) / "report.summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["expected"], 1)

    def test_validation_refuses_corrupted_raw_or_score(self):
        manifest = self.prepare(self.suite())
        path, _ = self.artifacts(manifest)
        result = json.loads(path.read_text())
        result["results"][manifest["runs"][0]["task"]]["exact_match,final-answer"] = 0
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, "Aggregate score"):
            validate(manifest["runs"][0], manifest)
        self.artifacts(manifest)
        raw = Path(manifest["runs"][0]["raw_generation_path"])
        raw.write_text(raw.read_text().replace("boxed{A}", "boxed{B}"))
        with self.assertRaisesRegex(RuntimeError, "lossless raw export"):
            validate(manifest["runs"][0], manifest)

    def test_pilot_separate_and_required_complete(self):
        suite = self.suite()
        pilot = self.prepare(suite, "pilot", pilot=True)
        full = self.prepare(suite, "full")
        self.assertEqual(pilot["identity_sha256"], full["identity_sha256"])
        self.assertEqual(pilot["runs"][0]["expected_samples"], 2)
        self.assertNotEqual(pilot["runs"][0]["cache_path"], full["runs"][0]["cache_path"])
        with self.assertRaisesRegex(RuntimeError, "Pilot is not complete"):
            local.check_pilot(full, self.root / "pilot")
        self.artifacts(pilot)
        local.check_pilot(full, self.root / "pilot")
        with self.assertRaisesRegex(ValueError, "Pilot must match"):
            local.check_pilot(full, self.root / "full")
        with self.assertRaisesRegex(RuntimeError, "Report is incomplete"):
            local.refresh_report(self.root / "full", require_complete=True)

    def test_original_mc2_probabilities_and_option_completeness(self):
        from greek_translated.prepare_data import truthfulqa_original_hashes
        suite_path = self.suite()
        suite = json.loads(suite_path.read_text())
        bench = suite["benchmarks"][0]
        bench.update(id="truthfulqa", task="greektruthfulqa_mc2", expected_samples=1,
                     label="Original TruthfulQA MC2", original_task_hashes=truthfulqa_original_hashes())
        doc = dict(uid="truth-q", question="Ποιο;", mc2_targets=dict(choices=["ένα", "δύο"], labels=[1, 0]))
        Path(bench["data_path"]).write_text(json.dumps(doc) + "\n")
        bench["data_sha256"] = local.digest(bench["data_path"])
        suite_path.write_text(json.dumps(suite))
        manifest = self.prepare(suite_path)
        item = manifest["runs"][0]
        self.assertFalse(item["apply_chat_template"])
        self.assertFalse(item["native_thinking"])
        self.assertIsNone(item["raw_generation_path"])
        self.assertIsNone(item["generation_kwargs"])
        output = Path(item["results_dir"])
        output.mkdir(parents=True)
        raws = [dict(task_name=item["task"], uid=doc["uid"], option_index=i,
                     request_hash=f"likelihood-{i}", total_loglikelihood=-1.0,
                     continuation_token_logprobs=[-1.0], continuation_token_ids=[i]) for i in range(2)]
        Path(item["raw_likelihood_path"]).write_text("".join(json.dumps(row) + "\n" for row in raws))
        responses = [["-1.0", "True"], ["-1.0", "False"]]
        sample = dict(doc=doc, resps=[[row] for row in responses], filtered_resps=responses, filter="none", acc=0.5)
        stamp = "2026-01-01T00-00-00"
        sample_path = output / f"samples_{item['task']}_{stamp}.jsonl"
        sample_path.write_text(json.dumps(sample) + "\n")
        aggregate = dict(configs={item["task"]: {}}, results={item["task"]: {"acc,none": 0.5}},
                         config={"limit": None}, **{"n-samples": {item["task"]: {"effective": 1, "original": 1}}})
        result_path = output / f"results_{stamp}.json"
        result_path.write_text(json.dumps(aggregate))
        self.assertEqual(validate(item, manifest), (result_path, 1))
        raw_path = Path(item["raw_likelihood_path"])
        original_raw = raw_path.read_text()
        raw_path.write_text(json.dumps(raws[0]) + "\n")
        with self.assertRaisesRegex(RuntimeError, "every planned question and option"):
            validate(item, manifest)
        raw_path.write_text(original_raw)
        nonfinite = dict(raws[0], total_loglikelihood=float("nan"))
        raw_path.write_text(json.dumps(nonfinite) + "\n" + json.dumps(raws[1]) + "\n")
        with self.assertRaisesRegex(RuntimeError, "Invalid or incomplete raw option"):
            validate(item, manifest)
        raw_path.write_text(original_raw)
        sample["doc"]["mc2_targets"]["choices"] = ["δύο", "ένα"]
        sample_path.write_text(json.dumps(sample) + "\n")
        with self.assertRaisesRegex(RuntimeError, "targets or original filter"):
            validate(item, manifest)
        sample["doc"]["mc2_targets"]["choices"] = ["ένα", "δύο"]
        sample["doc"]["mc2_targets"]["labels"] = [0, 1]
        sample_path.write_text(json.dumps(sample) + "\n")
        with self.assertRaisesRegex(RuntimeError, "targets or original filter"):
            validate(item, manifest)


if __name__ == "__main__":
    unittest.main()
