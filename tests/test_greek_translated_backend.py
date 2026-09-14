"""CPU-only regressions for response persistence and vLLM request identity."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from greek_translated.backend import ResponseStore, TranslatedVLLM


class ResponseStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.cache = self.path / "cache.sqlite"
        self.raw = self.path / "raw.jsonl"

    def tearDown(self):
        self.directory.cleanup()

    def record(self, key="one"):
        return {"request_hash": key, "raw_response": "<think>σκέψη\\boxed{Α}</think>\n"
                "Τελική απάντηση: \\boxed{B}<|im_end|>", "token_ids": [1, 2, 3]}

    def test_lossless_roundtrip_and_no_duplicate_on_resume(self):
        store = ResponseStore(self.cache, self.raw)
        record = self.record()
        store.put(record)
        self.assertEqual(store.get("one"), record)
        store.close()
        store = ResponseStore(self.cache, self.raw)
        self.assertEqual(store.get("one"), record)
        self.assertEqual([json.loads(line) for line in self.raw.read_text().splitlines()], [record])
        store.close()

    def test_crash_after_jsonl_append_before_sqlite_flag_is_exactly_once(self):
        store = ResponseStore(self.cache, self.raw)
        store.put(self.record())
        pending = self.record("two")
        with store.db:
            store.db.execute("INSERT INTO generations(request_hash,record) VALUES(?,?)",
                             ("two", json.dumps(pending, ensure_ascii=False)))
        # Emulate both a full uncommitted export and a truncated final record.
        with self.raw.open("ab") as handle:
            handle.write((json.dumps(pending, ensure_ascii=False) + "\n{partial").encode())
        store.close()
        store = ResponseStore(self.cache, self.raw)
        records = [json.loads(line) for line in self.raw.read_text().splitlines()]
        self.assertEqual(records, [self.record(), pending])
        store.close()

    def test_missing_export_rebuilt_from_authoritative_cache(self):
        store = ResponseStore(self.cache, self.raw)
        store.put(self.record())
        store.close()
        self.raw.unlink()
        store = ResponseStore(self.cache, self.raw)
        self.assertEqual(json.loads(self.raw.read_text()), self.record())
        store.close()

    def test_unknown_existing_export_is_not_overwritten(self):
        self.raw.write_text("user data\n")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            ResponseStore(self.cache, self.raw)
        self.assertEqual(self.raw.read_text(), "user data\n")


class BackendTests(unittest.TestCase):
    def bare_backend(self):
        backend = object.__new__(TranslatedVLLM)
        backend._identity = {"model": "example", "protocol_hash": "test-v1"}
        backend._max_length = 65536
        backend._max_gen_toks = 32768
        backend._saved_eos = [1, 250019]
        backend.tokenizer = SimpleNamespace(eos_token_id=1)
        backend._SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        backend._final_output_kind = "final-only"
        backend._TokensPrompt = lambda **kwargs: kwargs
        return backend

    def test_request_cache_and_seed_are_stable_and_isolated(self):
        backend = self.bare_backend()
        kwargs = {"max_gen_toks": 32, "until": ["<|end|>"], "temperature": 1.0}
        before = copy.deepcopy(kwargs)
        first = backend._sampling([1, 2, 3], kwargs, {"uid": "doc1"})
        same = backend._sampling([1, 2, 3], kwargs, {"uid": "doc1"})
        other = backend._sampling([1, 2, 3], kwargs, {"uid": "doc2"})
        self.assertEqual(first[0:2], same[0:2])
        self.assertNotEqual(first[0], other[0])
        self.assertEqual(kwargs, before)
        self.assertFalse(first[2].skip_special_tokens)
        self.assertTrue(first[2].include_stop_str_in_output)
        self.assertEqual(first[2].stop_token_ids, [1, 250019])
        self.assertEqual(first[2].output_kind, "final-only")
        backend._identity["protocol_hash"] = "test-v2"
        changed = backend._sampling([1, 2, 3], kwargs, {"uid": "doc1"})
        self.assertNotEqual(first[0], changed[0])

    def test_refuses_truncation_and_multiple_samples(self):
        backend = self.bare_backend()
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            backend._sampling([1] * 65000, {"max_gen_toks": 1000})
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            backend._sampling([1], {"max_gen_toks": 10, "n": 2})

    def test_greedy_and_string_boolean(self):
        backend = self.bare_backend()
        _, params, _ = backend._sampling([1], {"do_sample": "false", "temperature": 1.0})
        self.assertEqual(params["temperature"], 0.0)

    def test_internal_to_external_mapping_not_completion_order(self):
        backend = self.bare_backend()
        chunks = [{"name": "first", "prompt_tokens": [1], "sampling": None},
                  {"name": "second", "prompt_tokens": [2], "sampling": None}]
        states = {}
        steps = []

        def enqueue(prompts, sampling_params, use_tqdm):
            states.update({"random-A": SimpleNamespace(external_req_id="external-7"),
                           "random-B": SimpleNamespace(external_req_id="external-3")})
            steps.extend(["external-3", "external-7"])
            return ["random-A", "random-B"]

        def step():
            identifier = steps.pop(0)
            for key in list(states):
                if states[key].external_req_id == identifier:
                    del states[key]
            return [SimpleNamespace(request_id=identifier, finished=True,
                                    outputs=[SimpleNamespace(token_ids=[100, 200])])]

        engine = SimpleNamespace(output_processor=SimpleNamespace(request_states=states),
                                 has_unfinished_requests=lambda: bool(steps), step=step)
        backend.model = SimpleNamespace(llm_engine=engine, enqueue=enqueue)
        actual = list(backend._generate_stream(chunks))
        self.assertEqual([row[0]["name"] for row in actual], ["second", "first"])

    def test_wrong_external_id_fails_instead_of_cross_assigning_answers(self):
        backend = self.bare_backend()
        states = {}
        queued = [False]

        def enqueue(*args, **kwargs):
            queued[0] = True
            states["internal"] = SimpleNamespace(external_req_id="real")
            return ["internal"]

        engine = SimpleNamespace(output_processor=SimpleNamespace(request_states=states),
            has_unfinished_requests=lambda: queued[0],
            step=lambda: [SimpleNamespace(request_id="wrong", finished=True, outputs=[])])
        backend.model = SimpleNamespace(llm_engine=engine, enqueue=enqueue)
        with self.assertRaisesRegex(RuntimeError, "Unexpected completion"):
            list(backend._generate_stream([{"prompt_tokens": [1], "sampling": None}]))

    def test_full_response_telemetry_and_resumed_stochastic_sample(self):
        backend = self.bare_backend()
        backend.native_thinking = True
        backend.completion_batch_size = 128
        backend.tok_encode = lambda strings: [[index + 1] for index, _ in enumerate(strings)]
        texts = {
            100: "Σκέψη με ενδιάμεσο \\boxed{A}. </think>\nΤελική απάντηση: \\boxed{B}<|im_end|>",
            200: "Ατελής σκέψη: \\boxed{A}",
        }
        backend.tokenizer.decode = lambda ids, **kwargs: texts[ids[0]]
        requests = [SimpleNamespace(task_name="test", doc_id=index,
            doc={"uid": f"doc-{index}", "choices": ["a", "b"], "target": "B"},
            args=(f"question-{index}", {"max_gen_toks": 32, "do_sample": True}))
            for index in range(2)]

        def stream(chunk):
            for item in reversed(chunk):
                identifier = 100 if item["request"].doc_id == 0 else 200
                completion = SimpleNamespace(token_ids=[identifier], text=texts[identifier],
                    finish_reason="stop" if identifier == 100 else "length", stop_reason=None)
                yield item, SimpleNamespace(outputs=[completion])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            backend._store = ResponseStore(path / "cache.sqlite", path / "raw.jsonl")
            backend._generate_stream = stream
            responses = backend.generate_until(requests, disable_tqdm=True)
            self.assertEqual(responses, [texts[100], texts[200]])
            records = {row["doc_id"]: row for row in
                       (json.loads(line) for line in (path / "raw.jsonl").read_text().splitlines())}
            self.assertEqual(records[0]["uid"], "doc-0")
            self.assertEqual(records[0]["extracted_answer"], "B")
            self.assertEqual(records[0]["answer_format"], "boxed")
            self.assertEqual(records[0]["token_ids"], [100])
            self.assertFalse(records[1]["has_final_answer"])
            self.assertEqual(records[1]["finish_reason"], "length")
            self.assertNotIn("target", records[0])
            backend._generate_stream = lambda chunk: self.fail("Cached stochastic requests must not resample")
            self.assertEqual(backend.generate_until(requests, disable_tqdm=True), responses)
            self.assertEqual(len((path / "raw.jsonl").read_text().splitlines()), 2)
            backend._store.close()


class K2CompatibilityTests(unittest.TestCase):
    def test_grouped_norm_is_preserved_and_worker_audited(self):
        base = SimpleNamespace(replace_rms_norm_class=lambda norm, size: "replacement")
        utils = SimpleNamespace(replace_rms_norm_class=base.replace_rms_norm_class)
        fake = SimpleNamespace(base=base, utils=utils)
        path = Path(__file__).resolve().parents[1] / "greek_translated/vllm_compat.py"
        spec = importlib.util.spec_from_file_location("translated_compat_cpu_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"vllm.model_executor.models.transformers": fake}):
            spec.loader.exec_module(module)
        norm_class = type("K2HorizonRMSNorm", (), {})
        norms = [norm_class() for _ in range(73)]
        for norm in norms:
            norm.n_groups = 2
        self.assertIs(base.replace_rms_norm_class(norms[0], 2048), norms[0])
        self.assertEqual(base.replace_rms_norm_class(object(), 2048), "replacement")
        worker = module.K2WorkerExtension()
        worker.model_runner = SimpleNamespace(model=SimpleNamespace(
            config=SimpleNamespace(model_type="k2_horizon", num_hidden_layers=36, layernorm_num_groups=2),
            modules=lambda: norms))
        self.assertEqual(worker.greek_translated_k2_norm_audit(), {
            "preserved_k2_norm_layers": 73, "groups": 2, "original_forward": True})
        norms[0].n_groups = 1
        with self.assertRaisesRegex(RuntimeError, "K2 norm mismatch"):
            worker.greek_translated_k2_norm_audit()


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.backend = BackendTests().bare_backend()
        self.backend.evaluation_mode = "loglikelihood"
        self.backend.native_thinking = False
        self.backend.completion_batch_size = 128
        self.backend._likelihood_store = ResponseStore(self.path / "cache.sqlite", self.path / "raw_likelihoods.jsonl")

    def tearDown(self):
        self.backend._likelihood_store.close()
        self.directory.cleanup()

    @staticmethod
    def request(context_ids, continuation_ids, index=0):
        return {"identity": {"task_name": "truthfulqa_mc2", "doc_id": 0,
                             "uid": "truthfulqa_mc2/0", "option_index": index},
                "context": "original context", "continuation": f"option-{index}",
                "context_token_ids": context_ids, "continuation_token_ids": continuation_ids}

    @staticmethod
    def output(item, continuation_logprobs, greedy=True):
        positions = [None] + [{token: SimpleNamespace(logprob=-99.0, rank=1)}
                             for token in item["prompt_tokens"][1:]]
        for index, (token, value) in enumerate(zip(item["continuation_token_ids"], continuation_logprobs, strict=True),
                                             len(item["context_token_ids"])):
            positions[index] = {token: SimpleNamespace(logprob=value, rank=1 if greedy else 2)}
            if not greedy:
                positions[index][999] = SimpleNamespace(logprob=value + 0.1, rank=1)
        return SimpleNamespace(prompt_token_ids=item["prompt_tokens"], prompt_logprobs=positions,
                               outputs=[SimpleNamespace(token_ids=[888])])

    def records(self):
        path = self.path / "raw_likelihoods.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_sums_only_continuation_and_preserves_variable_lengths_and_order(self):
        requests = [self.request([10, 11, 12], [20, 21], 0), self.request([10], [30], 1)]

        def stream(chunk):
            for item in reversed(chunk):
                values = [-0.4, -0.8] if item["identity"]["option_index"] == 0 else [-2.0]
                yield item, self.output(item, values, greedy=len(values) == 2)

        self.backend._generate_stream = stream
        results = self.backend._loglikelihood_tokens(requests, disable_tqdm=True)
        self.assertAlmostEqual(results[0][0], -1.2)
        self.assertTrue(results[0][1])
        self.assertEqual(results[1], (-2.0, False))
        records = self.records()
        self.assertEqual([row["option_index"] for row in records], [1, 0])
        self.assertEqual(records[1]["continuation_token_logprobs"], [-0.4, -0.8])
        self.assertEqual(records[1]["continuation_token_ids"], [20, 21])
        self.assertEqual(records[1]["context_token_ids"], [10, 11, 12])
        self.assertEqual(records[1]["request_type"], "loglikelihood")
        self.assertEqual(records[1]["auxiliary_decode_tokens"], 1)
        self.assertFalse(records[1]["chat_template_applied"])
        self.assertNotIn("raw_response", records[1])
        self.assertNotIn("generated_tokens", records[1])
        self.assertFalse((self.path / "raw_generations.jsonl").exists())
        self.assertTrue(records[1]["scoring_params"]["skip_reading_prefix_cache"])
        self.backend._generate_stream = lambda chunk: self.fail("Repeated likelihood must reuse original score")
        self.assertEqual(self.backend._loglikelihood_tokens(requests, disable_tqdm=True), results)
        self.assertEqual(len(self.records()), 2)

    def test_harness_whitespace_boundary_tokenization_is_preserved(self):
        encodings = {"question": [10], "question ans": [10, 20], "question ": [10, 999]}
        self.backend.tok_encode = lambda text, **kwargs: encodings[text]
        request = SimpleNamespace(args=("question ", "ans"), doc={}, task_name="truthfulqa_mc2", doc_id=7, idx=3)

        def stream(chunk):
            self.assertEqual(chunk[0]["context_token_ids"], [10])
            self.assertEqual(chunk[0]["continuation_token_ids"], [20])
            yield chunk[0], self.output(chunk[0], [-0.25])

        self.backend._generate_stream = stream
        self.assertEqual(self.backend.loglikelihood([request], disable_tqdm=True), [(-0.25, True)])
        record = self.records()[0]
        self.assertEqual(record["context"], "question ")
        self.assertEqual(record["continuation"], "ans")
        self.assertEqual(record["uid"], "truthfulqa_mc2/7")
        self.assertEqual(record["option_index"], 3)

    def test_each_finished_option_is_saved_before_later_engine_failure(self):
        requests = [self.request([10], [20], 0), self.request([10], [30], 1)]

        def failing_stream(chunk):
            yield chunk[0], self.output(chunk[0], [-0.5])
            # Generator resumes only after the backend commits this first row.
            self.assertEqual(len(self.records()), 1)
            raise RuntimeError("simulated engine failure")

        self.backend._generate_stream = failing_stream
        with self.assertRaisesRegex(RuntimeError, "simulated engine failure"):
            self.backend._loglikelihood_tokens(requests, disable_tqdm=True)
        self.assertEqual(len(self.records()), 1)

        def resumed_stream(chunk):
            self.assertEqual(len(chunk), 1)
            self.assertEqual(chunk[0]["identity"]["option_index"], 1)
            yield chunk[0], self.output(chunk[0], [-1.5])

        self.backend._generate_stream = resumed_stream
        self.assertEqual(self.backend._loglikelihood_tokens(requests, disable_tqdm=True), [(-0.5, True), (-1.5, True)])
        self.assertEqual(len(self.records()), 2)

    def test_missing_or_misaligned_logprobs_fail_without_cache(self):
        item = self.backend._likelihood_item(self.request([10, 11], [20]))
        valid = self.output(item, [-0.2])
        bad = copy.deepcopy(valid)
        bad.prompt_logprobs.pop()
        with self.assertRaisesRegex(RuntimeError, "partial prompt"):
            self.backend._likelihood_record(item, bad)
        bad = copy.deepcopy(valid)
        bad.prompt_token_ids = [10, 11, 99]
        with self.assertRaisesRegex(RuntimeError, "differ from"):
            self.backend._likelihood_record(item, bad)
        bad = copy.deepcopy(valid)
        bad.prompt_logprobs[-1] = {99: SimpleNamespace(logprob=-1.0, rank=1)}
        with self.assertRaisesRegex(RuntimeError, "Missing continuation token"):
            self.backend._likelihood_record(item, bad)
        self.assertEqual(self.records(), [])

    def test_likelihood_refuses_truncation_and_wrong_native_mode(self):
        self.backend._max_length = 3
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            self.backend._likelihood_item(self.request([1, 2], [3]))
        with self.assertRaisesRegex(ValueError, "at least one token"):
            self.backend._likelihood_item(self.request([], [3]))
        self.backend.native_thinking = True
        with self.assertRaisesRegex(ValueError, "native_thinking=False"):
            self.backend._loglikelihood_tokens([], disable_tqdm=True)
        with self.assertRaisesRegex(ValueError, "likelihood-only"):
            self.backend.generate_until([], disable_tqdm=True)

    def test_empty_continuation_has_unit_probability_without_engine_call(self):
        self.backend._generate_stream = lambda chunk: self.fail("Empty continuation does not require inference")
        self.assertEqual(self.backend._loglikelihood_tokens([self.request([1], [])], disable_tqdm=True), [(0, True)])
        self.assertEqual(self.records()[0]["continuation_token_logprobs"], [])
        self.assertEqual(self.records()[0]["auxiliary_decode_tokens"], 0)

    def test_cache_separates_options_and_continuation_boundary(self):
        first = self.backend._likelihood_item(self.request([10], [20, 30], 0))
        other_option = self.backend._likelihood_item(self.request([10], [20, 30], 1))
        other_boundary = self.backend._likelihood_item(self.request([10, 20], [30], 0))
        self.assertNotEqual(first["hash"], other_option["hash"])
        self.assertNotEqual(first["hash"], other_boundary["hash"])
        self.assertEqual(first["hash"], self.backend._likelihood_item(self.request([10], [20, 30], 0))["hash"])

    def test_template_token_triple_interface(self):
        item = self.backend._likelihood_item((("context", "continuation"), [10], [20]))
        self.assertEqual(item["context"], "context")
        self.assertEqual(item["continuation"], "continuation")
        self.assertEqual(item["prompt_tokens"], [10, 20])

    def test_constructor_accepts_no_generative_path_in_likelihood_mode(self):
        tokenizer = SimpleNamespace(eos_token_id=1)
        engine_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5"), max_model_len=65536)
        fake_model = SimpleNamespace(get_tokenizer=lambda: tokenizer,
                                     llm_engine=SimpleNamespace(model_config=engine_config))
        model_arguments = []

        def llm(**kwargs):
            model_arguments.append(kwargs)
            return fake_model

        fake_vllm = SimpleNamespace(LLM=llm, SamplingParams=lambda **kwargs: kwargs, TokensPrompt=lambda **kwargs: kwargs)
        fake_sampling = SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY="final-only"))
        with patch.dict("sys.modules", {"vllm": fake_vllm, "vllm.sampling_params": fake_sampling}):
            instance = TranslatedVLLM(pretrained=str(self.path / "checkpoint"), raw_generation_path=None,
                raw_likelihood_path=str(self.path / "constructor_likelihoods.jsonl"),
                cache_path=str(self.path / "constructor.sqlite"), evaluation_mode="loglikelihood",
                native_thinking=False, enable_thinking=False, protocol_hash="test")
            self.assertIsNone(instance.raw_generation_path)
            self.assertFalse(hasattr(instance, "_store"))
            self.assertTrue(hasattr(instance, "_likelihood_store"))
            self.assertNotIn("raw_likelihood_path", model_arguments[0])
            self.assertNotIn("evaluation_mode", model_arguments[0])
            instance._likelihood_store.close()
            with self.assertRaisesRegex(ValueError, "requires raw_generation_path"):
                TranslatedVLLM(pretrained="unused", raw_generation_path=None, cache_path="unused",
                               evaluation_mode="generate_until", protocol_hash="test")
            with self.assertRaisesRegex(ValueError, "requires raw_likelihood_path"):
                TranslatedVLLM(pretrained="unused", raw_generation_path=None, cache_path="unused",
                               evaluation_mode="loglikelihood", protocol_hash="test")
        self.assertEqual(len(model_arguments), 1)
        self.assertFalse((self.path / "raw_generations.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
