"""Lossless, resumable vLLM inference for translated Greek evaluations.

This intentionally implements TemplateLM directly: the cloned harness predates
vLLM 0.21, whereas the engine API below is tested against that installed version.
No model package or checkpoint is changed. Reference answers never enter cache
keys, seeds, generation parameters, or prompts constructed by this backend.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bool(value):
    return value.lower() in {"1", "true", "yes"} if isinstance(value, str) else bool(value)


class ResponseStore:
    """SQLite is authoritative; JSONL is an exactly-once recoverable export.

    The committed byte offset makes recovery unambiguous if a process dies
    between fsync of a JSONL append and the corresponding SQLite transaction.
    Only the uncommitted export tail is removed; its records remain in SQLite.
    """

    def __init__(self, cache_path, raw_path):
        self.raw_path = Path(raw_path)
        cache_path = Path(cache_path)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS generations (request_hash TEXT PRIMARY KEY, "
                        "record TEXT NOT NULL, exported INTEGER NOT NULL DEFAULT 0)")
        self.db.execute("CREATE TABLE IF NOT EXISTS export_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        state = self.db.execute("SELECT value FROM export_state WHERE key='committed_bytes'").fetchone()
        if state is None:
            if self.raw_path.exists() and self.raw_path.stat().st_size:
                raise RuntimeError("Existing nonempty raw export has no cache journal; refusing to overwrite it")
            self.db.execute("INSERT INTO export_state VALUES('committed_bytes','0')")
        self.db.commit()
        self._recover_export()
        self.export_pending()

    def _offset(self):
        return int(self.db.execute("SELECT value FROM export_state WHERE key='committed_bytes'").fetchone()[0])

    def _recover_export(self):
        offset = self._offset()
        if not self.raw_path.exists():
            with self.db:
                self.db.execute("UPDATE generations SET exported=0")
                self.db.execute("UPDATE export_state SET value='0' WHERE key='committed_bytes'")
            return
        size = self.raw_path.stat().st_size
        if size < offset:
            raise RuntimeError("Raw export is shorter than its committed journal; preserve it and inspect the cache")
        if size > offset:
            with self.raw_path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())

    def get(self, request_hash):
        row = self.db.execute("SELECT record FROM generations WHERE request_hash=?", (request_hash,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, record):
        encoded = _json(record)
        with self.db:
            self.db.execute("INSERT INTO generations(request_hash,record) VALUES(?,?)",
                            (record["request_hash"], encoded))
        self.export_pending()

    def export_pending(self):
        # Bound memory during recovery of a long run or recreation of an export.
        while True:
            rows = self.db.execute("SELECT request_hash,record FROM generations WHERE exported=0 "
                                   "ORDER BY rowid LIMIT 128").fetchall()
            if not rows:
                return
            offset = self._offset()
            with self.raw_path.open("ab") as handle:
                if handle.tell() != offset:
                    raise RuntimeError("Concurrent modification of raw export; one writer per cache is required")
                for _, record in rows:
                    handle.write((record + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
                new_offset = handle.tell()
            with self.db:
                self.db.executemany("UPDATE generations SET exported=1 WHERE request_hash=?",
                                    ((key,) for key, _ in rows))
                self.db.execute("UPDATE export_state SET value=? WHERE key='committed_bytes'", (str(new_offset),))

    def close(self):
        self.db.close()


@register_model("greek_translated_vllm")
class TranslatedVLLM(TemplateLM):
    def __init__(self, pretrained, raw_generation_path, cache_path,
                 native_thinking=False, completion_batch_size=128, protocol_hash=None,
                 chat_template_args=None, enable_thinking=True, max_gen_toks=32768,
                 max_model_len=None, max_length=None, batch_size="auto",
                 evaluation_mode="generate_until", raw_likelihood_path=None, **kwargs):
        super().__init__()
        if int(kwargs.pop("data_parallel_size", 1)) != 1:
            raise ValueError("Use one vLLM engine per Slurm task")
        if kwargs.pop("think_end_token", None) is not None:
            raise ValueError("think_end_token would discard reasoning; leave it unset")
        if max_model_len is not None and max_length is not None:
            raise ValueError("Specify max_model_len or max_length, not both")
        kwargs.pop("device", None)
        kwargs.pop("max_batch_size", None)
        self.native_thinking = _bool(native_thinking)
        if evaluation_mode not in {"generate_until", "loglikelihood"}:
            raise ValueError("evaluation_mode must be generate_until or loglikelihood")
        if evaluation_mode == "loglikelihood" and self.native_thinking:
            raise ValueError("Original likelihood evaluation must not use native thinking/chat formatting")
        self.evaluation_mode = evaluation_mode
        self.completion_batch_size = int(completion_batch_size)
        if self.completion_batch_size < 1:
            raise ValueError("completion_batch_size must be positive")
        self._max_gen_toks = int(max_gen_toks)
        self.batch_size = batch_size
        self.chat_template_args = copy.deepcopy(chat_template_args or {})
        self.chat_template_args.setdefault("enable_thinking", _bool(enable_thinking))
        self.add_bos_token = _bool(kwargs.pop("add_bos_token", False))
        self.raw_generation_path = Path(raw_generation_path) if raw_generation_path is not None else None
        if evaluation_mode == "generate_until" and self.raw_generation_path is None:
            raise ValueError("Generative evaluation requires raw_generation_path")
        self.raw_likelihood_path = (Path(raw_likelihood_path) if raw_likelihood_path is not None
                                    else self.raw_generation_path.with_name("raw_likelihoods.jsonl")
                                    if self.raw_generation_path is not None else None)
        if evaluation_mode == "loglikelihood" and self.raw_likelihood_path is None:
            raise ValueError("Likelihood evaluation requires raw_likelihood_path")
        self.cache_path = Path(cache_path)
        model_path = Path(pretrained).resolve()
        source = Path(__file__).with_name("protocol.py")
        if protocol_hash is None:
            protocol_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        config_digests = {}
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
            path = model_path / name
            if path.exists():
                config_digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._identity = {
            "model": str(model_path), "configs": config_digests,
            "dtype": str(kwargs.get("dtype", "auto")),
            "model_impl": kwargs.get("model_impl", "auto"),
            "revision": kwargs.get("revision"), "tokenizer": kwargs.get("tokenizer"),
            "tokenizer_revision": kwargs.get("tokenizer_revision"),
            "native_thinking": self.native_thinking, "chat_template_args": self.chat_template_args,
            "seed": int(kwargs.get("seed", 1234)), "protocol_hash": protocol_hash,
            "backend_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cache_schema": 2,
            "evaluation_mode": evaluation_mode,
        }
        self.model_args = {"model": str(model_path), **copy.deepcopy(kwargs)}
        self.model_args["max_model_len"] = int(max_model_len or max_length or 65536)
        self.model_args.setdefault("seed", 1234)
        self.model_args.setdefault("generation_config", "vllm")
        # Lazy imports allow deterministic CPU tests without initializing CUDA.
        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self._final_output_kind = RequestOutputKind.FINAL_ONLY
        self.model = LLM(**self.model_args)
        self.tokenizer = self.model.get_tokenizer()
        self._config = self.model.llm_engine.model_config.hf_config
        self._max_length = int(self.model.llm_engine.model_config.max_model_len)
        if getattr(self._config, "model_type", "") == "k2_horizon":
            audit = self.model.collective_rpc("greek_translated_k2_norm_audit", timeout=60)
            if not audit or not all(row.get("original_forward") and
                                    row.get("preserved_k2_norm_layers") == 73 and row.get("groups") == 2
                                    for row in audit):
                raise RuntimeError(f"K2 3.7B normalization was not verified in all workers: {audit}")
            print(f"K2 grouped normalization audit: {audit}", flush=True)
            self._identity["k2_grouped_norm_preserved"] = audit
        self._saved_eos = []
        generation_path = model_path / "generation_config.json"
        if generation_path.exists():
            eos = json.loads(generation_path.read_text(encoding="utf-8")).get("eos_token_id", [])
            self._saved_eos = [eos] if isinstance(eos, int) else list(eos or [])
        if evaluation_mode == "loglikelihood":
            self._likelihood_store = ResponseStore(self.cache_path, self.raw_likelihood_path)
        else:
            self._store = ResponseStore(self.cache_path, self.raw_generation_path)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(self, string, add_special_tokens=False, **kwargs):
        if kwargs.get("truncation") or kwargs.get("left_truncate_len"):
            raise ValueError("Question truncation is prohibited in this evaluation")
        return self.tokenizer(string, add_special_tokens=add_special_tokens or self.add_bos_token,
                              truncation=False, return_attention_mask=False).input_ids

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        history = copy.deepcopy(chat_history)
        if getattr(self._config, "model_type", "") == "k2_horizon":
            for message in history:
                if message.get("role") == "assistant" and not any(field in message for field in
                        ("think", "reasoning_content", "reasoning", "think_fast", "think_faster")):
                    message["reasoning_content"] = ""
        return self.tokenizer.apply_chat_template(history, tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt, **self.chat_template_args)

    def loglikelihood(self, requests, disable_tqdm=False):
        """Preserve harness pair tokenization and original per-option identity.

        No chat template, boxed prompt, or reference label is consulted. The
        inherited _encode_pair moves trailing context whitespace onto the
        continuation before separately encoding the context and complete pair,
        exactly as the cloned harness does for other causal LM backends.
        """
        encoded = []
        for request in requests:
            context, continuation = request.args
            if not isinstance(context, str) or not isinstance(continuation, str):
                raise TypeError("Likelihood context and continuation must be strings")
            if context == "":
                if self.prefix_token_id is None:
                    raise ValueError("Empty likelihood context requires a prefix token")
                context_ids, continuation_ids = [self.prefix_token_id], self.tok_encode(continuation)
            else:
                context_ids, continuation_ids = self._encode_pair(context, continuation)
            doc = request.doc or {}
            identity = {"task_name": request.task_name, "doc_id": request.doc_id,
                        "uid": doc.get("uid", f"{request.task_name}/{request.doc_id}"),
                        "option_index": request.idx}
            encoded.append({"identity": identity, "context": context, "continuation": continuation,
                            "context_token_ids": context_ids, "continuation_token_ids": continuation_ids})
        return self._loglikelihood_tokens(encoded, disable_tqdm=disable_tqdm)

    def _likelihood_item(self, request):
        # Retain TemplateLM's conventional token-triple interface as well as the
        # enriched records from loglikelihood(), which preserve document IDs.
        if not isinstance(request, dict):
            pair, context_ids, continuation_ids = request
            context, continuation = pair if pair is not None else (None, None)
            request = {"identity": {}, "context": context, "continuation": continuation,
                       "context_token_ids": context_ids, "continuation_token_ids": continuation_ids}
        context_ids = list(request["context_token_ids"])
        continuation_ids = list(request["continuation_token_ids"])
        if not context_ids:
            raise ValueError("Likelihood context must contain at least one token")
        tokens = context_ids + continuation_ids
        # vLLM's generation entrypoint requires room for one auxiliary decode
        # token. It is never included in the option score or treated as an answer.
        if len(tokens) + bool(continuation_ids) > self.max_length:
            raise ValueError(f"Likelihood prompt has {len(tokens)} tokens; context {self.max_length} "
                             "cannot fit it plus one auxiliary token; refusing to truncate")
        params = {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1,
                  "detokenize": False, "ignore_eos": True, "n": 1,
                  "skip_reading_prefix_cache": True, "flat_logprobs": False}
        fingerprint = {"request_type": "loglikelihood", "model": self._identity,
                       "request": request["identity"], "context": request["context"],
                       "continuation": request["continuation"], "context_token_ids": context_ids,
                       "continuation_token_ids": continuation_ids, "scoring_params": params}
        key = hashlib.sha256(_json(fingerprint).encode()).hexdigest()
        return {**request, "context_token_ids": context_ids, "continuation_token_ids": continuation_ids,
                "hash": key, "prompt_tokens": tokens, "params": params,
                "sampling": self._SamplingParams(**params, output_kind=self._final_output_kind)}

    def _likelihood_record(self, item, output):
        continuation_ids = item["continuation_token_ids"]
        values, ranks, greedy = [], [], True
        if continuation_ids:
            if output is None or list(output.prompt_token_ids) != item["prompt_tokens"]:
                raise RuntimeError("Likelihood output prompt tokens differ from the submitted option")
            positions = output.prompt_logprobs
            if positions is None or len(positions) != len(item["prompt_tokens"]):
                raise RuntimeError("Missing or partial prompt log-probabilities; refusing an incomplete score")
            for position, token in enumerate(continuation_ids, len(item["context_token_ids"])):
                probabilities = positions[position]
                if not probabilities or token not in probabilities:
                    raise RuntimeError(f"Missing continuation token {token} at position {position}")
                numeric = {key: float(getattr(value, "logprob", value))
                           for key, value in probabilities.items()}
                value = numeric[token]
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite continuation log-probability at position {position}")
                values.append(value)
                ranks.append(getattr(probabilities[token], "rank", None))
                greedy = greedy and max(numeric, key=numeric.get) == token
        return {
            "request_hash": item["hash"], "request_type": "loglikelihood",
            "created_at": datetime.now(timezone.utc).isoformat(), **item["identity"],
            "model": self._identity["model"], "protocol_hash": self._identity["protocol_hash"],
            "context": item["context"], "continuation": item["continuation"],
            "context_token_ids": item["context_token_ids"], "prompt_token_ids": item["prompt_tokens"],
            "prompt_tokens": len(item["prompt_tokens"]), "continuation_token_ids": continuation_ids,
            "continuation_token_logprobs": values, "continuation_token_ranks": ranks,
            "total_loglikelihood": sum(values), "is_greedy": bool(greedy),
            "auxiliary_decode_tokens": len(output.outputs[0].token_ids) if output is not None else 0,
            "scoring_params": item["params"], "chat_template_applied": False,
        }

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        if getattr(self, "evaluation_mode", None) != "loglikelihood" or self.native_thinking:
            raise ValueError("Likelihood scoring requires evaluation_mode=loglikelihood and native_thinking=False")
        if not requests:
            return []
        responses, pending = [None] * len(requests), {}
        progress = tqdm(total=len(requests), disable=disable_tqdm or self.rank != 0,
                        desc="Translated Greek option likelihoods")
        try:
            for index, request in enumerate(requests):
                item = self._likelihood_item(request)
                cached = self._likelihood_store.get(item["hash"])
                if cached is not None:
                    responses[index] = (cached["total_loglikelihood"], cached["is_greedy"])
                    progress.update(1)
                elif item["hash"] in pending:
                    pending[item["hash"]]["indices"].append(index)
                elif not item["continuation_token_ids"]:
                    record = self._likelihood_record(item, None)
                    self._likelihood_store.put(record)
                    responses[index] = (record["total_loglikelihood"], record["is_greedy"])
                    progress.update(1)
                else:
                    pending[item["hash"]] = {**item, "indices": [index]}
            print(f"Likelihood options={len(requests)} ready={sum(x is not None for x in responses)} "
                  f"uncached={len(pending)}", flush=True)
            work = sorted(pending.values(), key=lambda item: -len(item["prompt_tokens"]))
            for start in range(0, len(work), self.completion_batch_size):
                for item, output in self._generate_stream(work[start:start+self.completion_batch_size]):
                    record = self._likelihood_record(item, output)
                    # Commit each option before asking the engine for the next;
                    # an interrupted batch never loses already-finished scores.
                    self._likelihood_store.put(record)
                    answer = (record["total_loglikelihood"], record["is_greedy"])
                    for index in item["indices"]:
                        responses[index] = answer
                    progress.update(len(item["indices"]))
        finally:
            progress.close()
        if any(response is None for response in responses):
            raise RuntimeError("Missing likelihood scores after inference")
        return responses

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError("Rolling perplexity is outside this benchmark protocol")

    def _sampling(self, prompt_token_ids, generation, request_identity=None):
        params = copy.deepcopy(generation)
        limit = int(params.pop("max_gen_toks", self.max_gen_toks))
        if limit < 1 or len(prompt_token_ids) + limit > self.max_length:
            raise ValueError(f"Prompt has {len(prompt_token_ids)} tokens and requests {limit} new tokens, "
                             f"exceeding context {self.max_length}; refusing to truncate the question")
        stops = params.pop("until", []) or []
        stops = [stops] if isinstance(stops, str) else stops
        if not _bool(params.pop("do_sample", True)):
            params["temperature"] = 0.0
        params.setdefault("temperature", 1.0)
        params.update(max_tokens=limit, stop=list(dict.fromkeys(stops)), skip_special_tokens=False,
                      spaces_between_special_tokens=False, include_stop_str_in_output=True)
        eos_ids = list(params.get("stop_token_ids") or []) + self._saved_eos
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        params["stop_token_ids"] = sorted(set(eos_ids))
        if int(params.get("n", 1)) != 1:
            raise ValueError("Exactly one completion per question is supported")
        fingerprint = {"model": self._identity, "request": request_identity,
                       "prompt_token_ids": prompt_token_ids, "generation": params}
        preliminary = hashlib.sha256(_json(fingerprint).encode()).hexdigest()
        params.setdefault("seed", int(preliminary[:8], 16))
        request_hash = hashlib.sha256(_json(fingerprint).encode()).hexdigest()
        # FINAL_ONLY avoids repeatedly copying long partial completions. Progress
        # still counts live engine token IDs without printing any prompt content.
        sampling = self._SamplingParams(**params, output_kind=self._final_output_kind)
        return request_hash, params, sampling

    def _generate_stream(self, chunk):
        engine = self.model.llm_engine
        if engine.has_unfinished_requests():
            raise RuntimeError("Cannot mix a new batch with unfinished engine requests")
        internal_ids = self.model.enqueue(
            [self._TokensPrompt(prompt_token_ids=item["prompt_tokens"]) for item in chunk],
            sampling_params=[item["sampling"] for item in chunk], use_tqdm=False)
        states = engine.output_processor.request_states
        # vLLM 0.21 returns internal enqueue IDs, but step returns external IDs.
        external_ids = [states[identifier].external_req_id for identifier in internal_ids]
        if len(set(external_ids)) != len(chunk):
            raise RuntimeError("Non-unique external request IDs in this batch")
        waiting = dict(zip(external_ids, chunk, strict=True))
        last_report = time.monotonic()
        finished_tokens = 0
        while waiting:
            if not engine.has_unfinished_requests():
                raise RuntimeError("Engine stopped before every submitted request completed")
            for output in engine.step():
                if output.finished:
                    if output.request_id not in waiting:
                        raise RuntimeError(f"Unexpected completion: {output.request_id}")
                    if len(output.outputs) != 1:
                        raise RuntimeError("Engine returned multiple completions for one question")
                    finished_tokens += len(output.outputs[0].token_ids)
                    yield waiting.pop(output.request_id), output
            if time.monotonic() - last_report >= 60:
                active_lengths = [len(state.detokenizer.output_token_ids)
                                  for state in states.values()
                                  if state.external_req_id in waiting and state.detokenizer is not None]
                print(f"Generation heartbeat: batch_completed={len(chunk)-len(waiting)}/{len(chunk)} "
                      f"pending={len(waiting)} active_generated_tokens={sum(active_lengths)} "
                      f"longest_active={max(active_lengths, default=0)} "
                      f"finished_generated_tokens={finished_tokens}", flush=True)
                last_report = time.monotonic()

    def generate_until(self, requests, disable_tqdm=False):
        if getattr(self, "evaluation_mode", "generate_until") != "generate_until":
            raise ValueError("Generative requests cannot be mixed into a likelihood-only evaluation")
        from .protocol import INVALID, answer_format, extract_final_answer, reasoning_is_closed
        if not requests:
            return []
        responses = [None] * len(requests)
        pending = {}
        tokenized = self.tok_encode([request.args[0] for request in requests])
        progress = tqdm(total=len(requests), disable=disable_tqdm or self.rank != 0,
                        desc="Translated Greek generation")
        cache_hits = 0
        try:
            for index, (request, tokens) in enumerate(zip(requests, tokenized, strict=True)):
                request_identity = {"task_name": request.task_name, "doc_id": request.doc_id,
                                    "uid": request.doc["uid"]}
                key, params, sampling = self._sampling(tokens, request.args[1], request_identity)
                cached = self._store.get(key)
                if cached is not None:
                    responses[index] = cached["raw_response"]
                    progress.update(1)
                    cache_hits += 1
                elif key in pending:
                    pending[key]["indices"].append(index)
                else:
                    pending[key] = {"indices": [index], "request": request, "hash": key,
                                    "prompt_tokens": tokens, "params": params, "sampling": sampling}
            print(f"Generation requests={len(requests)} cached={cache_hits} uncached={len(pending)}", flush=True)
            work = sorted(pending.values(), key=lambda item: -len(item["prompt_tokens"]))
            for start in range(0, len(work), self.completion_batch_size):
                for item, output in self._generate_stream(work[start:start+self.completion_batch_size]):
                    completion = output.outputs[0]
                    token_ids = list(completion.token_ids)
                    raw = self.tokenizer.decode(token_ids, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                    request = item["request"]
                    num_choices = len(request.doc["choices"])
                    answer = extract_final_answer(raw, num_choices,
                                                  require_thinking_close=self.native_thinking)
                    form = answer_format(raw, num_choices, require_thinking_close=self.native_thinking)
                    record = {
                        "request_hash": item["hash"], "created_at": datetime.now(timezone.utc).isoformat(),
                        "task_name": request.task_name, "doc_id": request.doc_id, "uid": request.doc["uid"],
                        "subject": request.doc.get("subject"), "shots": request.doc.get("shots"),
                        "model": self._identity["model"], "protocol_hash": self._identity["protocol_hash"],
                        "prompt": request.args[0], "prompt_tokens": len(item["prompt_tokens"]),
                        "raw_response": raw, "engine_text": completion.text,
                        "token_ids": token_ids, "generated_tokens": len(token_ids),
                        "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                        "native_thinking": self.native_thinking, "reasoning_closed": reasoning_is_closed(raw),
                        "has_final_answer": answer != INVALID, "extracted_answer": answer,
                        "answer_format": form, "generation": item["params"],
                    }
                    self._store.put(record)
                    for index in item["indices"]:
                        responses[index] = raw
                    progress.update(len(item["indices"]))
        finally:
            progress.close()
        if any(response is None for response in responses):
            raise RuntimeError("Missing responses after generation")
        return responses
