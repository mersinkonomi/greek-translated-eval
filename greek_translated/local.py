"""Portable, resumable local-checkpoint evaluation; no Slurm or model downloads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading

PROJECT = Path(__file__).resolve().parents[1]
PROFILES = {"k2_base", "qwen_instruct", "qwen_base", "generic_chat", "generic_base"}
ENGINE_DEFAULTS = dict(dtype="bfloat16", max_length=65536, max_gen_toks=32768,
    gpu_memory_utilization=0.80, min_free_gpu_fraction=0.82, max_num_seqs=16,
    max_num_batched_tokens=8192, mc2_max_num_batched_tokens=1024,
    enforce_eager=True, seed=1234, completion_batch_size=128)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    from greek_translated.report import write_atomic
    write_atomic(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def code_hashes():
    files = list((PROJECT / "lm_eval").rglob("*.py"))
    files += list((PROJECT / "greek_translated").glob("*.py"))
    files += list((PROJECT / "lm_eval/tasks/greektruthfulqa").glob("*.yaml"))
    return {str(path.relative_to(PROJECT)): digest(path) for path in sorted(files)}


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "vllm", "datasets", "tokenizers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def checkpoint_files(path):
    return sorted(p for p in Path(path).rglob("*") if p.is_file()
                  and not any(part.startswith(".") for part in p.relative_to(path).parts))


def checkpoint_identity(path):
    """Hash actual weights, not just model/config names, once at preparation."""
    path = Path(path)
    files = checkpoint_files(path)
    if not files:
        raise ValueError(f"Empty checkpoint/tokenizer directory: {path}")
    output = {}
    for file in files:
        stat = file.stat()
        sha = digest(file)
        after = file.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Checkpoint changed while hashing: {file}")
        output[str(file.relative_to(path))] = dict(sha256=sha, size=stat.st_size,
                                                  mtime_ns=stat.st_mtime_ns)
    return output


def check_checkpoint(path, recorded, strong=False):
    current = {str(p.relative_to(path)): p for p in checkpoint_files(path)}
    if set(current) != set(recorded):
        raise RuntimeError(f"Checkpoint file list changed: {path}; prepare a new run")
    for name, info in recorded.items():
        stat = current[name].stat()
        if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"Checkpoint metadata changed: {current[name]}; prepare a new run")
        if strong and digest(current[name]) != info["sha256"]:
            raise RuntimeError(f"Checkpoint contents changed: {current[name]}")


def load_config(path):
    import yaml
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or set(value) - {"models", "engine"}:
        raise ValueError("Config must contain models and optional engine only")
    engine = dict(ENGINE_DEFAULTS)
    override = value.get("engine", {})
    if not isinstance(override, dict) or set(override) - set(engine):
        raise ValueError("Unknown engine setting; see models.example.yaml")
    engine.update(override)
    for key in ("max_length", "max_gen_toks", "max_num_seqs", "max_num_batched_tokens",
                "mc2_max_num_batched_tokens", "completion_batch_size"):
        if isinstance(engine[key], bool) or not isinstance(engine[key], int) or engine[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if engine["max_length"] <= engine["max_gen_toks"]:
        raise ValueError("max_length must leave room for prompt plus max_gen_toks; no truncation is allowed")
    if not isinstance(engine["enforce_eager"], bool) or not isinstance(engine["seed"], int):
        raise ValueError("enforce_eager must be boolean and seed must be integer")
    if engine["dtype"] not in {"bfloat16", "float16", "float32", "auto"}:
        raise ValueError("Unsupported dtype")
    for key in ("gpu_memory_utilization", "min_free_gpu_fraction"):
        if (isinstance(engine[key], bool) or not isinstance(engine[key], (int, float))
                or not math.isfinite(engine[key])):
            raise ValueError(f"{key} must be a finite fraction")
    if not 0 < engine["gpu_memory_utilization"] < engine["min_free_gpu_fraction"] <= 1:
        raise ValueError("Require 0 < gpu_memory_utilization < min_free_gpu_fraction <= 1")
    models, seen = [], set()
    entries = value.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Provide at least one local model")
    allowed = {"name", "path", "profile", "tokenizer_path", "trust_remote_code", "revision",
               "chat_template_args", "require_native_thinking_close", "generation", "add_bos_token"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("Unknown model setting")
        model = dict(entry)
        name = model.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("Model names must be nonempty and unique")
        seen.add(name)
        profile = model.get("profile")
        if profile not in PROFILES:
            raise ValueError(f"Choose an explicit model profile: {sorted(PROFILES)}")
        if not isinstance(model.get("path"), str) or not model["path"].strip():
            raise ValueError("Every model requires an explicit local path")
        for key in ("path", "tokenizer_path"):
            local_path = Path(model.get(key, model.get("path", ""))).expanduser()
            local_path = (path.parent / local_path).resolve() if not local_path.is_absolute() else local_path.resolve()
            if not local_path.is_dir():
                raise FileNotFoundError(f"Local checkpoint/tokenizer directory required: {local_path}")
            model[key] = str(local_path)
        if not (Path(model["path"]) / "config.json").is_file():
            raise FileNotFoundError(f"config.json missing from {model['path']}")
        hf_config = json.loads((Path(model["path"]) / "config.json").read_text())
        trust = model.setdefault("trust_remote_code", False)
        if not isinstance(trust, bool):
            raise ValueError("trust_remote_code must be explicit boolean")
        is_k2 = hf_config.get("model_type") == "k2_horizon"
        if profile == "k2_base" and not is_k2:
            raise ValueError("k2_base requires a K2 Horizon checkpoint")
        if is_k2 and (profile != "k2_base" or not trust
                      or hf_config.get("num_hidden_layers") != 36
                      or hf_config.get("layernorm_num_groups") != 2):
            raise ValueError("K2 requires k2_base, explicit trust_remote_code: true, 36 layers and 2 norm groups")
        model["apply_chat_template"] = profile in {"qwen_instruct", "generic_chat"}
        thinking = model.setdefault("require_native_thinking_close", profile == "qwen_instruct")
        if not isinstance(thinking, bool) or thinking and not model["apply_chat_template"]:
            raise ValueError("Thinking-close requirement is boolean and only applies to chat profiles")
        chat = {"enable_thinking": profile == "qwen_instruct"} if model["apply_chat_template"] else {}
        if not isinstance(model.get("chat_template_args", {}), dict):
            raise ValueError("chat_template_args must be a mapping")
        chat.update(model.get("chat_template_args", {}))
        if not model["apply_chat_template"] and chat:
            raise ValueError("Base profiles cannot apply chat template arguments")
        model["chat_template_args"] = chat
        generation = dict(max_gen_toks=engine["max_gen_toks"], do_sample=True, temperature=1.0,
                          top_p=0.95, top_k=20 if profile.startswith("qwen_") else -1,
                          presence_penalty=1.5 if profile == "qwen_instruct" else 0.0,
                          repetition_penalty=1.0)
        extra = model.get("generation", {})
        if not isinstance(extra, dict) or set(extra) - (set(generation) - {"max_gen_toks"}):
            raise ValueError("Unknown generation override; max_gen_toks belongs in engine")
        generation.update(extra)
        if not isinstance(generation["do_sample"], bool):
            raise ValueError("do_sample must be boolean")
        for key in ("temperature", "top_p", "presence_penalty", "repetition_penalty"):
            if isinstance(generation[key], bool) or not isinstance(generation[key], (int, float)) or not math.isfinite(generation[key]):
                raise ValueError(f"Invalid generation setting: {key}")
        if (generation["temperature"] < 0 or not 0 < generation["top_p"] <= 1
                or generation["repetition_penalty"] <= 0
                or not isinstance(generation["top_k"], int)
                or generation["top_k"] not in {-1} and generation["top_k"] < 1):
            raise ValueError("Invalid sampling parameters")
        model["generation"] = generation
        bos = model.setdefault("add_bos_token", profile == "k2_base")
        if not isinstance(bos, bool):
            raise ValueError("add_bos_token must be boolean")
        if profile == "k2_base" and not bos:
            raise ValueError("K2 Base requires its native BOS token")
        if bos and profile not in {"k2_base", "generic_base"}:
            raise ValueError("Only k2_base and generic_base profiles support an added BOS token")
        model["model_impl"] = "transformers" if is_k2 else "auto"
        models.append(model)
    return dict(models=models, engine=engine)


def check_integrity(manifest, strong=False, model_path=None):
    if code_hashes() != manifest["code_sha256"]:
        raise RuntimeError("Evaluation source changed; prepare a new run")
    if digest(manifest["suite_path"]) != manifest["suite_sha256"]:
        raise RuntimeError("Prepared suite changed")
    if package_versions() != manifest["package_versions"]:
        raise RuntimeError("Inference package versions changed; prepare a new run")
    for path, sha in manifest["input_sha256"].items():
        if digest(path) != sha:
            raise RuntimeError(f"Prepared data/task changed: {path}")
    for path, identity in manifest["checkpoint_files"].items():
        if model_path is None or path in model_path:
            check_checkpoint(Path(path), identity, strong=strong)


def prepare(args):
    root, suite_path = args.run_dir.resolve(), args.suite.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Run directory must be new and empty: {root}")
    config = load_config(args.config)
    suite = json.loads(suite_path.read_text())
    benches = suite["benchmarks"]
    if (not benches or len({b["id"] for b in benches}) != len(benches)
            or len({b["task"] for b in benches}) != len(benches)):
        raise ValueError("Suite must contain nonempty, unique benchmark and task IDs")
    inputs = {}
    for bench in benches:
        data = Path(bench["data_path"])
        if not data.is_absolute():
            data = (suite_path.parent / data).resolve()
            bench["data_path"] = str(data)
        inputs[str(data)] = digest(data)
        if inputs[str(data)] != bench["data_sha256"]:
            raise RuntimeError(f"Frozen data checksum mismatch: {bench['id']}")
    tasks = suite_path.parent / "tasks"
    for path in sorted(tasks.glob("*")):
        if path.suffix in {".yaml", ".py"}:
            inputs[str(path)] = digest(path)
    for bench in benches:
        if bench["id"] == "truthfulqa":
            from greek_translated.prepare_data import truthfulqa_original_hashes
            if bench["task"] != "greektruthfulqa_mc2" or bench.get("original_task_hashes") != truthfulqa_original_hashes():
                raise ValueError("TruthfulQA must retain the original MC2 task and source checksums")
        elif not (tasks / f"{bench['task']}.yaml").is_file():
            raise ValueError(f"Missing prepared generative YAML for {bench['task']}")
    checkpoints = {}
    for model in config["models"]:
        for key in ("path", "tokenizer_path"):
            if model[key] not in checkpoints:
                print(f"Hashing local checkpoint files: {model[key]}", flush=True)
                checkpoints[model[key]] = checkpoint_identity(model[key])
    code = code_hashes()
    versions = package_versions()
    identity = fingerprint(dict(config=config, inputs=inputs, code=code,
                                checkpoints=checkpoints, packages=versions,
                                suite_sha256=digest(suite_path)))
    runs = []
    for bench in sorted(benches, key=lambda b: b["expected_samples"]):
        for number, model in enumerate(config["models"]):
            index = len(runs)
            mc2 = bench["id"] == "truthfulqa"
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model["name"]).strip(".-")[:70] or "model"
            output = root / bench["id"] / f"{number:02d}-{slug}"
            runs.append(dict(index=index, model=model["name"], model_path=model["path"],
                model_config=model, benchmark=bench["id"], label=bench["label"], task=bench["task"],
                tasks=[bench["task"]], shots=bench["shots"], expected_tasks=1,
                expected_samples=min(2, bench["expected_samples"]) if args.pilot else bench["expected_samples"],
                output_type="multiple_choice" if mc2 else "generate_until",
                scoring_mode="original_mc2" if mc2 else "generative_regex",
                metric="acc,none" if mc2 else "exact_match,final-answer",
                results_dir=str(output), cache_path=str(root / "response_cache" / f"{index}.sqlite"),
                raw_generation_path=None if mc2 else str(output / "raw_generations.jsonl"),
                raw_likelihood_path=str(output / "raw_likelihoods.jsonl") if mc2 else None,
                native_thinking=False if mc2 else model["require_native_thinking_close"],
                apply_chat_template=False if mc2 else model["apply_chat_template"],
                chat_template_args={} if mc2 else model["chat_template_args"],
                generation_kwargs=None if mc2 else model["generation"], status="prepared"))
    manifest = dict(schema_version=1, run_id=root.name, run_dir=str(root), created_at=now(),
        pilot=args.pilot, limit=2 if args.pilot else None, runs=runs, configuration=config,
        configuration_path=str(args.config.resolve()), configuration_sha256=digest(args.config),
        suite=suite, suite_path=str(suite_path), suite_sha256=digest(suite_path),
        task_include_path=str(tasks), input_sha256=inputs, code_sha256=code,
        checkpoint_files=checkpoints, identity_sha256=identity,
        package_versions=versions,
        inference_identity_sha256=fingerprint(dict(config=config, code=code,
                                                    checkpoints=checkpoints, packages=versions)),
        max_length=config["engine"]["max_length"], max_gen_toks=config["engine"]["max_gen_toks"],
        dtype=config["engine"]["dtype"], seed=config["engine"]["seed"], backend="vLLM",
        report_path=str(root / "report.html"), protocol=dict(
            model_profiles=config["models"], engine=config["engine"],
            scoring="Generative benchmarks use regex-scored final answers; when present, TruthfulQA retains original probability-based MC2. No overall average and no LLM judge.",
            prompting="Greek reasoning plus terminal Τελική απάντηση: \\boxed{A}. Pre-rendered benchmark-specific demonstrations. Base profiles use raw completion; chat profiles use their local tokenizer template. MC2 always uses its original raw QA context.",
            checkpoint_identity="SHA-256 of every non-hidden checkpoint/tokenizer file, including actual weights. Full hashes rechecked at run start; metadata rechecked before each worker. Do not modify checkpoints during evaluation.",
            raw_outputs="Lossless special-token decoding, token IDs and finish reasons for generation; per-option continuation log probabilities for original MC2. SQLite caches and JSONL exports remain local.",
            benchmark_notes={b["id"]: b.get("notes", "") for b in benches},
            overrides="All profile and engine overrides are recorded above; changing them requires a new run directory.",
            pilot="Pilot artifacts are separate and cannot count toward full results."))
    atomic_json(root / "manifest.json", manifest)
    refresh_report(root)
    print(root, flush=True)


def load_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["run_dir"] != str(root):
        raise ValueError("For execution, run directories must stay at their prepared absolute path")
    return manifest


def refresh_report(root, require_complete=False):
    from greek_translated.report import assemble, render, write_atomic
    root = Path(root).resolve()
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, runs, warnings = assemble(root, root / "manifest.json")
        output = root / "report.html"
        write_atomic(output, render(root, root / "manifest.json", output, manifest, runs, warnings))
        complete = sum(run["completed"] for run in runs)
        atomic_json(root / "report.summary.json", dict(updated_at=now(), pilot=manifest["pilot"],
            completed=complete, expected=len(runs), complete=complete == len(runs), warnings=warnings))
    print(f"Report: {output} ({complete}/{len(runs)} validated)", flush=True)
    if require_complete and complete != len(runs):
        raise RuntimeError(f"Report is incomplete: {complete}/{len(runs)}; inspect logs/status")


def check_pilot(manifest, pilot_root):
    from greek_translated.validation import validate
    pilot = load_manifest(pilot_root)
    if not pilot["pilot"] or pilot["identity_sha256"] != manifest["identity_sha256"]:
        raise ValueError("Pilot must match the exact config, checkpoints, code and prepared suite")
    if len(pilot["runs"]) != len(manifest["runs"]):
        raise ValueError("Pilot does not cover every planned model and task")
    for item in pilot["runs"]:
        status = Path(pilot_root) / "status" / f"{item['index']}.json"
        if not status.is_file() or json.loads(status.read_text()).get("status") != "completed":
            raise RuntimeError(f"Pilot is not complete: index {item['index']}")
        validate(item, pilot)


def devices_for_children(spec, environ=None):
    environ = os.environ if environ is None else environ
    requested = spec.split(",")
    if not requested or any(not re.fullmatch(r"\d+", item) for item in requested):
        raise ValueError("--devices must be comma-separated nonnegative logical GPU indices")
    if len({int(x) for x in requested}) != len(requested):
        raise ValueError("Duplicate GPU indices are prohibited")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(int(item)) for item in requested]
    allocated = [item.strip() for item in visible.split(",") if item.strip()]
    if not allocated or allocated == ["-1"] or any(int(x) >= len(allocated) for x in requested):
        raise ValueError("Requested GPU is outside the parent CUDA_VISIBLE_DEVICES allocation")
    mapped = [allocated[int(item)] for item in requested]
    if len(set(mapped)) != len(mapped) or "-1" in mapped:
        raise ValueError("Parent GPU allocation contains duplicate or disabled devices")
    return mapped


def run(args):
    root = args.run_dir.resolve()
    manifest = load_manifest(root)
    devices = devices_for_children(args.devices)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Verifying frozen code, data and full checkpoint hashes before execution…", flush=True)
        check_integrity(manifest, strong=True)
        if not manifest["pilot"]:
            if args.pilot_run_dir:
                check_pilot(manifest, args.pilot_run_dir.resolve())
            elif not args.allow_unpiloted:
                raise ValueError("Full runs require --pilot-run-dir with a successful matching pilot; explicit --allow-unpiloted bypasses this safety check")
        work, failures = queue.Queue(), []
        for item in manifest["runs"]:
            work.put(item["index"])
        active, mutex = {}, threading.Lock()
        stopping = threading.Event()
        def consume(device):
            while not stopping.is_set():
                try:
                    index = work.get_nowait()
                except queue.Empty:
                    return
                log = root / "logs" / f"{index}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                           PYTHONPATH=str(PROJECT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
                print(f"GPU {device}: index {index} → {log}", flush=True)
                with log.open("a") as output:
                    process = subprocess.Popen([sys.executable, "-m", "greek_translated.local", "worker",
                        "--run-dir", str(root), "--index", str(index), "--parent-verified"],
                        cwd=PROJECT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    with mutex:
                        active[process.pid] = process
                    returncode = process.wait()
                    if returncode:
                        status_path = root / "status" / f"{index}.json"
                        status = json.loads(status_path.read_text()) if status_path.is_file() else dict(manifest["runs"][index])
                        if status.get("status") != "completed":
                            status.update(status="failed", exit_code=returncode, updated_at=now(),
                                          log_path=str(log), error=status.get("error") or f"Worker exited with code {returncode}; see log")
                            atomic_json(status_path, status)
                    with mutex:
                        active.pop(process.pid, None)
                        if returncode:
                            failures.append(index)
                work.task_done()
        pool = ThreadPoolExecutor(max_workers=len(devices))
        futures = [pool.submit(consume, device) for device in devices]
        try:
            for future in futures:
                future.result()
        except BaseException:
            stopping.set()
            with mutex:
                for process in active.values():
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
            raise
        finally:
            pool.shutdown(wait=True)
            refresh_report(root)
        if failures:
            raise RuntimeError(f"Failed indices: {sorted(failures)}; inspect logs and rerun the same command to resume")
        refresh_report(root, require_complete=True)


def worker(args):
    root = args.run_dir.resolve()
    manifest = load_manifest(root)
    if not 0 <= args.index < len(manifest["runs"]):
        raise ValueError("Unknown run index")
    item = dict(manifest["runs"][args.index])
    model, engine = item["model_config"], manifest["configuration"]["engine"]
    status_path = root / "status" / f"{args.index}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with (root / "status" / f"{args.index}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _worker_locked(args, root, manifest, item, model, engine, status_path)


def _worker_locked(args, root, manifest, item, model, engine, status_path):
    from greek_translated.validation import validate
    check_integrity(manifest, strong=not args.parent_verified,
                    model_path={model["path"], model["tokenizer_path"]})
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        validate(item, manifest)
        print("Already completed and validated; no duplicate generation.", flush=True)
        return
    def update(state, **extra):
        item.update(status=state, updated_at=now(), **extra)
        atomic_json(status_path, item)
    try:
        update("loading", started_at=now(), hostname=socket.gethostname(),
               log_path=str(root / "logs" / f"{args.index}.log"))
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
        os.environ["GREEK_TRANSLATED_REQUIRE_THINKING_CLOSE"] = "1" if item["native_thinking"] else "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        jit = root / "compiler_cache" / socket.gethostname() / str(args.index)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(jit / "flashinfer")
        os.environ["TORCH_EXTENSIONS_DIR"] = str(jit / "torch")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each worker must expose exactly one allocated NVIDIA GPU")
        free, total = torch.cuda.mem_get_info()
        if free / total < engine["min_free_gpu_fraction"]:
            raise RuntimeError(f"Allocated GPU has {free / 2**30:.1f}/{total / 2**30:.1f} GiB free; "
                               f"requires {engine['min_free_gpu_fraction']:.1%}. Do not use another user's allocation.")
        versions = package_versions()
        update("running", gpu=torch.cuda.get_device_name(0), free_gpu_gib=free / 2**30,
               visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), package_versions=versions)
        from greek_translated.backend import TranslatedVLLM
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager
        from lm_eval.loggers import EvaluationTracker
        mc2 = item["output_type"] == "multiple_choice"
        model_args = dict(pretrained=model["path"], tokenizer=model["tokenizer_path"],
            dtype=engine["dtype"], trust_remote_code=model["trust_remote_code"],
            model_impl=model["model_impl"], add_bos_token=model["add_bos_token"],
            max_model_len=engine["max_length"], max_gen_toks=engine["max_gen_toks"],
            max_num_seqs=engine["max_num_seqs"],
            max_num_batched_tokens=engine["mc2_max_num_batched_tokens" if mc2 else "max_num_batched_tokens"],
            gpu_memory_utilization=engine["gpu_memory_utilization"], tensor_parallel_size=1,
            seed=engine["seed"], enforce_eager=engine["enforce_eager"], generation_config="vllm",
            raw_generation_path=item["raw_generation_path"], raw_likelihood_path=item["raw_likelihood_path"],
            cache_path=item["cache_path"], evaluation_mode="loglikelihood" if mc2 else "generate_until",
            native_thinking=item["native_thinking"], completion_batch_size=engine["completion_batch_size"],
            chat_template_args=item["chat_template_args"],
            enable_thinking=item["chat_template_args"].get("enable_thinking", False),
            protocol_hash=fingerprint(dict(inference_identity=manifest["inference_identity_sha256"],
                protocol=manifest["code_sha256"]["greek_translated/protocol.py"])))
        if model["profile"] == "k2_base":
            model_args["worker_extension_cls"] = "greek_translated.vllm_compat.K2WorkerExtension"
        elif model["profile"].startswith("qwen_"):
            model_args["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        print(json.dumps(dict(model=item["model"], task=item["task"], model_args=model_args), ensure_ascii=False), flush=True)
        lm = TranslatedVLLM(**model_args)
        if model["profile"] == "k2_base":
            if lm.tokenizer.bos_token_id != 0 or lm.tok_encode("δοκιμή")[0] != 0:
                raise RuntimeError("K2 Base native BOS token 0 was not preserved")
        if item["apply_chat_template"] and not getattr(lm.tokenizer, "chat_template", None):
            raise ValueError("Chat profile selected but local tokenizer has no chat template")
        tracker = EvaluationTracker(output_path=item["results_dir"])
        tasks = item["tasks"]
        if mc2:
            from greek_translated.prepare_data import load_truthfulqa_mc2_task
            bench = next(b for b in manifest["suite"]["benchmarks"] if b["id"] == "truthfulqa")
            tasks = [load_truthfulqa_mc2_task(bench["data_path"], cache_dir=root / "dataset_cache",
                                             expected_hashes=bench["original_task_hashes"])]
        results = simple_evaluate(model=lm, model_args=f"pretrained={model['path']}", tasks=tasks,
            task_manager=TaskManager(include_path=manifest["task_include_path"]),
            num_fewshot=0, batch_size="auto", device="cuda:0", use_cache=None,
            apply_chat_template=item["apply_chat_template"], fewshot_as_multiturn=False,
            gen_kwargs=item["generation_kwargs"], limit=manifest["limit"], log_samples=True,
            evaluation_tracker=tracker, random_seed=0, numpy_random_seed=1234,
            torch_random_seed=engine["seed"], fewshot_random_seed=1234,
            bootstrap_iters=1000 if manifest["pilot"] else 100000, verbosity="INFO")
        if results is None:
            raise RuntimeError("No evaluation result returned")
        tracker.general_config_tracker.model_source = "greek_translated_vllm"
        results["config"]["model_args"] = model_args
        results["config"]["prompt_fewshot"] = item["shots"]
        samples = results.pop("samples")
        tracker.save_results_aggregated(results=results, samples=samples)
        for task in results["configs"]:
            tracker.save_results_samples(task_name=task, samples=samples[task])
        path, count = validate(item, manifest)
        update("completed", result_path=str(path), samples=count, completed_at=now())
    except BaseException as error:
        update("failed", error=str(error), completed_at=now())
        raise
    finally:
        refresh_report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze config, dataset, code and actual checkpoint hashes")
    prep.add_argument("--config", required=True, type=Path)
    prep.add_argument("--suite", type=Path, default=Path("artifacts/suite/suite.json"))
    prep.add_argument("--run-dir", required=True, type=Path)
    prep.add_argument("--pilot", action="store_true", help="Two questions from every benchmark/model, in a separate run")
    prep.set_defaults(func=prepare)
    launch = commands.add_parser("run", help="One worker per allocated GPU, with resumable per-benchmark outputs")
    launch.add_argument("--run-dir", required=True, type=Path)
    launch.add_argument("--devices", default="0", help="Logical indices within parent CUDA_VISIBLE_DEVICES, e.g. 0,1")
    launch.add_argument("--pilot-run-dir", type=Path)
    launch.add_argument("--allow-unpiloted", action="store_true", help="Explicitly bypass the successful-pilot requirement")
    launch.set_defaults(func=run)
    child = commands.add_parser("worker", help="Run one record; CUDA_VISIBLE_DEVICES must expose exactly one GPU")
    child.add_argument("--run-dir", required=True, type=Path)
    child.add_argument("--index", required=True, type=int)
    child.add_argument("--parent-verified", action="store_true", help=argparse.SUPPRESS)
    child.set_defaults(func=worker)
    report = commands.add_parser("report", help="Refresh the English HTML report without running inference")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--require-complete", action="store_true")
    report.set_defaults(func=lambda a: refresh_report(a.run_dir, a.require_complete))
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
