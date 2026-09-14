"""Prepare reproducible Greek benchmark JSONL inputs and harness task YAMLs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from greek_translated.protocol import LABELS, build_prompt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT_DIR = Path("artifacts/suite")
PINS_PATH = HERE / "dataset_pins.json"
TRUTHFULQA_DIR = PROJECT / "lm_eval/tasks/greektruthfulqa"
SPECS = [
    ("mmlu", "MMLU Greek", "ilsp/mmlu_greek", None, "test", 5, 14042),
    ("mmlu_pro", "MMLU-Pro Greek", "ilsp/MMLU-Pro_greek", "default", "test", 0, 12032),
    ("global_mmlu", "Global-MMLU Greek", "CohereLabs/Global-MMLU", "el", "test", 5, 14042),
    ("arc_easy", "ARC Greek — Easy", "ilsp/arc_greek", "ARC-Easy", "test", 25, 2376),
    ("arc_challenge", "ARC Greek — Challenge", "ilsp/arc_greek", "ARC-Challenge", "test", 25, 1168),
    ("hellaswag", "HellaSwag Greek", "ilsp/hellaswag_greek", "default", "validation", 10, 10024),
    ("truthfulqa", "TruthfulQA Greek — original MC2", "ilsp/truthful_qa_greek", "multiple_choice", "train", 0, 817),
    ("belebele", "Belebele Greek", "facebook/belebele", "ell_Grek", "test", 5, 900),
]
NOTES = {
    "mmlu": "All 57 translated MMLU subjects; five development examples from the matching subject. Not the native dascim/GreekMMLU benchmark.",
    "mmlu_pro": "Zero-shot because the Greek release has only a test split. Gold cot_content and English originals are never included in prompts.",
    "global_mmlu": "Greek el only, full 57-subject test set; five matching-subject development examples. CohereLabs is the current namespace of CohereForAI.",
    "arc_easy": "Twenty-five training examples. Preserve all choices; numeric source labels map through choices.label.",
    "arc_challenge": "Twenty-five training examples. Preserve all choices; numeric source labels map through choices.label.",
    "hellaswag": "Labeled validation split, not unlabeled test. Ten training examples; upstream Greek HellaSwag context/ending preprocessing preserved.",
    "truthfulqa": "Unmodified upstream greektruthfulqa_mc2: original choice order and MC2 normalized probability mass over all true answers. Zero sampled few-shot examples, but the original task's six fixed QA priming examples remain. No generated answer, regex, shuffling, reasoning instruction, or native chat template. Only the dataset transport is replaced by a frozen local copy of the original source rows.",
    "belebele": "Only a test split exists. For each question, take the first five other test examples whose passage and source link both differ. All 900 questions remain evaluation items. This is leave-current-passage-out in-context evaluation, not a disjoint development-set protocol.",
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_pins():
    """Immutable source revisions and normalized input digests, with no host paths."""
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    expected_ids = {spec[0] for spec in SPECS}
    if pins.get("schema_version") != 1 or set(pins.get("benchmarks", {})) != expected_ids:
        raise ValueError("Dataset pin manifest does not describe the eight benchmark variants")
    if pins.get("expected_samples_per_model") != sum(spec[6] for spec in SPECS):
        raise ValueError("Dataset pin total disagrees with the benchmark counts")
    for benchmark, _, repo, config, split, shots, count in SPECS:
        pin = pins["benchmarks"][benchmark]
        expected = {"dataset": repo, "config": config, "split": split, "shots": shots,
                    "expected_samples": count}
        if any(pin.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Dataset pin metadata disagrees with protocol for {benchmark}")
        if not re.fullmatch(r"[a-f0-9]{40}", pin.get("revision", "")):
            raise ValueError(f"An immutable 40-character dataset revision is required for {benchmark}")
        if not re.fullmatch(r"[a-f0-9]{64}", pin.get("data_sha256", "")):
            raise ValueError(f"A normalized input SHA256 is required for {benchmark}")
    return pins


@contextmanager
def offline_mode(enabled):
    """Disable Hub requests, including metadata, even in an already-imported SDK.

    datasets' DownloadConfig(local_files_only=True) alone does not cover every
    Hub metadata call. Preparation is single-threaded; restore the caller's
    environment and library flags when leaving this scoped operation.
    """
    if not enabled:
        yield
        return
    from datasets import config as datasets_config
    from huggingface_hub import constants as hub_constants
    environment = {name: os.environ.get(name) for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")}
    flags = [(module, name, getattr(module, name)) for module, names in (
        (datasets_config, ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")),
        (hub_constants, ("HF_HUB_OFFLINE",)),
    ) for name in names if hasattr(module, name)]
    try:
        for name in environment:
            os.environ[name] = "1"
        for module, name, _ in flags:
            setattr(module, name, True)
        yield
    finally:
        for module, name, previous in flags:
            setattr(module, name, previous)
        for name, previous in environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _load(repo, config, private_cache, offline=False):
    # Use the supported datasets cache, never inspect a machine-specific Arrow
    # directory or resolve mutable main. Cache misses in offline mode must fail.
    from datasets import DownloadConfig, load_dataset
    revisions = {pin["revision"] for pin in dataset_pins()["benchmarks"].values()
                 if pin["dataset"] == repo}
    if len(revisions) != 1:
        raise ValueError(f"No unique pinned dataset revision for {repo}")
    revision = revisions.pop()
    kwargs = {"revision": revision, "cache_dir": str(private_cache)}
    if offline:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    with offline_mode(offline):
        dataset = load_dataset(repo, config, **kwargs)
    return dict(dataset), {"dataset": repo, "config": config, "revision": revision,
                          "source": "pinned public Hugging Face dataset"}


def mmlu_subjects():
    import yaml
    subjects = sorted(yaml.safe_load(path.read_text(encoding="utf-8"))["dataset_name"]
                      for path in (PROJECT / "lm_eval/tasks/greekmmlu").glob("greekmmlu_*.yaml"))
    if len(subjects) != 57 or len(set(subjects)) != 57:
        raise ValueError(f"Expected 57 unique upstream translated MMLU subjects, got {len(subjects)}")
    return subjects


def _mmlu_datasets(repo, subjects, private_cache, offline=False, download_workers=4):
    def load_subject(subject):
        return _load(repo, subject, private_cache, offline=offline)
    if offline or download_workers == 1:
        # offline_mode deliberately changes SDK globals inside a restored scope.
        for subject in subjects:
            yield subject, load_subject(subject)
    else:
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            # map yields in input order: parallel downloads never change example
            # ordering, demonstration selection, normalized bytes, or scoring.
            yield from zip(subjects, pool.map(load_subject, subjects))


def _cache_directory(output_dir, cache_dir=None):
    # Explicit CLI cache wins, then normal HF dataset-cache environment settings.
    # Without an override all generated artifacts remain under the output tree.
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    if os.environ.get("HF_DATASETS_CACHE"):
        return Path(os.environ["HF_DATASETS_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "datasets").resolve()
    return output_dir / "cache/datasets"


def _uid(repo, config, split, index):
    return f"{repo}/{config}/{split}/{index}"


def _base(repo, config, split, index, question, choices, target_index, subject="", passage=""):
    choices = [str(choice).strip() for choice in choices]
    if not 2 <= len(choices) <= len(LABELS) or not 0 <= int(target_index) < len(choices):
        raise ValueError(f"Invalid choices/target for {_uid(repo, config, split, index)}")
    return {"uid": _uid(repo, config, split, index), "question": str(question).strip(), "passage": str(passage).strip(),
            "choices": choices, "target": LABELS[int(target_index)], "subject": str(subject), "dataset": repo,
            "dataset_config": config, "split": split}


def _preprocess_hellaswag(text):
    # Exactly the preprocessing in lm_eval/tasks/greekhellaswag/utils.py.
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def canonicalize(benchmark, repo, config, split, rows):
    output = []
    for index, row in enumerate(rows):
        if benchmark == "mmlu":
            doc = _base(repo, config, split, index, row["question"], row["choices"], row["answer"], config)
        elif benchmark == "mmlu_pro":
            # The official MMLU-Pro harness scores the answer string. One Greek
            # source row (question_id 3983) has a contradictory answer_index.
            answer_index = LABELS.index(row["answer"].strip().upper())
            doc = _base(repo, config, split, index, row["question"], row["options"], answer_index, row["category"])
            doc["source_question_id"] = row["question_id"]
            doc["source_answer"] = row["answer"]
            doc["source_answer_index"] = int(row["answer_index"])
            doc["source_answer_index_mismatch"] = answer_index != int(row["answer_index"])
        elif benchmark == "global_mmlu":
            doc = _base(repo, config, split, index, row["question"], [row[f"option_{x}"] for x in "abcd"], LABELS.index(row["answer"].strip().upper()), row["subject"])
            doc["source_question_id"] = row["sample_id"]
        elif benchmark.startswith("arc_"):
            doc = _base(repo, config, split, index, row["question"], row["choices"]["text"], row["choices"]["label"].index(row["answerKey"]), config)
            doc["source_question_id"] = row["id"]
        elif benchmark == "hellaswag":
            context = _preprocess_hellaswag(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
            doc = _base(repo, config, split, index, "Ποια από τις παρακάτω επιλογές αποτελεί την πιο πιθανή συνέχεια του κειμένου;", [_preprocess_hellaswag(x) for x in row["endings"]], int(row["label"]), row["activity_label"], context)
            doc["source_question_id"] = row["ind"]
        elif benchmark == "truthfulqa":
            # Preserve every original value and list order, including both
            # target structures. uid is audit metadata, never prompt content.
            doc = dict(row)
            if "uid" in doc:
                raise ValueError("Source TruthfulQA unexpectedly already has uid")
            doc["uid"] = _uid(repo, config, split, index)
        elif benchmark == "belebele":
            doc = _base(repo, config, split, index, row["question"], [row[f"mc_answer{i}"] for i in range(1, 5)], int(row["correct_answer_num"]) - 1, "reading_comprehension", row["flores_passage"])
            doc["source_link"] = row["link"]
            doc["source_question_id"] = row["question_number"]
        else:
            raise ValueError(benchmark)
        output.append(doc)
    return output


def attach_prompts(benchmark, docs, demonstration_pool, shots):
    if benchmark == "truthfulqa":
        raise ValueError("TruthfulQA uses its unmodified upstream prompt, not generated prompts")
    for doc in docs:
        candidates = demonstration_pool
        if benchmark in {"mmlu", "global_mmlu"}:
            candidates = [x for x in demonstration_pool if x["subject"] == doc["subject"]]
        elif benchmark == "belebele":
            candidates = [x for x in demonstration_pool if x["uid"] != doc["uid"] and x["passage"] != doc["passage"] and x.get("source_link") != doc.get("source_link")]
        examples = candidates[:shots]
        if len(examples) != shots or doc["uid"] in {x["uid"] for x in examples}:
            raise ValueError(f"Invalid demonstration set for {doc['uid']}")
        doc["shots"] = shots
        doc["demonstration_ids"] = [x["uid"] for x in examples]
        doc["prompt"] = build_prompt(doc, examples)


def _write_yaml(benchmark, data_path, shots, output_dir=None):
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    task = "greek_gen_" + benchmark
    lines = [f"task: {task}", "dataset_path: json", "dataset_kwargs:", "  data_files:",
        f"    test: {json.dumps(str(data_path))}", "test_split: test", "num_fewshot: 0", "output_type: generate_until",
        "doc_to_text: !function protocol.doc_to_text", "doc_to_target: !function protocol.doc_to_target",
        "generation_kwargs:", '  until: ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|ifm|endoftext|>", "<|ifm|im_end|>", "</s>"]',
        "  do_sample: true", "  temperature: 1.0", "  top_p: 0.95", "  max_gen_toks: 32768", "filter_list:",
        "  - name: final-answer", "    filter:", "      - function: custom", "        filter_fn: !function protocol.extract_final_answers",
        "      - function: take_first", "metric_list:", "  - metric: exact_match", "    aggregation: mean", "    higher_is_better: true",
        "metadata:", "  version: 1.0", "  protocol: greek_translated_reasoning_v1", f"  actual_num_fewshot: {shots}", "  pre_rendered_demonstrations: true"]
    yaml_path = output_dir / "tasks" / f"{task}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    # This harness resolves !function against the YAML directory rather than
    # Python's import path. A stable import shim keeps task artifacts relocatable
    # without copying protocol logic or embedding a checkout's absolute path.
    (yaml_path.parent / "protocol.py").write_text(
        "from greek_translated.protocol import doc_to_text, doc_to_target, extract_final_answers\n",
        encoding="utf-8")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return task, yaml_path


def truthfulqa_original_hashes():
    return {(TRUTHFULQA_DIR / name).relative_to(PROJECT).as_posix(): _sha(TRUTHFULQA_DIR / name)
            for name in ("greektruthfulqa_mc1.yaml", "greektruthfulqa_mc2.yaml", "utils.py")}


def portable_task_hash(yaml_path, data_path=None):
    """Hash task content independent of the destination machine's dataset path.

    Only our generated, JSON-quoted data_files path is normalized. Original
    upstream TruthfulQA task files are hashed byte-for-byte without changes.
    """
    content = Path(yaml_path).read_text(encoding="utf-8")
    if data_path is not None:
        actual = json.dumps(str(data_path))
        portable = json.dumps("${SUITE_DIR}/data/" + Path(data_path).name)
        if content.count(actual) != 1:
            raise ValueError("Expected exactly one generated task dataset path to normalize")
        content = content.replace(actual, portable)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_truthfulqa_mc2_task(data_path=None, cache_dir=None, expected_hashes=None):
    """Instantiate the exact upstream MC2 task using frozen local source rows.

    Only dataset_path, dataset_name and dataset_kwargs change. The original
    task id, prompt, six fixed demonstrations, choice order, result function,
    metric, zero-shot setting and default 'none' filter are untouched.
    """
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.utils import load_yaml_config
    if expected_hashes is not None and truthfulqa_original_hashes() != expected_hashes:
        raise ValueError("Original TruthfulQA task source hashes changed")
    config = load_yaml_config(str(TRUTHFULQA_DIR / "greektruthfulqa_mc2.yaml"))
    config["dataset_path"] = "json"
    config["dataset_name"] = None
    config["dataset_kwargs"] = {
        "data_files": {"train": str(Path(data_path or DEFAULT_OUTPUT_DIR / "data/truthfulqa.jsonl").expanduser().resolve())},
        "cache_dir": str(Path(cache_dir or DEFAULT_OUTPUT_DIR / "cache/task_datasets").expanduser().resolve()),
    }
    return ConfigurableTask(config=config)


def prepare_all(output_dir=None, belebele_shots=5, cache_dir=None, offline=False, download_workers=4):
    if belebele_shots not in (0, 5):
        raise ValueError("Belebele supports either 0 or 5 shots")
    if not 1 <= download_workers <= 16:
        raise ValueError("download_workers must be between 1 and 16")
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    private_cache = _cache_directory(output_dir, cache_dir)
    private_cache.mkdir(parents=True, exist_ok=True)
    pins = dataset_pins()
    records = []
    for benchmark, label, repo, config, split, shots, expected in SPECS:
        if benchmark == "belebele":
            shots = belebele_shots
        print(f"Preparing {benchmark} ({repo})", flush=True)
        sources, docs, examples = [], [], []
        if benchmark == "mmlu":
            subjects = mmlu_subjects()
            for position, (subject, (dataset, source)) in enumerate(
                    _mmlu_datasets(repo, subjects, private_cache, offline, download_workers), 1):
                docs.extend(canonicalize(benchmark, repo, subject, split, dataset[split]))
                examples.extend(canonicalize(benchmark, repo, subject, "dev", dataset["dev"]))
                sources.append(source)
                print(f"  Subject {position}/57: {subject}", flush=True)
        else:
            dataset, source = _load(repo, config, private_cache, offline=offline)
            sources.append(source)
            docs = canonicalize(benchmark, repo, config, split, dataset[split])
            if benchmark == "global_mmlu":
                examples = canonicalize(benchmark, repo, config, "dev", dataset["dev"])
            elif benchmark.startswith("arc_") or benchmark == "hellaswag":
                examples = canonicalize(benchmark, repo, config, "train", dataset["train"].select(range(shots)))
            elif benchmark == "belebele":
                examples = docs
        if len(docs) != expected:
            raise ValueError(f"{benchmark}: expected {expected} evaluation examples, got {len(docs)}")
        if benchmark != "truthfulqa":
            attach_prompts(benchmark, docs, examples, shots)
        data_path = data_dir / f"{benchmark}.jsonl"
        temporary = data_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for doc in docs:
                stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
        expected_digest = pins["benchmarks"][benchmark]["data_sha256"]
        if not (benchmark == "belebele" and shots != 5) and _sha(temporary) != expected_digest:
            raise ValueError(f"{benchmark}: normalized input SHA256 differs from the pinned protocol; "
                             f"expected {expected_digest}, got {_sha(temporary)}. "
                             "Refusing to replace the final dataset; inspect the .jsonl.tmp artifact.")
        temporary.replace(data_path)
        if benchmark == "truthfulqa":
            task, yaml_path = "greektruthfulqa_mc2", TRUTHFULQA_DIR / "greektruthfulqa_mc2.yaml"
        else:
            task, yaml_path = _write_yaml(benchmark, data_path, shots, output_dir)
        revisions = sorted(set(x["revision"] for x in sources if x["revision"]))
        record = {"id": benchmark, "task": task, "task_name": task, "label": label, "shots": shots,
            "expected_samples": len(docs), "data_path": str(data_path), "data_sha256": _sha(data_path),
            "yaml_path": str(yaml_path), "dataset": repo, "dataset_config": config,
            "portable_task_sha256": portable_task_hash(yaml_path, None if benchmark == "truthfulqa" else data_path),
            "portable_task_hash_normalization": "Generated data_files path replaced with ${SUITE_DIR}/data/FILENAME; upstream TruthfulQA unmodified",
            "dataset_revision": revisions[0] if len(revisions) == 1 else revisions, "split": split,
            "subject_counts": dict(Counter(x.get("subject", benchmark) for x in docs)),
            "target_counts": {} if benchmark == "truthfulqa" else dict(Counter(x["target"] for x in docs)),
            "choice_counts": dict(Counter(len(x["mc2_targets"]["choices"] if benchmark == "truthfulqa" else x["choices"]) for x in docs)),
            "mode": "likelihood_mc2" if benchmark == "truthfulqa" else "generative",
            "output_type": "multiple_choice" if benchmark == "truthfulqa" else "generate_until",
            "metric": "acc" if benchmark == "truthfulqa" else "exact_match",
            "filter": "none" if benchmark == "truthfulqa" else "final-answer",
            "metric_key": "acc,none" if benchmark == "truthfulqa" else "exact_match,final-answer",
            "source_answer_index_mismatches": [x["uid"] for x in docs if x.get("source_answer_index_mismatch")],
            "notes": ("Zero-shot evaluation on all 900 Greek test questions. No test examples or gold answers are supplied as demonstrations." if benchmark == "belebele" and shots == 0 else NOTES[benchmark]), "sources": sources}
        records.append(record)
        if benchmark == "truthfulqa":
            record.update({"original_task_hashes": truthfulqa_original_hashes(),
                "fixed_priming_examples": 6, "binary_metric": False, "apply_chat_template": False,
                "option_order": "unchanged source mc2_targets.choices",
                "source_gold_policy": "Original MC2 labels; normalized probability mass across every label=1 choice",
                "true_answer_counts": dict(Counter(sum(x["mc2_targets"]["labels"]) for x in docs)),
                "task_loader": "greek_translated.prepare_data.load_truthfulqa_mc2_task"})
        if benchmark == "mmlu_pro":
            mismatches = record["source_answer_index_mismatches"]
            record["source_gold_policy"] = "Use source answer letter, matching the upstream generative MMLU-Pro task; retain answer_index for audit."
            record["notes"] += f" Source answer/answer_index disagreements: {len(mismatches)} ({', '.join(mismatches)}). The source answer letter is authoritative, matching the upstream harness."
        print(f"  {len(docs):,} questions, {shots}-shot, targets={record['target_counts']}", flush=True)
    suite = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": {
        "name": "greek_translated_reasoning_v1", "language": "el", "max_gen_toks": 32768,
        "scoring": "Seven benchmarks use generated final-letter exact match and regex only. TruthfulQA uses unchanged upstream MC2 likelihood probability-mass scoring, by explicit user request. No LLM judge.",
        "final_answer": "Τελική απάντηση: \\boxed{A}", "native_thinking_close_env": "GREEK_TRANSLATED_REQUIRE_THINKING_CLOSE",
        "truthfulqa": NOTES["truthfulqa"],
        "fewshot": "Seven generative benchmarks: pre-rendered Greek examples in one user prompt, without invented gold reasoning. TruthfulQA: original zero sampled shots plus six fixed QA priming examples."},
        "expected_samples_per_model": sum(x["expected_samples"] for x in records), "benchmarks": records,
        "source_hashes": {path.relative_to(PROJECT).as_posix(): _sha(path) for path in
                          (PINS_PATH, HERE / "prepare_data.py", HERE / "protocol.py")},
        "tasks_dir": str(output_dir / "tasks")}
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {suite_path}", flush=True)
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path,
                        help="Datasets cache (default: HF_DATASETS_CACHE, HF_HOME/datasets, or OUTPUT/cache/datasets)")
    parser.add_argument("--offline", action="store_true", help="Use cached pinned inputs only; fail on cache misses")
    parser.add_argument("--download-workers", type=int, default=4,
                        help="Bounded concurrent MMLU config downloads (1–16, default 4; offline uses 1)")
    parser.add_argument("--belebele-shots", type=int, choices=(0, 5), default=5)
    args = parser.parse_args()
    prepare_all(args.output_dir, args.belebele_shots, args.cache_dir, args.offline, args.download_workers)
