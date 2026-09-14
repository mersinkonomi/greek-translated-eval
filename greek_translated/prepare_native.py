"""Prepare native dascim/GreekMMLU, separately from translated benchmarks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from greek_translated.prepare_data import (
    HERE, PROJECT, _base, _cache_directory, _sha, _write_yaml, attach_prompts, offline_mode,
)

PINS_PATH = HERE / "native_pins.json"
DEFAULT_OUTPUT_DIR = Path("artifacts/native-suite")
DATASET = "dascim/GreekMMLU"
PROTOCOL = "native_greekmmlu_unified_reasoning_v1"


def native_pins():
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    if (pins.get("schema_version") != 1 or pins.get("dataset") != DATASET
            or len(pins.get("configs", {})) != 45
            or pins.get("expected_samples") != 16632
            or sum(x["test"] for x in pins["configs"].values()) != 16632
            or any(x["dev"] != 5 for x in pins["configs"].values())
            or not re.fullmatch(r"[a-f0-9]{40}", pins.get("revision", ""))):
        raise ValueError("Native pins must describe all 45 configs and 16,632 test rows")
    if set(pins.get("data_sha256", {})) != {"0", "5"} or any(
            not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in pins["data_sha256"].values()):
        raise ValueError("Native pins require normalized input SHA256 for both shot settings")
    return pins


def _load(config, cache_dir, offline=False):
    from datasets import DownloadConfig, load_dataset
    revision = native_pins()["revision"]
    kwargs = {"revision": revision, "cache_dir": str(cache_dir)}
    if offline:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    with offline_mode(offline):
        dataset = load_dataset(DATASET, config, **kwargs)
    return dict(dataset), {"dataset": DATASET, "config": config,
                          "revision": revision, "source": "pinned public Hugging Face dataset"}


def canonicalize(config, split, rows):
    """Preserve choice order; native source answers are zero-based integers."""
    if split not in {"test", "dev"}:
        raise ValueError(f"Unexpected native split: {split}")
    docs = []
    for index, row in enumerate(rows):
        answer = row["answer"]
        if not 2 <= len(row["choices"]) <= 4:
            raise ValueError(f"{config}/{split}/{index}: expected two to four choices")
        if (isinstance(answer, bool) or not isinstance(answer, int)
                or not 0 <= answer < len(row["choices"])):
            raise ValueError(f"{config}/{split}/{index}: answer must index an existing choice")
        doc = _base(DATASET, config, split, index, row["question"],
                    row["choices"], answer, config)
        doc.update(source_answer_index=answer, source_subject=str(row.get("subject", "")),
                   group=str(row.get("group", "")), level=str(row.get("level", "")))
        docs.append(doc)
    return docs


def _question_key(doc):
    return (doc["question"], tuple(doc["choices"]))


def attach_native_prompts(docs, dev_docs, shots):
    if shots not in (0, 5):
        raise ValueError("Native GreekMMLU supports exactly 0-shot and 5-shot")
    if any(doc["split"] != "test" for doc in docs):
        raise ValueError("Evaluation items must come from the test split")
    if any(doc["split"] != "dev" for doc in dev_docs):
        raise ValueError("Demonstrations must come from the development split")
    for subject in {doc["subject"] for doc in docs}:
        examples = [doc for doc in dev_docs if doc["subject"] == subject]
        if len(examples) != 5:
            raise ValueError(f"{subject}: expected exactly five matching development examples")
        test_keys = {_question_key(doc) for doc in docs if doc["subject"] == subject}
        if test_keys.intersection(map(_question_key, examples)):
            raise ValueError(f"{subject}: overlapping test/development question and choices")
    # Same grouping logic as translated MMLU: first five development examples
    # from the current source config, never another test item.
    attach_prompts("mmlu", docs, dev_docs, shots)


def _write_jsonl(path, docs, expected_sha256):
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
    if _sha(temporary) != expected_sha256:
        raise ValueError(f"{path.name}: normalized input SHA256 differs from the pinned protocol; "
                         "the final file was not replaced. Inspect the .jsonl.tmp artifact.")
    temporary.replace(path)


def prepare_all(output_dir=None, cache_dir=None, offline=False):
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    private_cache = _cache_directory(output_dir, cache_dir)
    private_cache.mkdir(parents=True, exist_ok=True)
    pins = native_pins()
    test_docs, dev_docs, sources = [], [], []
    for config, expected in sorted(pins["configs"].items()):
        print(f"Preparing native GreekMMLU / {config}", flush=True)
        dataset, source = _load(config, private_cache, offline)
        if set(dataset) != {"test", "dev"}:
            raise ValueError(f"{config}: expected only test and dev splits")
        for split, target in (("test", test_docs), ("dev", dev_docs)):
            if len(dataset[split]) != expected[split]:
                raise ValueError(f"{config}/{split}: unexpected source row count")
            target.extend(canonicalize(config, split, dataset[split]))
        sources.append(source)
    if len(test_docs) != pins["expected_samples"]:
        raise ValueError("Unexpected native test row total")
    records = []
    for shots in (0, 5):
        docs = [dict(doc) for doc in test_docs]
        attach_native_prompts(docs, dev_docs, shots)
        benchmark = f"native_greekmmlu_{shots}shot"
        data_path = data_dir / f"{benchmark}.jsonl"
        _write_jsonl(data_path, docs, pins["data_sha256"][str(shots)])
        task, yaml_path = _write_yaml(benchmark, data_path, shots, output_dir)
        yaml_path.write_text(yaml_path.read_text(encoding="utf-8").replace(
            "protocol: greek_translated_reasoning_v1", f"protocol: {PROTOCOL}"), encoding="utf-8")
        records.append({
            "id": benchmark, "task": task, "task_name": task,
            "label": f"Native GreekMMLU ({shots}-shot)", "shots": shots,
            "expected_samples": len(docs), "data_path": str(data_path),
            "data_sha256": _sha(data_path), "yaml_path": str(yaml_path),
            "dataset": DATASET, "dataset_config": list(sorted(pins["configs"])),
            "dataset_revision": pins["revision"], "split": "test",
            "subject_counts": dict(Counter(x["subject"] for x in docs)),
            "group_counts": dict(Counter(x["group"] for x in docs)),
            "target_counts": dict(Counter(x["target"] for x in docs)),
            "choice_counts": dict(Counter(len(x["choices"]) for x in docs)),
            "mode": "generative", "output_type": "generate_until",
            "metric": "exact_match", "filter": "final-answer",
            "metric_key": "exact_match,final-answer", "sources": sources,
            "notes": ("Native dascim/GreekMMLU, not translated MMLU. All 45 subject/level configs. "
                      + ("No demonstrations. " if shots == 0 else
                         "First five source dev examples from the matching config, in source order. ")
                      + "Unified Greek reasoning/boxed ASCII A-D protocol; not identical to older native prompts. "
                        "Preserve source choice order and map source answer indices 0..3 to A..D. "
                        "Do not mix historical native or translated results into this run."),
        })
    suite = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"name": PROTOCOL, "language": "el", "max_gen_toks": 32768,
                     "final_answer": "Τελική απάντηση: \\boxed{A}",
                     "scoring": "Generated final-letter exact match, regex only. No LLM judge or likelihood scoring.",
                     "fewshot": "0-shot and 5-shot; matching-config development split only. No invented gold reasoning.",
                     "comparability": "A new unified prompt protocol; do not merge older native GreekMMLU scores."},
        "expected_samples_per_model": sum(x["expected_samples"] for x in records),
        "benchmarks": records, "tasks_dir": str(output_dir / "tasks"),
        "source_hashes": {path.relative_to(PROJECT).as_posix(): _sha(path) for path in
                          (PINS_PATH, HERE / "prepare_native.py", HERE / "prepare_data.py", HERE / "protocol.py")},
    }
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {suite_path}: {len(test_docs):,} questions per shot setting", flush=True)
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path,
                        help="Datasets cache; default follows HF cache variables or OUTPUT/cache/datasets")
    parser.add_argument("--offline", action="store_true", help="Use cached pinned inputs only; fail on cache misses")
    args = parser.parse_args()
    prepare_all(args.output_dir, args.cache_dir, args.offline)
