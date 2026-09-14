"""Validate scored samples against frozen inputs and lossless raw exports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from greek_translated.protocol import INVALID, extract_final_answer


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def validate(item, manifest):
    files = sorted(Path(item["results_dir"]).rglob("results_*.json"))
    if not files:
        raise RuntimeError("No aggregate result was saved")
    result_path = files[-1]
    result = json.loads(result_path.read_text())
    task = item["task"]
    if set(result["configs"]) != {task} or set(result["n-samples"]) != {task}:
        raise RuntimeError("The evaluated tasks differ from the explicit Greek task")
    count = result["n-samples"][task]["effective"]
    if count != item["expected_samples"]:
        raise RuntimeError(f"Incomplete scored dataset: {count}/{item['expected_samples']}")
    if not manifest["pilot"] and (result["n-samples"][task]["original"] != count
                                  or result.get("config", {}).get("limit") is not None):
        raise RuntimeError("A limited evaluation cannot be accepted as a full benchmark")
    bench = next(b for b in manifest["suite"]["benchmarks"] if b["task"] == task)
    expected = {}
    is_mc2 = item["output_type"] == "multiple_choice"
    for doc in rows(bench["data_path"]):
        if len(expected) >= count and manifest["pilot"]:
            break
        uid = doc["uid"]
        if uid in expected:
            raise RuntimeError("Duplicate UID in frozen input")
        expected[uid] = (doc["mc2_targets"] if is_mc2 else
                         {"target": doc["target"], "choices": doc["choices"]})
    if len(expected) != count:
        raise RuntimeError("Frozen input count differs from planned evaluation")
    raw_records = {}
    raw_path = item["raw_likelihood_path"] if is_mc2 else item["raw_generation_path"]
    hashes = set()
    for raw in rows(raw_path):
        if raw.get("task_name") != task or raw.get("uid") not in expected:
            raise RuntimeError("Raw export contains an unexpected task or question")
        if not raw.get("request_hash") or raw["request_hash"] in hashes:
            raise RuntimeError("Missing or duplicate raw request identity")
        hashes.add(raw["request_hash"])
        key = (raw["uid"], raw["option_index"]) if is_mc2 else raw["uid"]
        if key in raw_records:
            raise RuntimeError("Multiple raw requests for one question/option")
        if is_mc2:
            value = float(raw["total_loglikelihood"])
            probabilities = raw["continuation_token_logprobs"]
            if (not math.isfinite(value) or len(probabilities) != len(raw["continuation_token_ids"])
                    or not all(math.isfinite(x) for x in probabilities)
                    or not math.isclose(value, sum(probabilities), abs_tol=1e-9)):
                raise RuntimeError("Invalid or incomplete raw option likelihood")
            raw_records[key] = value
        else:
            if len(raw["token_ids"]) != raw["generated_tokens"] or "finish_reason" not in raw:
                raise RuntimeError("Raw response lacks token/termination metadata")
            raw_records[key] = hashlib.sha256(raw["raw_response"].encode()).hexdigest()
    raw_expected = sum(len(value["choices"]) for value in expected.values()) if is_mc2 else count
    if len(raw_records) != raw_expected:
        raise RuntimeError("Raw exports do not cover every planned question and option")
    stamp = result_path.name.removeprefix("results_").removesuffix(".json")
    samples = list(result_path.parent.glob(f"samples_{task}_{stamp}.jsonl"))
    if len(samples) != 1:
        raise RuntimeError("Expected one matching scored sample export")
    seen, scores, valid_finals = set(), [], 0
    if is_mc2:
        from lm_eval.tasks.greektruthfulqa.utils import process_results_mc2
    for row in rows(samples[0]):
        uid = row["doc"]["uid"]
        if uid in seen or uid not in expected:
            raise RuntimeError("Duplicate or unexpected scored sample")
        seen.add(uid)
        if not row.get("resps") or not row.get("filtered_resps"):
            raise RuntimeError("Missing raw response or score extraction")
        if is_mc2:
            if row["doc"]["mc2_targets"] != expected[uid] or row.get("filter") != "none":
                raise RuntimeError("TruthfulQA targets or original filter changed")
            options = row["filtered_resps"]
            if len(options) != len(expected[uid]["choices"]):
                raise RuntimeError("MC2 must score every original option")
            typed = []
            for index, (raw, filtered) in enumerate(zip(row["resps"], options, strict=True)):
                if (len(raw) != 1 or len(filtered) != 2 or raw[0] != filtered
                        or str(filtered[1]).lower() not in {"true", "false"}):
                    raise RuntimeError("Inconsistent MC2 option response")
                value = float(filtered[0])
                if not math.isfinite(value) or not math.isclose(value, raw_records[(uid, index)], abs_tol=1e-9):
                    raise RuntimeError("Saved option score differs from raw likelihood")
                typed.append((value, str(filtered[1]).lower() == "true"))
            score = float(process_results_mc2(row["doc"], typed)["acc"])
            if not math.isfinite(score) or not 0 <= score <= 1 or not math.isclose(score, float(row["acc"]), abs_tol=1e-9):
                raise RuntimeError("MC2 score differs from the original task's probability mass")
        else:
            if any(row["doc"][key] != expected[uid][key] for key in ("target", "choices")):
                raise RuntimeError("Scored sample differs from frozen choices/reference")
            raw = row["resps"][0][0]
            if hashlib.sha256(raw.encode()).hexdigest() != raw_records[uid]:
                raise RuntimeError("Scored response differs from the lossless raw export")
            extracted = extract_final_answer(raw, len(expected[uid]["choices"]), item["native_thinking"])
            if row["filtered_resps"][0] != extracted:
                raise RuntimeError("Saved extraction differs from regex extraction")
            valid_finals += extracted != INVALID
            score = float(extracted == expected[uid]["target"])
            if float(row["exact_match"]) != score:
                raise RuntimeError("Sample exact-match score disagrees with reference")
        scores.append(score)
    if seen != set(expected) or len(raw_records) != raw_expected:
        raise RuntimeError("Raw/scored exports do not cover every planned question and option")
    aggregate = float(result["results"][task][item["metric"]])
    if not math.isfinite(aggregate) or not math.isclose(aggregate, sum(scores) / count, abs_tol=1e-9):
        raise RuntimeError("Aggregate score differs from saved sample mean")
    if manifest["pilot"] and not is_mc2 and not valid_finals:
        raise RuntimeError("Pilot produced no extractable final answers; inspect raw outputs before a full run")
    return result_path, count
