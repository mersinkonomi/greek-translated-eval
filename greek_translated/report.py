#!/usr/bin/env python3
"""Build an offline English report for the translated-Greek benchmark suite.

Only final, validated runs receive accuracy scores. Raw generation telemetry is
streamed and deduplicated by request_hash; it supplies progress, not a provisional
benchmark score. The HTML embeds a bounded sample preview, never the full corpus.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


METRIC = "exact_match,final-answer"
STDERR = "exact_match_stderr,final-answer"
PREVIEW_LIMIT = 8_000
EXAMPLES_PER_RUN = 50
INVALID = {"", "[invalid]", "invalid", "none", "null"}
COMPLETED = {"completed", "complete", "success", "succeeded"}


def is_mc2(item):
    """MC2 is a fractional probability-mass score, not a generated answer."""
    return (item.get("output_type") == "multiple_choice"
            or str(item.get("metric", "")) == "acc,none"
            or "mc2" in str(item.get("scoring_mode", "")).lower()
            or item.get("task") == "greektruthfulqa_mc2")


def metric_name(item):
    return str(item.get("metric") or ("acc,none" if is_mc2(item) else METRIC))


def stderr_name(item):
    metric, _, filter_name = metric_name(item).partition(",")
    return f"{metric}_stderr,{filter_name}"


def escape(value):
    return html.escape(str(value), quote=True)


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentage(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def count(value):
    return "—" if value is None else f"{int(value):,}"


def unwrap(value):
    while isinstance(value, list):
        value = value[0] if value else None
    return value


def preview(value, maximum=PREVIEW_LIMIT):
    value = "" if value is None else str(value)
    if len(value) <= maximum:
        return value
    edge = (maximum - 100) // 2
    return value[:edge] + f"\n\n[… {len(value) - 2 * edge:,} characters omitted; see full JSONL …]\n\n" + value[-edge:]


def link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.parent.resolve()), safe="/.-_")


def resolve(path, root):
    if not path:
        return None
    path = Path(path)
    return path if path.is_absolute() else root / path


def read_json(path, warnings):
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        warnings.append(f"Cannot read {path}: {error}")
        return None


def jsonl_rows(path, warnings):
    """Tolerate the incomplete last record of a file that is still being written."""
    try:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("expected a JSON object")
                except ValueError as error:
                    warnings.append(f"Skipped unreadable JSONL record {path}:{lineno}: {error}")
                    continue
                yield lineno, value
    except OSError as error:
        warnings.append(f"Cannot read {path}: {error}")


def read_telemetry(path, warnings):
    result = dict(unique=0, duplicates=0, missing_hash=0, final=0, final_known=0,
                  capped=0, finish_known=0, token_total=0, token_known=0,
                  token_max=0, finish_reasons={}, answer_formats={}, examples=[])
    if path is None or not path.is_file():
        return result
    seen, reasons, formats = set(), Counter(), Counter()
    for lineno, row in jsonl_rows(path, warnings):
        identity = row.get("request_hash")
        if not identity:
            result["missing_hash"] += 1
            continue
        if identity in seen:
            result["duplicates"] += 1
            continue
        seen.add(identity)
        result["unique"] += 1
        final = row.get("has_final_answer")
        if isinstance(final, bool):
            result["final_known"] += 1
            result["final"] += final
        reason = row.get("finish_reason")
        capped = row.get("capped")
        if reason is not None or isinstance(capped, bool):
            result["finish_known"] += 1
            result["capped"] += capped is True or str(reason).lower() in {"length", "max_tokens", "max_length"}
        reasons[str(reason or "unknown")] += 1
        formats[str(row.get("answer_format") or "unknown")] += 1
        tokens = numeric(row.get("generated_tokens"))
        if tokens is None and isinstance(row.get("token_ids"), list):
            tokens = len(row["token_ids"])
        if tokens is not None and tokens >= 0:
            result["token_known"] += 1
            result["token_total"] += int(tokens)
            result["token_max"] = max(result["token_max"], int(tokens))
        # Raw-only previews make the report useful before lm-eval writes samples.
        if len(result["examples"]) < min(EXAMPLES_PER_RUN, 12):
            result["examples"].append(dict(
                uid=str(row.get("uid", row.get("doc_id", identity))),
                subject=str(row.get("subject", row.get("task_name", ""))),
                question="Scored question record is not yet available; the exact prompt is shown below.",
                prompt=preview(row.get("prompt", "")), choices=[], target=None,
                prediction=str(row.get("extracted_answer", "")), correct=None,
                invalid=final is False, source="raw", source_path=str(path), line=lineno,
                response=preview(row.get("raw_response", row.get("response", row.get("engine_text", "")))),
                generated_tokens=tokens, finish_reason=reason,
            ))
    result["finish_reasons"] = dict(reasons)
    result["answer_formats"] = dict(formats)
    if result["missing_hash"]:
        warnings.append(f"{path}: {result['missing_hash']:,} raw records lack request_hash and were excluded from deduplicated progress.")
    return result


def read_likelihood_telemetry(path, warnings):
    """Count option requests separately from questions; never call them generations."""
    result = read_telemetry(None, warnings)
    result.update(question_ids=0, likelihood_tokens=0, likelihood_token_known=0,
                  finite_loglikelihoods=0)
    if path is None or not path.is_file():
        return result
    seen, questions = set(), set()
    for lineno, row in jsonl_rows(path, warnings):
        identity = row.get("request_hash")
        if not identity:
            result["missing_hash"] += 1
            continue
        if identity in seen:
            result["duplicates"] += 1
            continue
        seen.add(identity)
        result["unique"] += 1
        uid = row.get("uid", row.get("doc_id"))
        if uid is not None:
            questions.add((str(row.get("task_name", "")), str(uid)))
        total = numeric(row.get("loglikelihood", row.get("total_loglikelihood", row.get("total_logprob"))))
        result["finite_loglikelihoods"] += total is not None
        token_ids = row.get("continuation_token_ids", row.get("token_ids"))
        tokens = len(token_ids) if isinstance(token_ids, list) else numeric(row.get("continuation_tokens"))
        if tokens is not None and tokens >= 0:
            result["likelihood_token_known"] += 1
            result["likelihood_tokens"] += int(tokens)
        if len(result["examples"]) < min(EXAMPLES_PER_RUN, 12):
            details = {key: row[key] for key in (
                "option_index", "choice_index", "continuation", "answer", "loglikelihood",
                "total_loglikelihood", "total_logprob", "continuation_token_ids",
                "continuation_token_logprobs", "continuation_logprobs", "token_logprobs",
                "continuation_token_ranks", "token_ids", "context_token_ids", "is_greedy",
                "auxiliary_decode_tokens", "chat_template_applied") if key in row}
            result["examples"].append(dict(
                uid=str(uid if uid is not None else identity), subject="TruthfulQA MC2",
                question=str(row.get("question", "One answer option scored under the original TruthfulQA context.")),
                prompt=preview(row.get("context", row.get("prompt", ""))), choices=[],
                target=None, prediction=None, correct=None, invalid=False,
                source="raw-likelihood", scoring_mode="mc2", source_path=str(path), line=lineno,
                response="", likelihood_details=preview(json.dumps(details, ensure_ascii=False, indent=2)),
                mc2_score=None,
            ))
    result["question_ids"] = len(questions)
    if result["missing_hash"]:
        warnings.append(f"{path}: {result['missing_hash']:,} likelihood records lack request_hash and were excluded from deduplicated counts.")
    return result


def discover_result(item, output_dir, warnings):
    if output_dir is None or not output_dir.is_dir():
        return None, None, []
    task = item.get("task")
    metric = metric_name(item)
    candidates = []
    for path in sorted(output_dir.rglob("results_*.json")):
        data = read_json(path, warnings)
        if not data:
            continue
        entries = data.get("results", {})
        if not isinstance(entries, dict):
            continue
        selected = entries.get(task) if task else None
        if selected is None and not task:
            matches = [(key, value) for key, value in entries.items()
                       if isinstance(value, dict) and metric in value]
            if len(matches) == 1:
                task, selected = matches[0]
        if isinstance(selected, dict) and metric in selected:
            candidates.append((path, data, task))
    if not candidates:
        return None, None, []
    path, data, task = candidates[-1]
    if len(candidates) > 1:
        warnings.append(f"Run {item.get('index')}: found {len(candidates)} aggregate files; displaying newest matching task file {path.name}.")
    stamp = path.name[len("results_"):-len(".json")]
    files = sorted(path.parent.glob(f"samples_{task}_{stamp}.jsonl"))
    # Current and older harness versions can vary the sample filename prefix.
    if not files:
        files = sorted(path.parent.glob(f"samples_*_{stamp}.jsonl"))
    return path, data, files


def read_samples(paths, warnings, item=None):
    mc2 = is_mc2(item or {})
    selected_metric = metric_name(item or {})
    sample_metric = selected_metric.split(",", 1)[0]
    result = dict(unique=0, duplicates=0, correct=0, scored=0, invalid=0,
                  filter_known=0, disagreement=0, score_sum=0.0, subjects={}, examples=[])
    seen, subjects = set(), {}
    example_buckets = {"correct": [], "incorrect": [], "invalid": [], "mc2": []}
    for path in paths:
        for lineno, row in jsonl_rows(path, warnings):
            doc = row.get("doc") or {}
            identity = doc.get("uid")
            if identity is None:
                identity = (row.get("task_name", path.name.split("_20")[0]), row.get("doc_id", lineno))
            identity = str(identity)
            if identity in seen:
                result["duplicates"] += 1
                continue
            seen.add(identity)
            result["unique"] += 1
            if mc2:
                value = row.get(sample_metric, row.get(selected_metric))
                value = float(value) if isinstance(value, bool) else numeric(value)
                valid = value is not None and 0 <= value <= 1
                result["scored"] += valid
                result["score_sum"] += value if valid else 0.0
                if len(example_buckets["mc2"]) < EXAMPLES_PER_RUN:
                    targets = doc.get("mc2_targets") or {}
                    choices = targets.get("choices", doc.get("choices", []))
                    if not isinstance(choices, list):
                        choices = [str(choices)]
                    example_buckets["mc2"].append(dict(
                        uid=identity, subject="TruthfulQA MC2", question=preview(doc.get("question", "")),
                        choices=[preview(choice, 2_000) for choice in choices],
                        choice_labels=targets.get("labels", []), target=None, prediction=None,
                        correct=None, invalid=False, mc2_score=value if valid else None,
                        scoring_mode="mc2", response="", prompt=preview(doc.get("prompt", "")),
                        likelihood_details=preview(json.dumps({"option_likelihood_results": row.get("filtered_resps", row.get("resps")), "mc2_labels": targets.get("labels", [])}, ensure_ascii=False, indent=2)),
                        source="scored", source_path=str(path), line=lineno,
                        dataset=doc.get("dataset", "ilsp/truthful_qa_greek"), split=doc.get("split", "train"),
                    ))
                continue
            filtered = unwrap(row.get("filtered_resps"))
            prediction = "" if filtered is None else str(filtered).strip()
            target = str(row.get("target", doc.get("target", ""))).strip()
            filter_known = "filtered_resps" in row
            invalid = filter_known and prediction.lower() in INVALID
            metric = row.get(sample_metric, row.get(selected_metric))
            metric = float(metric) if isinstance(metric, bool) else numeric(metric)
            correct = bool(metric == 1) if metric in (0, 1) else None
            result["filter_known"] += filter_known
            result["invalid"] += invalid
            result["scored"] += correct is not None
            result["correct"] += correct is True
            result["score_sum"] += metric if correct is not None else 0.0
            if correct is not None and filter_known and target:
                result["disagreement"] += correct != (prediction == target)
            subject = str(doc.get("subject") or doc.get("category") or "All questions")
            group = subjects.setdefault(subject, dict(n=0, correct=0, scored=0, invalid=0))
            group["n"] += 1
            group["correct"] += correct is True
            group["scored"] += correct is not None
            group["invalid"] += invalid
            bucket = "invalid" if invalid else "correct" if correct is True else "incorrect"
            if len(example_buckets[bucket]) < EXAMPLES_PER_RUN:
                choices = doc.get("choices") or []
                if not isinstance(choices, list):
                    choices = [str(choices)]
                example_buckets[bucket].append(dict(
                    uid=identity, subject=subject, question=preview(doc.get("question", "")),
                    passage=preview(doc.get("passage", "")), choices=[preview(value, 2_000) for value in choices],
                    target=target, prediction=prediction, correct=correct, invalid=invalid,
                    response=preview(unwrap(row.get("resps"))), prompt=preview(doc.get("prompt", "")),
                    source="scored", source_path=str(path), line=lineno,
                    dataset=doc.get("dataset", ""), split=doc.get("split", ""),
                    demonstration_ids=doc.get("demonstration_ids", []),
                    option_permutation=doc.get("option_permutation"),
                ))
    # A deterministic balanced preview is for inspection, not an unbiased audit.
    while len(result["examples"]) < EXAMPLES_PER_RUN:
        changed = False
        for key in ("invalid", "incorrect", "correct", "mc2"):
            if example_buckets[key] and len(result["examples"]) < EXAMPLES_PER_RUN:
                result["examples"].append(example_buckets[key].pop(0))
                changed = True
        if not changed:
            break
    result["subjects"] = subjects
    return result


def assemble(root, manifest_path):
    warnings = []
    manifest = read_json(manifest_path, warnings)
    if not manifest:
        raise ValueError("A readable run manifest is required: " + "; ".join(warnings))
    records = manifest.get("runs", [])
    if isinstance(records, dict):
        records = list(records.values())
    runs = []
    for original in records:
        if not isinstance(original, dict):
            continue
        item = dict(original)
        state_path = root / "status" / f"{item.get('index')}.json"
        if state_path.is_file():
            state = read_json(state_path, warnings)
            if state:
                # State is allowed to update execution facts, never run identity.
                for key, value in state.items():
                    if key not in {"index", "model", "model_path", "benchmark", "task", "shots", "expected_samples", "results_dir", "raw_generation_path", "raw_likelihood_path", "cache_path", "output_type", "metric", "scoring_mode"}:
                        item[key] = value
        mc2 = is_mc2(item)
        expected = numeric(item.get("expected_samples"))
        expected = int(expected) if expected is not None and expected > 0 else None
        result_dir = resolve(item.get("results_dir"), root)
        raw_path = resolve(item.get("raw_likelihood_path" if mc2 else "raw_generation_path"), root)
        telemetry = read_likelihood_telemetry(raw_path, warnings) if mc2 else read_telemetry(raw_path, warnings)
        result_path, data, files = discover_result(item, result_dir, warnings)
        samples = read_samples(files, warnings, item)
        state = str(item.get("status", "pending")).lower()
        validation = []
        accuracy = stderr = None
        if state in COMPLETED:
            if expected is None:
                validation.append("Missing expected sample count in manifest.")
            if data is None:
                validation.append("No matching aggregate result file.")
            if samples["unique"] != expected:
                validation.append(f"Expected {count(expected)} unique scored samples; found {count(samples['unique'])}.")
            if samples["scored"] != samples["unique"]:
                validation.append("Some saved samples lack a finite MC2 score in [0, 1]." if mc2 else "Some saved samples have no binary exact-match metric.")
            if not mc2 and samples["filter_known"] != samples["unique"]:
                validation.append("Some saved samples lack extracted final-answer results.")
            if not mc2 and samples["disagreement"]:
                validation.append("Saved metrics disagree with the extracted answer and gold label.")
            if samples["duplicates"]:
                validation.append("Duplicate scored sample identities were found.")
            if data is not None:
                task = item.get("task")
                entries = data.get("results", {})
                entry = entries.get(task, {})
                if not task and len(entries) == 1:
                    task, entry = next(iter(entries.items()))
                saved_score = numeric(entry.get(metric_name(item)))
                saved_count = data.get("n-samples", {}).get(task, {})
                if saved_count.get("effective") != expected:
                    validation.append("Aggregate effective sample count differs from the manifest.")
                if saved_count.get("original") != expected and not manifest.get("pilot"):
                    validation.append("Aggregate original split count differs from the expected full dataset.")
                if data.get("config", {}).get("limit") is not None and not manifest.get("pilot"):
                    validation.append("Aggregate result was produced with a sample limit.")
                computed = samples["score_sum"] / samples["unique"] if samples["unique"] else None
                if saved_score is None or computed is None or not math.isclose(saved_score, computed, abs_tol=1e-9):
                    validation.append("Aggregate score does not match the mean of saved sample metrics.")
                if not validation:
                    accuracy, stderr = saved_score, numeric(entry.get(stderr_name(item)))
        if validation:
            state = "validation error"
            warnings.append(f"Run {item.get('index')}: " + " ".join(validation))
        runs.append(dict(item=item, state=state, expected=expected, mc2=mc2,
                         completed=state in COMPLETED and accuracy is not None,
                         accuracy=accuracy, stderr=stderr, validation=validation,
                         telemetry=telemetry, samples=samples, result_path=result_path,
                         sample_paths=files, raw_path=raw_path, state_path=state_path))
    return manifest, runs, warnings


CSS = """
:root{color-scheme:dark;--bg:#09111d;--panel:#111f30;--line:#26394d;--text:#edf4fc;--muted:#a4b5c9;--accent:#65e1cb;--warn:#ffd58a;--bad:#ffa3a3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:42px 28px 70px}h1{font-size:clamp(28px,4vw,46px);line-height:1.15;letter-spacing:-.035em;margin:10px 0 16px}h2{font-size:23px;margin:36px 0 14px}h3{font-size:17px;margin:0 0 10px}p{margin:10px 0}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.15em;font-size:12px}.muted,small{color:var(--muted)}small{display:block;font-size:12px}a{color:#85c8ff;text-decoration:none}a:hover{text-decoration:underline}code{font-family:ui-monospace,monospace;background:#071321;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}.intro{max-width:1000px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:25px 0}.card,.panel,details{border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:18px}.card strong{display:block;font-size:29px;line-height:1.4}.card span{color:var(--muted);font-size:13px}.scroll{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:15px 18px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:#101e2d}tr:last-child td{border-bottom:0}td strong{font-size:19px}.scorecell{min-width:210px}.pill{display:inline-block;font-size:11px;padding:3px 8px;border:1px solid var(--line);border-radius:20px;color:var(--muted);text-transform:uppercase;letter-spacing:.025em}.good{color:var(--accent)}.bad{color:var(--bad)}.warn{color:var(--warn)}.bar{width:100%;height:5px;background:#2a3848;border-radius:5px;margin:9px 0 5px;overflow:hidden}.bar i{display:block;background:var(--accent);height:100%}.note{padding:15px 18px;border-left:3px solid var(--warn);background:#252319;border-radius:0 8px 8px 0;font-size:14px}.protocol{display:grid;grid-template-columns:minmax(160px,230px) 1fr;gap:8px 20px;margin:0}.protocol dt{color:var(--muted)}.protocol dd{margin:0;overflow-wrap:anywhere}.controls{display:flex;flex-wrap:wrap;gap:10px;margin:15px 0}label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}select,input,button{background:#0c1724;color:var(--text);border:1px solid #38506a;padding:9px 12px;border-radius:7px;font:inherit}button{cursor:pointer}button:hover{border-color:var(--accent)}input{min-width:230px}.sample-list{display:grid;gap:12px}.sample-head{display:flex;justify-content:space-between;align-items:center;gap:15px}.sample-question{white-space:pre-wrap;overflow-wrap:anywhere}.choices{margin:10px 0;padding-left:24px}.choices li{padding:3px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#09131f;border-radius:8px;border:1px solid var(--line);padding:16px;max-height:650px;overflow:auto;font:13px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}summary{cursor:pointer;font-weight:600}details{margin:12px 0}details details{background:#0c1724}.files{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:10px}.mini{font-size:12px}.warnings{max-height:300px;overflow:auto}.footer{border-top:1px solid var(--line);padding-top:20px;margin-top:34px;font-size:13px;color:var(--muted)}.no-results{padding:25px;color:var(--muted)}@media(max-width:650px){main{padding:24px 14px}.protocol{grid-template-columns:1fr;gap:2px}.protocol dd{margin-bottom:12px}td,th{padding:12px}.sample-head{display:block}}
"""


JS = r"""
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const el=id=>document.getElementById(id);
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o)}
for(const b of DATA.benchmarks)option(el('benchmark-filter'),b.id,b.label);
for(const m of DATA.models)option(el('model-filter'),m,m);
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function codeDetail(parent,title,text){if(!text)return;const d=node('details');d.append(node('summary',title),node('pre',text));parent.append(d)}
function render(){
 const benchmark=el('benchmark-filter').value,model=el('model-filter').value,outcome=el('outcome-filter').value,query=el('sample-search').value.toLocaleLowerCase();
 const matches=DATA.samples.filter(s=>(!benchmark||s.benchmark===benchmark)&&(!model||s.model===model)&&(!outcome||(outcome==='invalid'&&s.invalid)||(outcome==='correct'&&s.correct===true)||(outcome==='incorrect'&&s.correct===false)||(outcome==='raw'&&s.source.startsWith('raw'))||(outcome==='mc2'&&s.scoring_mode==='mc2'))&&(!query||[s.uid,s.question,s.subject,s.response,s.likelihood_details].join(' ').toLocaleLowerCase().includes(query)));
 el('sample-count').textContent=`${matches.length.toLocaleString()} matching embedded previews; showing up to 30. Complete generation or likelihood records remain in the linked JSONL files.`;
 const target=el('sample-list');target.replaceChildren();
 for(const s of matches.slice(0,30)){
  const card=node('article',undefined,'panel'),head=node('div',undefined,'sample-head');
  head.append(node('h3',s.benchmark_label+' · '+s.model));
  const mc2=s.scoring_mode==='mc2',raw=s.source.startsWith('raw');
  const label=mc2?(raw?'MC2 · RAW OPTION LIKELIHOOD':'MC2 · PROBABILITY-MASS SCORE'):raw?'RAW · NOT SCORED':s.invalid?'INVALID FINAL':s.correct?'CORRECT':'INCORRECT';
  head.append(node('span',label,'pill '+(mc2?'':s.correct===true?'good':raw?'warn':'bad')));card.append(head);
  card.append(node('small',`Subject: ${s.subject||'—'} · ID: ${s.uid}`));
  if(s.passage)codeDetail(card,'Passage',s.passage);
  card.append(node('p',s.question,'sample-question'));
  if(s.choices&&s.choices.length){const list=node('ol',undefined,'choices');list.type=mc2?'1':'A';s.choices.forEach((choice,i)=>list.append(node('li',String(choice)+(mc2&&s.choice_labels&&s.choice_labels[i]!==undefined?` [reference label: ${s.choice_labels[i]===1?'true':'false'}]`:''))));card.append(list)}
  if(mc2){card.append(node('p',s.mc2_score===null||s.mc2_score===undefined?'This raw option record is not a completed question score.':`MC2 probability mass on true answers: ${(s.mc2_score*100).toFixed(4)}% (not binary answer accuracy).`,'mini'));codeDetail(card,'Saved option likelihoods / token log probabilities',s.likelihood_details)}
  else card.append(node('p',`Extracted: ${s.prediction||'—'} · Reference: ${s.target===null||s.target===undefined?'not scored yet':s.target}`,'mini'));
  if(s.generated_tokens!==undefined)card.append(node('small',`Generated tokens: ${s.generated_tokens??'unknown'} · Finish reason: ${s.finish_reason??'unknown'}`));
  if(!mc2)codeDetail(card,'Model response (start/end preview)',s.response);codeDetail(card,mc2?'Original likelihood context preview':'Exact prompt preview',s.prompt);
  if(s.demonstration_ids&&s.demonstration_ids.length)codeDetail(card,'Demonstration IDs',JSON.stringify(s.demonstration_ids,null,2));
  if(s.option_permutation)codeDetail(card,'Option permutation',JSON.stringify(s.option_permutation));
  const a=node('a',`Full ${s.source==='raw-likelihood'?'raw-likelihood':s.source==='raw'?'raw-generation':'scored-sample'} JSONL · line ${s.line}`);a.href=s.source_url;card.append(a);target.append(card);
 }
 if(!matches.length)target.append(node('p','No matching sample previews are available yet.','no-results'));
}
for(const id of ['benchmark-filter','model-filter','outcome-filter'])el(id).addEventListener('change',render);
el('sample-search').addEventListener('input',render);render();
"""


def status_class(state):
    if state in COMPLETED:
        return "good"
    if any(word in state for word in ("fail", "error", "cancel", "timeout", "out_of_memory")):
        return "bad"
    return "warn" if state in {"running", "submitted", "pending"} else ""


def score_cell(run):
    if run is None:
        return '<td class="scorecell"><span class="muted">Not scheduled</span></td>'
    state = f'<span class="pill {status_class(run["state"])}">{escape(run["state"])}</span>'
    if run["completed"]:
        error = "" if run["stderr"] is None else f' · ±{run["stderr"] * 100:.2f} pp SE'
        if run.get("mc2"):
            return f'<td class="scorecell"><strong>{percentage(run["accuracy"])}</strong><small>MC2 mean probability mass · {count(run["samples"]["unique"])} questions{error}</small>{state}</td>'
        return f'<td class="scorecell"><strong>{percentage(run["accuracy"])}</strong><small>{count(run["samples"]["correct"])} / {count(run["samples"]["unique"])} correct{error}</small>{state}</td>'
    if run.get("mc2"):
        return (f'<td class="scorecell"><strong class="muted">—</strong> {state}'
                f'<small>{count(run["telemetry"]["unique"])} option likelihoods saved</small>'
                f'<small>{count(run["telemetry"].get("question_ids", 0))} question IDs observed · {count(run["expected"])} planned questions</small></td>')
    generated, expected = run["telemetry"]["unique"], run["expected"]
    width = min(100, 100 * generated / expected) if expected else 0
    return (f'<td class="scorecell"><strong class="muted">—</strong> {state}'
            f'<small>{count(generated)} / {count(expected)} unique generations saved</small>'
            f'<div class="bar"><i style="width:{width:.2f}%"></i></div></td>')


def protocol_rows(manifest):
    protocol = manifest.get("protocol") or {}
    if not isinstance(protocol, dict):
        protocol = {"description": protocol}
    records = manifest.get("runs", [])
    records = list(records.values()) if isinstance(records, dict) else records
    has_mc2 = any(is_mc2(item) for item in records if isinstance(item, dict))
    rows = [("Run ID", manifest.get("run_id", "unknown")),
            ("Output and metric", "Generative tasks: generate_until · exact_match,final-answer. TruthfulQA exception: multiple_choice · acc,none (original MC2)." if has_mc2 else "generate_until · exact_match,final-answer · one selected option per question"),
            ("Generative scoring", "Only the final regex-extracted answer is scored on generative tasks. Missing or malformed final answers count as incorrect; model explanations are not graded."),
            ("Accepted generative final formats", "Generative prompts request a boxed answer. Extraction accepts a terminal boxed letter, or a terminal letter explicitly marked Τελική απάντηση:. Native thinking must close first. The displayed letter must be valid for the current choices."),
            ("Scope", "Translated Greek benchmarks only. This is not the native GreekMMLU (dascim/GreekMMLU) evaluation."),
            ("Comparability", "Generative exact-match is a different protocol from standard likelihood-based multiple-choice evaluation. No overall average is computed across these different benchmarks."),
            ("Sample previews", f"At most {EXAMPLES_PER_RUN} scored previews per run. Generative previews balance available correct, incorrect and invalid answers" + ("; MC2 previews use the first available scored questions." if has_mc2 else ".") + " These are inspection samples, not representative random audits."),
            ("Saved outputs", "Generative runs retain complete responses, generated token IDs and termination metadata. MC2 runs retain per-option contexts, continuations and token log probabilities in raw_likelihoods.jsonl; they do not generate reasoning or answer text." if has_mc2 else "Complete responses, generated token IDs and termination metadata remain in raw JSONL files. Scored sample JSONL files contain questions, choices, references and extracted answers."),
            ("Resume accounting", "Raw requests deduplicate by request_hash. Generation token totals exclude likelihood tokens. A capped generative answer is reported separately. MC2 option-request counts are not completed question counts." if has_mc2 else "Raw progress and token counts deduplicate by request_hash. A length-capped answer is reported separately; it is not silently replaced or treated as complete reasoning.")]
    if has_mc2:
        rows.extend([
            ("TruthfulQA MC2", "Original Greek task greektruthfulqa_mc2 with original answer order and mc2_targets. For each question, answer-sequence likelihoods are normalized across all options, and the probability mass of all true answers is summed. The reported score is the mean of these fractional [0,1] values, not a count of correct selected letters."),
            ("TruthfulQA prompting", "0 additional few-shot examples; the original task's six fixed Greek Q&A priming examples remain. The original likelihood context is used, with no reasoning-generation instruction, boxed-answer regex or answer-option shuffling."),
        ])
    elif any("truthful" in str(item.get("benchmark", "")).lower() for item in records if isinstance(item, dict)):
        rows.append(("TruthfulQA (legacy generative protocol)", "Generative single-best-answer accuracy with deterministically shuffled options. This historical mode is not original probability-based MC2."))
    for key, value in protocol.items():
        text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        rows.append((str(key).replace("_", " ").capitalize(), text))
    for key in ("backend", "dtype", "seed", "max_length", "generation", "repository", "repository_branch", "repository_commit", "source_git_head", "harness_git_head"):
        if key in manifest:
            value = manifest[key]
            rows.append((key.replace("_", " ").capitalize(), json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)))
    return "".join(f"<dt>{escape(key)}</dt><dd>{escape(value)}</dd>" for key, value in rows)


def render(root, manifest_path, output, manifest, runs, warnings):
    models = list(dict.fromkeys(str(run["item"].get("model", "Unknown model")) for run in runs))
    benchmarks = {}
    lookup = {}
    for run in runs:
        item = run["item"]
        key = str(item.get("benchmark", item.get("task", "unknown")))
        benchmarks.setdefault(key, dict(id=key, label=str(item.get("label", key)), shots=item.get("shots", item.get("num_fewshot")), expected=run["expected"], mc2=run["mc2"]))
        pair = (key, str(item.get("model", "Unknown model")))
        if pair in lookup:
            warnings.append(f"Multiple manifest runs for {pair}; overview displays the last record.")
        lookup[pair] = run
    complete = sum(run["completed"] for run in runs)
    generation_runs = [run for run in runs if not run["mc2"]]
    likelihood_runs = [run for run in runs if run["mc2"]]
    has_mc2 = bool(likelihood_runs)
    raw_count = sum(run["telemetry"]["unique"] for run in generation_runs)
    token_count = sum(run["telemetry"]["token_total"] for run in generation_runs)
    expected_total = sum(run["expected"] or 0 for run in generation_runs)
    likelihood_count = sum(run["telemetry"]["unique"] for run in likelihood_runs)
    has_native = any(str(run["item"].get("benchmark", "")).startswith("native_greekmmlu_") for run in runs)
    native_only = has_native and all(str(run["item"].get("benchmark", "")).startswith("native_greekmmlu_") for run in runs)
    title = ("Native GreekMMLU · Generative Evaluation" if native_only else
             "Greek Benchmarks · Native and Translated" if has_native else
             "Translated Greek · Generation + TruthfulQA MC2" if has_mc2 else "Translated Greek · Generative Evaluation")
    eyebrow = ("NATIVE GREEKMMLU · REGEX-SCORED GENERATION" if native_only else
               "GREEK BENCHMARKS · NATIVE AND TRANSLATED" if has_native else
               "GREEK TRANSLATED BENCHMARKS · MIXED PROTOCOL" if has_mc2 else "GREEK TRANSLATED BENCHMARKS · REGEX-SCORED GENERATION")
    introduction = ("Most benchmarks use generated reasoning and a regex-extracted final answer. TruthfulQA is the explicit exception: its original MC2 likelihood-based probability-mass score is preserved. These metrics are displayed separately; no LLM judge is used." if has_mc2 else "Models may explain their reasoning and must finish with a marked, boxed answer letter. Scores below come from saved final-answer matches, never from likelihoods or an LLM judge.")
    exception_note = ("TruthfulQA alone uses original MC2 likelihood scoring, not generated-answer accuracy. " if has_mc2 else "Generative scores use extracted final-answer exact match, not original likelihood metrics. ")
    dataset_note = ("This is native dascim/GreekMMLU, not translated ilsp/mmlu_greek. The 0-shot and 5-shot variants are separate evaluations of the same test questions. " if native_only else
                    "Native dascim/GreekMMLU and translated datasets are distinct benchmarks, shown separately. " if has_native else
                    "These are translated Greek datasets, not native GreekMMLU. ")
    if manifest.get("pilot"):
        title = "PILOT ONLY · " + title
        dataset_note = ("PILOT / SMOKE TEST ONLY: these limited samples are not full benchmark results "
                        "and must not enter the final comparison. " + dataset_note)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pieces = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body><main>',
              f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>',
              f'<p class="intro muted">{introduction}</p>',
              f'<p class="mini muted">Updated {now} · Run <code>{escape(manifest.get("run_id", root.name))}</code> · <a href="{link(manifest_path, output)}">Run manifest</a></p>',
              '<div class="cards">',
              f'<div class="card"><strong>{complete} / {len(runs)}</strong><span>Validated completed runs</span></div>',
              f'<div class="card"><strong>{len(benchmarks)}</strong><span>Benchmark variants · {len(models)} models</span></div>',
              f'<div class="card"><strong>{count(raw_count)}</strong><span>Unique generations saved / {count(expected_total)} planned generative questions</span></div>',
              f'<div class="card"><strong>{count(token_count)}</strong><span>Generated tokens · excludes likelihood scoring</span></div>',
              (f'<div class="card"><strong>{count(likelihood_count)}</strong><span>MC2 option likelihoods saved · {count(sum(run["expected"] or 0 for run in likelihood_runs))} planned MC2 question evaluations</span></div>' if has_mc2 else "") + '</div>',
              f'<p class="note">{dataset_note}{exception_note}The evaluated counts below come from the actual selected splits.</p>',
              '<h2>Benchmark comparison</h2><div class="scroll"><table><thead><tr><th>Benchmark / protocol</th>' + ''.join(f'<th>{escape(model)}</th>' for model in models) + '</tr></thead><tbody>']
    for key, benchmark in benchmarks.items():
        shots = f'{benchmark["shots"]}-shot' if benchmark["shots"] is not None else "See manifest"
        if benchmark["mc2"]:
            shots = "0 added shots · original 6 fixed priming Q&As"
        metric_label = "ORIGINAL MC2 · LIKELIHOOD" if benchmark["mc2"] else "GENERATIVE · EXACT MATCH"
        pieces.append(f'<tr><td><b>{escape(benchmark["label"])}</b><small>{escape(shots)} · {count(benchmark["expected"])} evaluation questions</small><span class="pill">{metric_label}</span></td>')
        pieces.extend(score_cell(lookup.get((key, model))) for model in models)
        pieces.append('</tr>')
    pieces.append('</tbody></table></div><p class="mini muted">A score is shown only after a run is marked completed and its aggregate mean and unique saved samples pass completeness checks. A dash is unavailable, not zero. Generative accuracy is correct/total; MC2 is mean normalized probability mass, not a binary question count. Raw likelihood progress counts answer options separately from question IDs.</p>' if has_mc2 else '</tbody></table></div><p class="mini muted">Accuracy is shown only after a run is marked completed and its aggregate score and unique saved samples pass completeness checks. A dash is unavailable, not zero. Raw progress counts unique cached generation requests.</p>')
    pieces.append('<h2>Generation completion and answer format</h2><div class="scroll"><table><thead><tr><th>Benchmark / model</th><th>State / job</th><th>Scored invalid finals</th><th>Raw final-answer coverage</th><th>Output limit reached</th><th>Tokens per generation</th></tr></thead><tbody>')
    for run in generation_runs:
        item, telemetry, samples = run["item"], run["telemetry"], run["samples"]
        invalid = f'{count(samples["invalid"])} / {count(samples["filter_known"])}' if samples["filter_known"] else "—"
        final = f'{count(telemetry["final"])} / {count(telemetry["final_known"])}' if telemetry["final_known"] else "—"
        formats = telemetry["answer_formats"]
        if any(key in formats for key in ("boxed", "marked-letter")):
            final += f'<small>{count(formats.get("boxed", 0))} boxed · {count(formats.get("marked-letter", 0))} marked letter</small>'
        capped = f'{count(telemetry["capped"])} / {count(telemetry["finish_known"])}' if telemetry["finish_known"] else "—"
        average = f'{telemetry["token_total"] / telemetry["token_known"]:,.1f} mean' if telemetry["token_known"] else "—"
        pieces.append(f'<tr><td>{escape(item.get("label", item.get("benchmark")))}<small>{escape(item.get("model"))}</small></td><td><span class="pill {status_class(run["state"])}">{escape(run["state"])}</span><small>{escape(item.get("job_id", item.get("jobid", "—")))}</small></td><td>{invalid}</td><td>{final}</td><td>{capped}</td><td>{average}<small>{count(telemetry["token_max"])} max · {count(telemetry["token_known"])} records with token counts</small></td></tr>')
    pieces.append('</tbody></table></div><p class="mini muted">Telemetry denominators include only records with the relevant metadata. Raw final-answer coverage is formatting information, not accuracy. Output-limit termination can coexist with a final answer. Neither a final marker nor EOS alone proves that reasoning is correct.</p>')
    if has_mc2:
        pieces.append('<h2>TruthfulQA MC2 likelihood diagnostics</h2><p class="muted">No answer text is generated for this benchmark. Regex extraction, final-answer coverage, generation tokens and output-length caps are not applicable.</p><div class="scroll"><table><thead><tr><th>Model</th><th>State</th><th>Scored questions</th><th>Option requests saved</th><th>Question IDs observed</th><th>Continuation tokens scored</th></tr></thead><tbody>')
        for run in likelihood_runs:
            item, telemetry = run["item"], run["telemetry"]
            pieces.append(f'<tr><td>{escape(item.get("model"))}</td><td><span class="pill {status_class(run["state"])}">{escape(run["state"])}</span></td><td>{count(run["samples"]["scored"])} / {count(run["expected"])}</td><td>{count(telemetry["unique"])}<small>{count(telemetry["finite_loglikelihoods"])} with finite total log likelihood</small></td><td>{count(telemetry["question_ids"])}<small>At least one saved option, not necessarily all</small></td><td>{count(telemetry["likelihood_tokens"])}<small>{count(telemetry["likelihood_token_known"])} option records with token counts</small></td></tr>')
        pieces.append('</tbody></table></div>')
    pieces.append('<h2>Subject breakdown</h2>')
    subject_sections = 0
    for key, benchmark in benchmarks.items():
        if "mmlu" not in (key + " " + benchmark["label"]).lower():
            continue
        available = [lookup.get((key, model)) for model in models]
        subject_names = sorted({subject for run in available if run and run["completed"] for subject in run["samples"]["subjects"]})
        if not subject_names:
            continue
        subject_sections += 1
        pieces.append(f'<details><summary>{escape(benchmark["label"])} · {len(subject_names)} subjects</summary><div class="scroll"><table><thead><tr><th>Subject</th>' + ''.join(f'<th>{escape(model)}</th>' for model in models) + '</tr></thead><tbody>')
        for subject in subject_names:
            pieces.append(f'<tr><td>{escape(subject.replace("_", " "))}</td>')
            for run in available:
                values = run["samples"]["subjects"].get(subject) if run and run["completed"] else None
                if values and values["scored"] == values["n"] and values["n"]:
                    pieces.append(f'<td>{percentage(values["correct"] / values["n"])}<small>{count(values["correct"])} / {count(values["n"])} correct · {count(values["invalid"])} invalid</small></td>')
                else:
                    pieces.append('<td class="muted">—</td>')
            pieces.append('</tr>')
        pieces.append('</tbody></table></div></details>')
    if not subject_sections:
        pieces.append('<p class="muted">MMLU-family subject scores will appear when complete runs have been validated.</p>')
    pieces.extend(['<h2>Inspect generated answers and MC2 likelihoods</h2>' if has_mc2 else '<h2>Inspect model answers</h2>', '<p class="muted">Embedded generation previews include formatting failures and wrong answers when available. MC2 previews show fractional question scores and option likelihoods, never a generated response. Full original records are retained in JSONL files.</p>' if has_mc2 else '<p class="muted">Embedded previews are bounded and intentionally include formatting failures and wrong answers when available. Full text is never truncated in the original JSONL files.</p>',
                   '<div class="controls"><label>Benchmark<select id="benchmark-filter"><option value="">All benchmarks</option></select></label><label>Model<select id="model-filter"><option value="">All models</option></select></label><label>Outcome<select id="outcome-filter"><option value="">All previews</option><option value="correct">Correct generated answer</option><option value="incorrect">Incorrect generated answer</option><option value="invalid">Invalid final answer</option><option value="raw">Raw / not yet scored</option>' + ('<option value="mc2">MC2 likelihood records</option>' if has_mc2 else '') + '</select></label><label>Search previews<input id="sample-search" type="search" placeholder="Question, response, subject or ID"></label></div><p id="sample-count" class="mini muted"></p><div id="sample-list" class="sample-list"></div><noscript>Enable JavaScript to browse the embedded sample previews. File links and score tables work without JavaScript.</noscript>',
                   '<h2>Methodology and provenance</h2><div class="panel"><dl class="protocol">' + protocol_rows(manifest) + '</dl></div>'])
    facts = manifest.get("suitefacts", manifest.get("suite_facts", manifest.get("suite", manifest.get("benchmarks"))))
    if facts:
        pieces.append('<details><summary>Dataset splits, demonstrations and suite facts</summary><pre>' + escape(json.dumps(facts, indent=2, ensure_ascii=False)) + '</pre></details>')
    pieces.append('<h2>Files and run diagnostics</h2><p class="mini muted">The HTML is self-contained for tables and embedded previews. Full-file links are relative: keep this report with its results directory when copying it to another computer, or download the desired JSONL files separately.</p>')
    embedded = []
    for run in runs:
        item = run["item"]
        key = str(item.get("benchmark", item.get("task", "unknown")))
        label = str(item.get("label", key))
        examples = run["samples"]["examples"] or run["telemetry"]["examples"]
        for original in examples:
            sample = dict(original, benchmark=key, benchmark_label=label, model=str(item.get("model")))
            sample["source_url"] = link(sample.pop("source_path"), output)
            embedded.append(sample)
        pieces.append(f'<details><summary>Run {escape(item.get("index"))} · {escape(label)} · {escape(item.get("model"))} · {escape(run["state"])}</summary>')
        diag = {"state": run["state"], "job_id": item.get("job_id", item.get("jobid")),
                "expected_samples": run["expected"], "unique_scored_samples": run["samples"]["unique"],
                "metric": metric_name(item), "output_type": item.get("output_type", "multiple_choice" if run["mc2"] else "generate_until"),
                "duplicate_raw_records_ignored": run["telemetry"]["duplicates"],
                "model_path": item.get("model_path"),
                "chat_template": item.get("chat_template", item.get("apply_chat_template")), "validation_errors": run["validation"]}
        if run["mc2"]:
            diag.update(unique_option_likelihood_requests=run["telemetry"]["unique"],
                        question_ids_with_any_options=run["telemetry"]["question_ids"],
                        finite_option_loglikelihoods=run["telemetry"]["finite_loglikelihoods"],
                        continuation_tokens_scored=run["telemetry"]["likelihood_tokens"])
        else:
            diag.update(unique_raw_generations=run["telemetry"]["unique"],
                        finish_reasons=run["telemetry"]["finish_reasons"],
                        answer_formats=run["telemetry"]["answer_formats"],
                        generation_kwargs=item.get("generation_kwargs", item.get("generation")))
        for name in ("error", "message", "started_at", "finished_at", "returncode", "exit_code", "scheduler_state"):
            if name in item:
                diag[name] = item[name]
        pieces.append('<pre>' + escape(json.dumps(diag, indent=2, ensure_ascii=False)) + '</pre><div class="files">')
        paths = [(run["result_path"], "Aggregate results"), (run["raw_path"], "Full raw option likelihoods" if run["mc2"] else "Full raw generations"), (run["state_path"], "Run status")]
        log = resolve(item.get("log_path"), root)
        if log:
            paths.append((log, "Job log"))
        paths.extend((path, "Scored samples · " + path.name) for path in run["sample_paths"])
        for path, name in paths:
            if path and path.is_file():
                pieces.append(f'<a href="{link(path, output)}">{escape(name)}</a>')
        pieces.append('</div></details>')
    if warnings:
        pieces.append(f'<details open><summary class="warn">Data warnings ({len(warnings)})</summary><ul class="warnings">' + ''.join('<li>' + escape(warning) + '</li>' for warning in warnings) + '</ul></details>')
    payload = json.dumps(dict(models=models, benchmarks=list(benchmarks.values()), samples=embedded), ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    pieces.append(f'<div class="footer">Generated from the run manifest, immutable saved model outputs and harness result files. No external scripts, fonts, analytics or network requests. Refreshed {now}.</div><script type="application/json" id="report-data">{payload}</script><script>{JS}</script></main></body></html>')
    return "\n".join(pieces)


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.results_dir.resolve()
    manifest_path = (args.manifest or root / "manifest.json").resolve()
    manifest, runs, warnings = assemble(root, manifest_path)
    content = render(root, manifest_path, args.output.resolve(), manifest, runs, warnings)
    write_atomic(args.output.resolve(), content)
    print(json.dumps({"output": str(args.output.resolve()), "runs": len(runs),
                      "completed": sum(run["completed"] for run in runs), "warnings": len(warnings)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
