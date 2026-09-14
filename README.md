# Greek LLM evaluation: translated benchmarks + native GreekMMLU

Evaluate **local Hugging Face checkpoints** on another Linux/NVIDIA machine.
No cluster-specific account, model directory, or Slurm configuration is required.

Two distinct suites are included:

| Suite / benchmark | Questions per model | Few-shot examples | Score |
|---|---:|---:|---|
| Translated MMLU Greek | 14,042 | 5, matching-subject dev | Generated final-letter accuracy |
| Translated MMLU-Pro Greek | 12,032 | 0 | Generated final-letter accuracy |
| Global-MMLU Greek | 14,042 | 5, matching-subject dev | Generated final-letter accuracy |
| ARC Greek Easy | 2,376 | 25, training | Generated final-letter accuracy |
| ARC Greek Challenge | 1,168 | 25, training | Generated final-letter accuracy |
| HellaSwag Greek | 10,024 | 10, training | Generated final-letter accuracy |
| TruthfulQA Greek | 817 | Original prompt | **Original MC2 probability mass** |
| Belebele Greek | 900 | 5, other test passages | Generated final-letter accuracy |
| **Native GreekMMLU, 0-shot** | **16,632** | **0** | **Generated final-letter accuracy** |
| **Native GreekMMLU, 5-shot** | **16,632** | **5, matching-subject dev** | **Generated final-letter accuracy** |

Translated: **55,401 questions/model**. Native: **45 subjects**, 16,632 test
questions evaluated twice, or **33,264 evaluations/model** across 0/5-shot.
These are different datasets: translated MMLU uses `ilsp/mmlu_greek`;
native GreekMMLU uses `dascim/GreekMMLU`. Their results are never pooled into one accuracy.

[Ελληνικές οδηγίες](QUICKSTART_EL.md) ·
[Prompt for the assistant on the other machine](PROMPT_FOR_OTHER_MACHINE.md) ·
[Model configuration example](models.example.yaml)

## 1. Install in a fresh environment

Use Linux, Python 3.11, a supported NVIDIA GPU/driver, and enough GPU memory for
your checkpoint and the requested context. This runner uses vLLM, not Apple MPS,
CPU inference, GGUF, or a remote inference API. One checkpoint must fit on one
GPU; multiple GPUs run independent evaluations rather than tensor parallelism.

After cloning this repository and changing into its directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install -r requirements-greek.txt --torch-backend=auto
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
```

The pinned core is vLLM 0.21.0, PyTorch 2.11.0, Transformers 5.5.4 and
datasets 4.8.5. The source cluster used CUDA 13.0; this is **not a promise that the
same binary works on every GPU**. Let `uv` select the Torch backend for the local
driver, then validate an actual GPU pilot. Some kernels require a matching CUDA
toolkit and C++ compiler. Do not replace system CUDA or a shared environment
without the machine owner's approval.
See [vLLM 0.21 GPU installation](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/).

For CPU regression checks only, using the installed environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  greek_translated/test_protocol.py \
  tests/test_greek_translated_backend.py \
  tests/test_greek_translated_report.py \
  tests/test_portable_data.py \
  tests/test_portable_native.py \
  tests/test_portable_runner.py
```

Five optional full-data integration checks run when `GREEK_EVAL_TEST_SUITE`
points to a prepared translated `suite.json` and `GREEK_EVAL_TEST_CACHE` points
to its populated dataset cache. Without these, only those five are skipped;
the synthetic original-MC2 and all other CPU regressions still run.

## 2. Supply your local model paths

```bash
cp models.example.yaml models.yaml
```

Edit `models.yaml`: replace the example paths, remove unwanted model entries,
and add your own. Each path must point to an actual checkpoint/snapshot directory
with weights, `config.json`, and tokenizer files—not the parent Hugging Face cache.
Relative paths and paths with spaces should be quoted appropriately in YAML.

| Profile | Formatting / intended checkpoint |
|---|---|
| `k2_base` | Horizon 3.7B Base; raw completion, native BOS, preserved grouped normalization |
| `qwen_instruct` | Qwen3.5 Instruct; native chat/thinking formatting |
| `qwen_base` | Qwen3.5 Base; raw completion |
| `generic_chat` | Other vLLM-supported chat checkpoint with a tokenizer chat template |
| `generic_base` | Other vLLM-supported base checkpoint; raw completion |

For Horizon Base, use **both model and tokenizer from `mid_4`**, resolved
commit `981a91d15a76f97c22ebf704d6026a84bf72bdb2`.
The default/main Horizon checkpoint from the earlier experiment was **Instruct**.
Do not rename or reuse it as Base.

A local path alone does not establish its upstream revision. Weight/configuration
hashes are recorded for reproducibility; any supplied revision label must describe
the checkpoint you actually downloaded. This project does not download models.

For `generic_base`, inspect the tokenizer's native special-token convention.
Set `add_bos_token: true` when it requires a leading BOS (the generic default is
false). K2 Base always keeps its BOS; Qwen presets retain their tested settings.

Set `trust_remote_code: true` only for checkpoints whose custom Python code you
trust. “Local” does not make custom model code safe to execute. Other architectures
are supported only if the installed vLLM/Transformers backend supports them; a
successful pilot is required, not assumed.

Defaults preserve 65,536 context tokens and up to 32,768 generated tokens. No
question is silently truncated. Adjust batch/concurrency/memory fractions to fit
your GPU; reducing context, output length, shots, or changing sampling defines a
different evaluation protocol and must be recorded. On shared GPUs reserve memory
through that machine's scheduler; a free-memory check is not a reservation.

## 3. Prepare both datasets

Internet access is needed the first time; immutable source revisions are pinned
in `greek_translated/dataset_pins.json` and `native_pins.json`.

```bash
export HF_HOME="$PWD/artifacts/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
python -m greek_translated.prepare_data --output-dir artifacts/translated-suite
python -m greek_translated.prepare_native --output-dir artifacts/native-suite
```

Both commands accept `--cache-dir PATH` and `--offline`.
The exports above keep both Hub snapshots and dataset caches local even if your
shell inherited another machine's cache settings. `--cache-dir` alone controls
the dataset cache, not the separate Hub snapshot cache.
Offline preparation requires the exact pinned datasets to be cached already.
Prepared data and task YAMLs are generated under the chosen output directory;
regenerate them on the destination machine rather than copying old absolute-path
manifests. Do not re-prepare inputs or edit evaluation code during an active run.

Datasets and weights are **not committed** to Git. Dataset access and licenses
remain those of their original publishers. No Hugging Face token should be put
in the configuration, code, shell history, or a published artifact.

## 4. Pilot, then full evaluation

Example using visible GPUs `0,1` (use `0` for one GPU).
If `CUDA_VISIBLE_DEVICES` is already set, these indices refer to that restricted
visible list; do not override a scheduler allocation.

Run the translated suite:

```bash
python -m greek_translated.local prepare \
  --config models.yaml --suite artifacts/translated-suite/suite.json \
  --run-dir artifacts/runs/translated-pilot --pilot

python -m greek_translated.local run \
  --run-dir artifacts/runs/translated-pilot --devices 0,1

python -m greek_translated.local prepare \
  --config models.yaml --suite artifacts/translated-suite/suite.json \
  --run-dir artifacts/runs/translated-full

python -m greek_translated.local run \
  --run-dir artifacts/runs/translated-full --devices 0,1 \
  --pilot-run-dir artifacts/runs/translated-pilot
```

Run native GreekMMLU **0-shot and 5-shot**:

```bash
python -m greek_translated.local prepare \
  --config models.yaml --suite artifacts/native-suite/suite.json \
  --run-dir artifacts/runs/native-pilot --pilot

python -m greek_translated.local run \
  --run-dir artifacts/runs/native-pilot --devices 0,1

python -m greek_translated.local prepare \
  --config models.yaml --suite artifacts/native-suite/suite.json \
  --run-dir artifacts/runs/native-full

python -m greek_translated.local run \
  --run-dir artifacts/runs/native-full --devices 0,1 \
  --pilot-run-dir artifacts/runs/native-pilot
```

Each pilot scores two questions per benchmark/model. It checks GPU loading,
saved responses, extraction, and completeness; it is **not a benchmark score**.
Inspect its raw outputs, including invalid finals, before the full run.
The full run checks that a matching pilot passed. Do not use the explicit
unpiloted override to bypass a failed pilot.

Run the two full-suite commands sequentially when using the same GPUs.
On a remote terminal, use your scheduler or a persistent terminal session such
as `tmux` so disconnecting your laptop does not stop the evaluation.

To resume an interrupted run, rerun its same `local run` command. Do **not**
rerun `prepare` into an existing run directory or delete caches. Completed
cases are validated and skipped; already saved responses in incomplete cases are
reused. Changing model weights, prepared data, code, or configuration requires a
new run directory and matching pilot. Checkpoint hashing can take time for large
models.

## 5. Results, raw answers, and HTML reports

```bash
python -m greek_translated.local report \
  --run-dir artifacts/runs/translated-full --require-complete
python -m greek_translated.local report \
  --run-dir artifacts/runs/native-full --require-complete
```

Each run directory contains its own frozen `manifest.json`, statuses/logs,
response cache, benchmark/model output folders, and HTML report. The commands
print the actual report paths. Reports are refreshed as workers finish; the
strict final command exits unsuccessfully if any model/benchmark is incomplete.

Look for:

- `raw_generations.jsonl`: full original responses, reasoning, special tokens,
  token IDs, finish reasons, extracted answers, and request identifiers.
- `raw_likelihoods.jsonl`: original TruthfulQA MC2 option likelihoods.
- `results_*.json`: aggregate scores and configuration.
- `samples_*.jsonl`: question, choices, gold answer, raw response, extracted
  prediction and per-question score.
- English HTML: per-benchmark and per-subject scores, missing/invalid answers,
  capped outputs, token statistics, previews, and links to the full artifacts.

HTML tables and embedded previews open offline in a browser. To retain links to
the **full** JSONL files when copying a report to your laptop, copy its entire
run directory, not just the HTML. Keep dataset/model licenses in mind before
sharing raw outputs or questions.

## Protocol details and limitations

Generative tasks request Greek reasoning, ending with
`Τελική απάντηση: \boxed{A}`. A deterministic terminal-answer regex accepts
the displayed letter (including supported Greek-letter equivalents). Missing,
malformed, or unfinished finals count as incorrect. There is no LLM judge.
Raw text is kept intact; scoring never replaces the saved original response.
Native-thinking profiles must close their thinking section before the final.

Native GreekMMLU uses this portable shared prompt and ASCII choice labels.
Its earlier cluster-only prompt variants used different formatting; **do not
merge their historical scores or response caches with this new protocol**.

TruthfulQA is deliberately the exception: its original Greek **MC2** task,
option order, true-answer labels, and six fixed priming Q&As are retained.
There are zero *additional sampled* examples. No regex/chat/reasoning prompt is
applied. The score is probability mass on true options, not single-answer accuracy.

Belebele has no separate dev split here: its five demonstrations are other test
passages, excluding the current passage, question, and source link. This is not
disjoint-development evaluation. MMLU-Pro is 0-shot because this pinned Greek
release has no dev split; source English/gold rationales never enter prompts.
One source MMLU-Pro answer/index disagreement is recorded and the upstream
answer letter remains authoritative.

Saved seeds and pinned inputs improve reproducibility, but identical numbers
across different GPU kernels, architectures, batch scheduling, or library builds
are not guaranteed. Do not change sampling, force valid outputs, or edit gold
answers to improve scores.

## Upstream

Based on [yangzhang33/lm-evaluation-harness, branch el](https://github.com/yangzhang33/lm-evaluation-harness/tree/el),
commit `a3af1f6a8de2fcf9f3840124f898d7d40f680cf2`, derived from EleutherAI's
evaluation harness. Original attribution and MIT license are retained.
See [upstream documentation](README_UPSTREAM.md). The portable entry point is
`greek_translated.local`, **not** a cluster-specific `run_eval.sh`.
The upstream broad CI and PyPI publication workflows are retained as inactive
reference files under `docs/upstream-workflows/`; pushing this fork does not
publish a Python package or launch unrelated model-download tests.
