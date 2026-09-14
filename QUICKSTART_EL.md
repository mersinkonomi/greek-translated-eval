# Εκτέλεση σε άλλο μηχάνημα

Το repo υποστηρίζει **και τα translated benchmarks και το native GreekMMLU**.
Τα μοντέλα πρέπει να υπάρχουν ήδη σε τοπικούς φακέλους Hugging Face. Δεν
ανεβαίνουν ούτε κατεβαίνουν βάρη μοντέλων από αυτή τη διαδικασία.

Χρειάζεται Linux με συμβατή NVIDIA GPU, όχι εκτέλεση απευθείας σε Mac/MPS.
Οι πλήρεις οδηγίες εγκατάστασης και πρωτοκόλλου είναι στο [README](README.md).

## Εγκατάσταση

Κάνε clone το repo στο μηχάνημα με τις GPU και μπες στον φάκελό του. Αν το repo
είναι private, χρησιμοποίησε τον δικό σου συνδεδεμένο λογαριασμό GitHub.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install -r requirements-greek.txt --torch-backend=auto
python -m pip check
nvidia-smi
cp models.example.yaml models.yaml
```

Στο `models.yaml` βάλε τα **πραγματικά τοπικά paths** και το σωστό profile:
`k2_base`, `qwen_instruct`, `qwen_base`, `generic_chat` ή `generic_base`.
Διέγραψε από το config όσα μοντέλα δεν θέλεις. Για Horizon Base χρειάζονται
μοντέλο και tokenizer από `mid_4`, όχι το παλιό main/instruct checkpoint.

## Προετοιμασία δεδομένων

```bash
export HF_HOME="$PWD/artifacts/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
python -m greek_translated.prepare_data --output-dir artifacts/translated-suite
python -m greek_translated.prepare_native --output-dir artifacts/native-suite
```

Αυτή η φάση κατεβάζει τα pinned datasets, όχι μοντέλα. Το native GreekMMLU
περιλαμβάνει 45 θεματικές και 16.632 ερωτήσεις για καθένα από τα 0-shot/5-shot.
Η μεταφρασμένη σουίτα περιλαμβάνει 55.401 ερωτήσεις ανά μοντέλο.

## Εκτέλεση και των δύο

Οι παρακάτω εντολές τρέχουν πρώτα pilot και μετά την πλήρη αξιολόγηση για κάθε
σουίτα, χωρίς να επικαλύπτουν τις ίδιες GPU. Άλλαξε το `0,1` σε `0` για μία GPU
ή στους επιτρεπόμενους ορατούς δείκτες της κατανομής σου.

```bash
set -euo pipefail
for suite in translated native; do
  python -m greek_translated.local prepare \
    --config models.yaml --suite "artifacts/$suite-suite/suite.json" \
    --run-dir "artifacts/runs/$suite-pilot" --pilot
  python -m greek_translated.local run \
    --run-dir "artifacts/runs/$suite-pilot" --devices 0,1
  python -m greek_translated.local prepare \
    --config models.yaml --suite "artifacts/$suite-suite/suite.json" \
    --run-dir "artifacts/runs/$suite-full"
  python -m greek_translated.local run \
    --run-dir "artifacts/runs/$suite-full" --devices 0,1 \
    --pilot-run-dir "artifacts/runs/$suite-pilot"
  python -m greek_translated.local report \
    --run-dir "artifacts/runs/$suite-full" --require-complete
done
```

Μην ξανατρέξεις το `prepare` σε υπάρχον run. Για συνέχεια μετά από διακοπή,
ξανατρέξε μόνο το αντίστοιχο `run` με τα ίδια paths. Τα ολοκληρωμένα αποτελέσματα
ελέγχονται και παραλείπονται· οι ήδη αποθηκευμένες απαντήσεις επαναχρησιμοποιούνται.
Χρησιμοποίησε scheduler ή `tmux` για να μη σταματήσει η εκτέλεση αν κλείσει το SSH.

## Τι αποθηκεύεται

Στο `artifacts/runs/translated-full/` και `artifacts/runs/native-full/` θα βρεις
ξεχωριστά αγγλικά HTML reports, manifests, logs και αναλυτικά αρχεία αποτελεσμάτων.
Οι πλήρεις απαντήσεις αποθηκεύονται στα `raw_generations.jsonl` και οι ανά
ερώτηση βαθμολογίες στα `samples_*.jsonl`.

Το TruthfulQA παραμένει **original MC2** με πιθανότητες: έχει
`raw_likelihoods.jsonl`, όχι παραγόμενη απάντηση. Όλα τα υπόλοιπα είναι
generative με regex για το τελικό γράμμα.

Το HTML ανοίγει offline. Αν θέλεις να λειτουργούν και οι σύνδεσμοι προς ολόκληρες
τις raw απαντήσεις στο laptop, αντέγραψε όλο τον φάκελο του run, όχι μόνο το HTML.

Έτοιμο κείμενο για τον βοηθό στο άλλο μηχάνημα:
[PROMPT_FOR_OTHER_MACHINE.md](PROMPT_FOR_OTHER_MACHINE.md).
