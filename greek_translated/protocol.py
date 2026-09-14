"""Greek prompts and deterministic extraction of terminal generated answers."""

import os
import re
import string

INVALID = "[invalid]"
LABELS = string.ascii_uppercase
REQUIRE_THINKING_CLOSE_ENV = "GREEK_TRANSLATED_REQUIRE_THINKING_CLOSE"
THINKING_PAIRS = {
    "<think>": "</think>",
    "<ifm|think>": "</ifm|think>",
    "<ifm|think_fast>": "</ifm|think_fast>",
    "<ifm|think_faster>": "</ifm|think_faster>",
    "<|ifm|think|>": "<|ifm|/think|>",
    "<|ifm|think>": "<|ifm|/think>",
}
THINKING_RE = re.compile("|".join(re.escape(x) for pair in THINKING_PAIRS.items() for x in pair))
QUESTION_RE = re.compile(r"(?im)^[ \t]*(?:Ερώτηση|Νέα ερώτηση|Επόμενη ερώτηση|Question)[ \t]*:")
STOP_RE = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|ifm\|endoftext\|>|<\|ifm\|im_end\|>|</s>)[ \t\r\n]*\Z")
MARKER = r"(?:\*\*)?Τελική[ \t]+απάντηση(?:\*\*)?[ \t]*:"
LETTER = r"(?:[A-Za-zΑΒΓΔαβγδ]|\\(?:Gamma|Delta|Alpha|Beta|gamma|delta|alpha|beta))"
CONTENTS = rf"(?:\\(?:text|mathrm|mathbf|mathsf|operatorname)\s*\{{\s*({LETTER})\s*\}}|({LETTER}))[ \t]*[.)]?"
BOX_RE = re.compile(
    rf"(?im)^[ \t]*(?:{MARKER}[ \t]*)?(?:\*\*)?[ \t]*"
    rf"(?:\$+|\\\[|\\\()?\s*\\boxed\s*\{{\s*{CONTENTS}\s*\}}"
    rf"\s*(?:\$+|\\\]|\\\))?(?:\*\*)?[ \t]*[.!]?[ \t]*\Z"
)
BARE_RE = re.compile(rf"(?im)^[ \t]*{MARKER}[ \t]*(?:\*\*)?[ \t]*({LETTER})[ \t]*[.)]?(?:\*\*)?[ \t]*\Z")


def _thinking_state(text):
    pending, close_end, malformed = [], None, False
    for match in THINKING_RE.finditer(text):
        token = match.group()
        if token in THINKING_PAIRS:
            pending.append(THINKING_PAIRS[token])
        else:
            if pending:
                if pending[-1] != token:
                    malformed = True
                else:
                    pending.pop()
            close_end = match.end()
    return close_end, bool(pending) or malformed


def reasoning_is_closed(text):
    """Whether native thinking, possibly opened in the prompt, was closed."""
    if not isinstance(text, str):
        return False
    close_end, pending = _thinking_state(text)
    return close_end is not None and not pending


def _current_response(text, preopened):
    pending, inherited = [], preopened
    events = sorted([(m.start(), "think", m) for m in THINKING_RE.finditer(text)] + [(m.start(), "question", m) for m in QUESTION_RE.finditer(text)])
    for position, kind, match in events:
        if kind == "question":
            if not pending and not inherited:
                return text[:position].strip()
            continue
        token = match.group()
        if token in THINKING_PAIRS:
            pending.append(THINKING_PAIRS[token])
        else:
            if pending and pending[-1] == token:
                pending.pop()
            inherited = False
    return text.strip()


def _normalize(label):
    greek = dict(zip("ΑΒΓΔαβγδ", "ABCDABCD"))
    latex = {"alpha": "A", "beta": "B", "gamma": "C", "delta": "D"}
    if label in greek:
        return greek[label]
    if label.startswith("\\"):
        return latex.get(label[1:].lower(), INVALID)
    return label.upper()


def _extract(text, num_choices, require_thinking_close=False):
    if not isinstance(text, str) or not 2 <= num_choices <= len(LABELS):
        return INVALID, "invalid"
    text = _current_response(text, require_thinking_close)
    close_end, pending = _thinking_state(text)
    if pending or (require_thinking_close and close_end is None):
        return INVALID, "invalid"
    if close_end is not None:
        text = text[close_end:].strip()
    while STOP_RE.search(text):
        text = STOP_RE.sub("", text).strip()
    match, fmt = BOX_RE.search(text), "boxed"
    if match is None:
        match, fmt = BARE_RE.search(text), "marked-letter"
    if match is None:
        return INVALID, "invalid"
    label = _normalize(next(x for x in match.groups() if x is not None))
    return (label, fmt) if label in LABELS[:num_choices] else (INVALID, "invalid")


def extract_final_answer(text, num_choices, require_thinking_close=False):
    return _extract(text, num_choices, require_thinking_close)[0]


def answer_format(text, num_choices, require_thinking_close=False):
    return _extract(text, num_choices, require_thinking_close)[1]


def extract_final_answers(resps, docs):
    require_close = os.environ.get(REQUIRE_THINKING_CLOSE_ENV, "0") == "1"
    return [[extract_final_answer(text, len(doc["choices"]), require_close) for text in batch] for batch, doc in zip(resps, docs)]


class FinalAnswerFilter:
    def apply(self, resps, docs):
        return extract_final_answers(resps, docs)


def doc_to_text(doc):
    return doc["prompt"]


def doc_to_target(doc):
    return doc["target"]


def question_text(doc):
    """Read input fields only, never current gold or rationale."""
    choices = doc["choices"]
    if not 2 <= len(choices) <= len(LABELS):
        raise ValueError(f"Expected 2–26 choices, got {len(choices)}")
    parts = []
    if doc.get("passage"):
        parts.append("Κείμενο:\n" + doc["passage"])
    parts.append("Ερώτηση: " + doc["question"])
    parts.append("\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices)))
    return "\n\n".join(parts)


def build_prompt(doc, demonstrations):
    parts = [
        "Λύσε την ερώτηση πολλαπλής επιλογής και εξήγησε τον συλλογισμό σου στα ελληνικά. "
        "Στο τέλος γράψε μία ξεχωριστή γραμμή της μορφής «Τελική απάντηση: \\boxed{γράμμα}», "
        "αντικαθιστώντας το «γράμμα» με το λατινικό γράμμα της επιλογής που θεωρείς σωστή. "
        "Το πλαίσιο πρέπει να περιέχει μόνο ένα γράμμα. Μη γράψεις άλλη ερώτηση ή κείμενο μετά την τελική γραμμή."
    ]
    if demonstrations:
        parts.append("Ακολουθούν παραδείγματα με τις τελικές απαντήσεις τους (χωρίς αναλυτικό συλλογισμό).")
        for index, example in enumerate(demonstrations, 1):
            parts.append(f"Παράδειγμα {index}\n{question_text(example)}\n\nΤελική απάντηση: \\boxed{{{example['target']}}}")
        parts.append("Τώρα απάντησε στην επόμενη ερώτηση με τον δικό σου συλλογισμό.")
    parts.append(question_text(doc))
    parts.append("Επίτρεπτά γράμματα: " + ", ".join(LABELS[:len(doc["choices"])]) + ".\n\nΑπάντηση:")
    return "\n\n".join(parts)
