import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from greek_translated import prepare_native as native
from greek_translated.protocol import extract_final_answer


def row(question="Ποια επιλογή;", answer=2, choices=None):
    return {"question": question, "choices": choices or ["ένα", "δύο", "τρία", "τέσσερα"],
            "answer": answer, "subject": "Mathematics", "level": "University", "group": "STEM"}


def fixture():
    test = native.canonicalize("Mathematics", "test", [row("Ερώτηση εξέτασης;")])
    dev = native.canonicalize("Mathematics", "dev", [row(f"Παράδειγμα {i};", i % 4) for i in range(5)])
    return test, dev


class NativePreparationTests(unittest.TestCase):
    def test_pins_have_exact_historical_native_scope(self):
        pins = native.native_pins()
        self.assertEqual(len(pins["configs"]), 45)
        self.assertEqual(sum(x["test"] for x in pins["configs"].values()), 16632)
        self.assertEqual(sum(x["dev"] for x in pins["configs"].values()), 225)
        self.assertEqual(pins["revision"], "6a03aa06b68beb932fb75edff3a34e50b3674649")

    def test_index_to_ascii_preserves_two_to_four_choices(self):
        for n in (2, 3, 4):
            with self.subTest(choices=n):
                doc = native.canonicalize("Mathematics", "test", [row(answer=n-1, choices=[str(i) for i in range(n)])])[0]
                self.assertEqual(doc["choices"], [str(i) for i in range(n)])
                self.assertEqual(doc["target"], "ABCD"[n-1])
                self.assertEqual(doc["subject"], "Mathematics")
                self.assertEqual(doc["source_answer_index"], n-1)
                self.assertIn("/test/0", doc["uid"])

    def test_bad_gold_and_choice_counts_fail(self):
        for value in (-1, 4, True, "Γ", 1.2):
            with self.subTest(answer=value), self.assertRaises(ValueError):
                native.canonicalize("Mathematics", "test", [row(answer=value)])
        with self.assertRaises(ValueError):
            native.canonicalize("Mathematics", "test", [row(answer=2, choices=["A", "B"])])
        with self.assertRaises(ValueError):
            native.canonicalize("Mathematics", "test", [row(answer=0, choices=["A"])])

    def test_zero_shot_has_no_demo_or_gold_in_current_prompt(self):
        docs, dev = fixture()
        native.attach_native_prompts(docs, dev, 0)
        self.assertEqual(docs[0]["demonstration_ids"], [])
        self.assertNotIn("Παράδειγμα", docs[0]["prompt"])
        changed = copy.deepcopy(docs)
        changed[0]["target"] = "A"
        native.attach_native_prompts(changed, dev, 0)
        self.assertEqual(changed[0]["prompt"], docs[0]["prompt"])

    def test_five_shot_matches_dev_config_and_order(self):
        docs, dev = fixture()
        other = native.canonicalize("Law", "dev", [row(f"Άλλο {i};") for i in range(5)])
        native.attach_native_prompts(docs, other + dev, 5)
        self.assertEqual(docs[0]["demonstration_ids"], [x["uid"] for x in dev])
        self.assertNotIn("Άλλο", docs[0]["prompt"])
        self.assertEqual(docs[0]["shots"], 5)

    def test_test_examples_and_same_question_leakage_rejected(self):
        docs, dev = fixture()
        bad = copy.deepcopy(dev)
        bad[0]["split"] = "test"
        with self.assertRaises(ValueError):
            native.attach_native_prompts(docs, bad, 5)
        bad = copy.deepcopy(dev)
        bad[0]["question"] = docs[0]["question"]
        with self.assertRaises(ValueError):
            native.attach_native_prompts(docs, bad, 5)
        with self.assertRaises(ValueError):
            native.attach_native_prompts(docs, dev[:4], 5)

    def test_regex_normalizes_greek_letters_and_rejects_out_of_range(self):
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{Γ}", 4), "C")
        self.assertEqual(extract_final_answer("Τελική απάντηση: \\boxed{Γ}", 2), "[invalid]")

    def test_prepare_two_records_and_reject_hash_drift(self):
        docs, dev = fixture()
        pins = {"configs": {"Mathematics": {"test": 1, "dev": 5}},
                "expected_samples": 1, "revision": "a" * 40, "data_sha256": {}}
        for shots in (0, 5):
            batch = copy.deepcopy(docs)
            native.attach_native_prompts(batch, dev, shots)
            pins["data_sha256"][str(shots)] = hashlib.sha256(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in batch).encode()).hexdigest()
        source_rows = {"test": [row("Ερώτηση εξέτασης;")],
                       "dev": [row(f"Παράδειγμα {i};", i % 4) for i in range(5)]}
        with tempfile.TemporaryDirectory() as temp, patch.object(native, "native_pins", return_value=pins), \
                patch.object(native, "_load", return_value=(source_rows, {"revision": "a" * 40})):
            suite = native.prepare_all(Path(temp))
            self.assertEqual(suite["expected_samples_per_model"], 2)
            self.assertEqual([x["id"] for x in suite["benchmarks"]],
                             ["native_greekmmlu_0shot", "native_greekmmlu_5shot"])
            for record in suite["benchmarks"]:
                self.assertEqual(record["metric_key"], "exact_match,final-answer")
                self.assertEqual(record["subject_counts"], {"Mathematics": 1})
                self.assertIn(native.PROTOCOL, Path(record["yaml_path"]).read_text())
            final = Path(suite["benchmarks"][0]["data_path"])
            before = final.read_bytes()
            pins["data_sha256"]["0"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                native.prepare_all(Path(temp))
            self.assertEqual(final.read_bytes(), before)

    def test_loader_pins_revision_and_honors_offline(self):
        with patch("datasets.load_dataset", return_value={"test": [], "dev": []}) as loader:
            native._load("Mathematics", Path("cache"), offline=True)
        args, kwargs = loader.call_args
        self.assertEqual(args, (native.DATASET, "Mathematics"))
        self.assertEqual(kwargs["revision"], native.native_pins()["revision"])
        self.assertTrue(kwargs["download_config"].local_files_only)


if __name__ == "__main__":
    unittest.main()
