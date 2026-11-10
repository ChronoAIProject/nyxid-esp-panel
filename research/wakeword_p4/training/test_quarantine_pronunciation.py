"""Boundary checks for training-only pronunciation screening."""
import json
from pathlib import Path
import tempfile
import unittest

from quarantine_pronunciation import canonical_prefix, create_view, digest


class PronunciationViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/"base"
        self.audit = self.root/"audit"
        self.output = self.root/"view"
        self.selection = self.root/"selection.jsonl"
        (self.base/"dataset").mkdir(parents=True)
        self.audit.mkdir()
        (self.base/"ambient").mkdir()
        (self.base/"ambient.json").write_text("[]\n")
        (self.base/"features.bin").write_bytes(b"original-feature-bytes")
        human = self.root/"previous-view"/"nested-human"
        human.mkdir(parents=True)
        (human/"h.npy").write_bytes(b"human-features")
        (self.base/"base-human").symlink_to(human.parent, target_is_directory=True)
        (self.base/"human.json").write_text(json.dumps([
            {"split": "testing", "path": "base-human/nested-human/h.npy"}]))
        self.rows = []
        for name, split, label, text in [
            ("good", "training", 1, "Hey Nyx"),
            ("bad", "training", 1, "Hey Nyx"),
            ("ambiguous", "training", 0, "Hey next"),
            ("ordinary", "training", 0, "Ordinary speech"),
            ("heldout", "validation", 1, "Hey Nyx"),
            ("test", "testing", 0, "Hey next")]:
            self.rows.append({"path": name+".wav", "split": split, "label": label,
                              "text": text, "sha256": name, "engine": "test-engine"})
        self.manifest = self.base/"dataset"/"manifest.jsonl"
        self.manifest.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        features = [{**r, "offset": i*10, "frames": 10} for i, r in enumerate(self.rows)]
        (self.base/"features.json").write_text(json.dumps(features))
        chosen = self.rows[:3]
        self.selection.write_text("".join(json.dumps(r)+"\n" for r in chosen))
        predictions = [{"path": r["path"], "sha256": r["sha256"], "source_label": r["label"],
                        "expected": r["text"], "phonemes": ["h", "eɪ", "n", "ɪ", "k", "s"]}
                       for r in chosen]
        predictions[1]["phonemes"] = ["h", "eɪ", "n", "ɛ", "k", "s", "t"]
        self.predictions = self.audit/"phonemes.jsonl"
        self.predictions.write_text("".join(json.dumps(r)+"\n" for r in predictions))
        self.report = {"source_manifest_sha256": digest(self.manifest),
                       "selection_sha256": digest(self.selection), "split": "training", "clips": 3,
                       "recognizer": {"model_sha256": "test-only"}}
        self.save_report()

    def save_report(self):
        (self.audit/"report.json").write_text(json.dumps(self.report))

    def run_view(self):
        return create_view(self.base, self.audit, self.selection, self.output)

    def test_preserves_holdouts_labels_bytes_and_chained_human_targets(self):
        result = self.run_view()
        rows = json.loads((self.output/"features.json").read_text())
        originals = json.loads((self.base/"features.json").read_text())
        self.assertEqual(rows, [r for r in originals if r["path"] not in ("bad.wav", "ambiguous.wav")])
        self.assertEqual(result["quarantined_sources"], 2)
        self.assertEqual((self.output/"features.bin").resolve(), (self.base/"features.bin").resolve())
        human = json.loads((self.output/"human.json").read_text())[0]
        self.assertEqual((self.output/human["path"]).read_bytes(), b"human-features")
        self.assertEqual(len((self.manifest).read_text().splitlines()), 6)

    def test_rejects_incomplete_audit_before_creating_view(self):
        self.predictions.write_text(self.predictions.read_text().splitlines()[0]+"\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_rejects_audit_label_tampering(self):
        rows = [json.loads(s) for s in self.predictions.read_text().splitlines()]
        rows[0]["source_label"] = 0
        self.predictions.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "identities differ"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_rejects_heldout_selection_even_with_matching_hash(self):
        self.selection.write_text(self.selection.read_text()+json.dumps(self.rows[4])+"\n")
        self.report["selection_sha256"] = digest(self.selection)
        self.save_report()
        with self.assertRaisesRegex(ValueError, "training-only"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_rejects_empty_positive_engine(self):
        rows = [json.loads(s) for s in self.predictions.read_text().splitlines()]
        rows[0]["phonemes"] = []
        self.predictions.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "empties"):
            self.run_view()
        self.assertFalse(self.output.exists())

    def test_rule_distinguishes_next_and_enforces_first_twelve_phones(self):
        self.assertTrue(canonical_prefix(["h", "eɪ", "n", "ɪ", "k", "s"]))
        self.assertTrue(canonical_prefix(["ɛ", "n", "i", "k", "s"]))
        self.assertFalse(canonical_prefix(["h", "eɪ", "n", "ɛ", "k", "s", "t"]))
        self.assertFalse(canonical_prefix(["x"]*8+["eɪ", "n", "ɪ", "k", "s"]))


if __name__ == "__main__":
    unittest.main()
