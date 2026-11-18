"""Persistence and resumption checks for the connection reviewer."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.review_matches import (
    REQUIRED_FIELDS, load_review, needs_review, reviewed_notes, save_decision,
)


class MatchReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "match_review.csv"
        self.fields = sorted(REQUIRED_FIELDS) + ["extra"]
        self.rows = []
        for index in range(2):
            row = dict.fromkeys(self.fields, "")
            row.update(connection_id=f"paper:paragraph:{index}:c0", match="uncertain",
                       review_notes="human review needed: Check the restriction.",
                       evidence='[{"block_id": "abstract", "quote": "A, B\\nC"}]',
                       extra="Unicode α, commas, and\nnewlines")
            self.rows.append(row)
        self.write_rows()

    def write_rows(self):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_saves_only_review_fields_and_resumes(self):
        result = save_decision(self.path, self.rows[0], "yes", self.rows[0]["review_notes"])
        fields, rows = load_review(self.path)
        self.assertEqual(fields, self.fields)
        self.assertEqual(rows[1], self.rows[1])
        for field in set(fields) - {"match", "review_notes"}:
            self.assertEqual(rows[0][field], self.rows[0][field])
        self.assertEqual(result, rows[0])
        self.assertEqual(rows[0]["review_notes"], "Check the restriction.")
        self.assertEqual([row["connection_id"] for row in rows if needs_review(row)],
                         [self.rows[1]["connection_id"]])

    def test_merges_other_rows_and_rejects_stale_candidate(self):
        first = save_decision(self.path, self.rows[0], "no", "Different target.")
        save_decision(self.path, self.rows[1], "yes", "Correct target.")
        self.assertEqual(load_review(self.path)[1][0], first)
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            save_decision(self.path, self.rows[0], "yes", "Stale edit")
        self.assertEqual(self.path.read_bytes(), saved)

    def test_uncertain_remains_pending_without_duplicating_marker(self):
        result = save_decision(self.path, self.rows[0], "uncertain",
                               "Human Review Needed: Still unclear.")
        self.assertTrue(needs_review(result))
        self.assertEqual(result["review_notes"], "human review needed: Still unclear.")
        self.assertEqual(reviewed_notes("", "yes"), "Human reviewed.")

    def test_duplicate_ids_rejected(self):
        self.rows[1]["connection_id"] = self.rows[0]["connection_id"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "unique"):
            load_review(self.path)

    def test_invalid_decision_does_not_write(self):
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Invalid match"):
            save_decision(self.path, self.rows[0], "maybe", "")
        self.assertEqual(self.path.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
