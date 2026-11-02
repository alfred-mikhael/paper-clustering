"""Metadata selection from real ZIP/Atom fixtures without network calls."""

from datetime import date, datetime
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
from zipfile import ZipFile

import requests

from paper_clustering.extract.extract_metadata import (
    MetadataExtractionError, get_metadata, parse_metadata_response, read_metadata_archive,
)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "metadata.zip"

    def record(self, paper_id="a", **changes):
        return {
            "id": paper_id, "title": " A\n technique ", "abstract": "Graph algorithms",
            "categories": "math.CO cs.DS", "authors_parsed": [["Doe", "Jane", ""]],
            "versions": [{"created": "Thu, 01 Jan 2026 00:00:00 GMT"},
                         {"created": "Sun, 01 Feb 2026 00:00:00 GMT"}],
            **changes,
        }

    def archive(self, *records):
        with ZipFile(self.path, "w") as archive:
            archive.writestr("data.jsonl", "\n".join(json.dumps(record) for record in records))

    def test_filters_combine_categories_keywords_and_original_publication_date(self):
        self.archive(self.record(), self.record("b", categories="physics.optics"),
                     self.record("c", abstract="Other subject", title="Other subject"))
        papers = read_metadata_archive(
            self.path, categories=[" CS.DS ", "math.AG"], keywords=["GRAPH", "absent"],
            start_date=date(2026, 1, 1), end_date=date(2026, 1, 1),
        )
        self.assertEqual([paper.arxiv_id for paper in papers], ["a"])
        self.assertEqual(papers[0].title, "A technique")
        self.assertEqual(papers[0].authors, ("Doe Jane",))
        self.assertEqual(papers[0].primary_category, "math.CO")
        self.assertEqual(read_metadata_archive(self.path, start_date=date(2026, 2, 1)), [])

    def test_limit_preserves_archive_order_and_blank_filters_are_unrestricted(self):
        self.archive(self.record("b"), self.record("a"))
        papers = read_metadata_archive(self.path, max_results=1, categories=[" "], keywords=[])
        self.assertEqual([paper.arxiv_id for paper in papers], ["b"])
        self.assertEqual(read_metadata_archive("nonexistent.zip", max_results=0), [])

    def test_malformed_json_and_incomplete_versions_are_skipped(self):
        with ZipFile(self.path, "w") as archive:
            archive.writestr("data.jsonl", "broken JSON\n\n" + "\n".join(map(json.dumps, (
                self.record("missing", versions=[]), self.record("good", authors_parsed=[], authors="A, B"),
            ))))
        papers = read_metadata_archive(self.path)
        self.assertEqual([paper.arxiv_id for paper in papers], ["good"])
        self.assertEqual(papers[0].authors, ("A", "B"))

    def test_invalid_bounds_and_ambiguous_archives_fail(self):
        for arguments, error in (
            ({"max_results": -1}, ValueError),
            ({"start_date": datetime(2026, 1, 1)}, TypeError),
            ({"end_date": "2026-01-01"}, TypeError),
            ({"start_date": date(2026, 2, 1), "end_date": date(2026, 1, 1)}, ValueError),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(error):
                read_metadata_archive(self.path, **arguments)
        for names in ([], ["one.json", "two.jsonl"]):
            with ZipFile(self.path, "w") as archive:
                for name in names:
                    archive.writestr(name, "")
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "exactly one"):
                read_metadata_archive(self.path)

    def test_http_failure_reports_identifier_and_status(self):
        session = Mock()
        response = session.get.return_value
        response.status_code = 503
        response.raise_for_status.side_effect = requests.HTTPError("unavailable")
        with self.assertRaisesRegex(MetadataExtractionError, "503.*paper-id"):
            get_metadata(session, "paper-id")
        self.assertEqual(session.get.call_args.kwargs["params"], {"id_list": "paper-id"})


@unittest.skipIf(importlib.util.find_spec("feedparser") is None, "feedparser is not installed")
class AtomTests(unittest.TestCase):
    XML = '''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <entry><id>https://arxiv.org/abs/math/1234567v2</id><title>A title</title>
    <summary>An abstract</summary><published>2026-01-02T12:30:00Z</published>
    <author><name>Jane Doe</name></author><arxiv:primary_category term="math.CO"/>
    <link rel="alternate" href="https://example.org/paper"/><arxiv:doi>10.1/example</arxiv:doi>
    </entry></feed>'''

    def test_atom_fields_versions_and_timezone(self):
        paper = parse_metadata_response(self.XML.encode())
        self.assertEqual(paper.arxiv_id, "math/1234567")
        self.assertEqual(paper.url, "https://example.org/paper")
        self.assertEqual(paper.authors, ("Jane Doe",))
        self.assertEqual(paper.publication_date.isoformat(), "2026-01-02T12:30:00+00:00")
        self.assertEqual(paper.doi, "10.1/example")

    def test_missing_required_fields_and_invalid_dates_are_rejected(self):
        for remove in ("<title>A title</title>", "<author><name>Jane Doe</name></author>",
                       '<arxiv:primary_category term="math.CO"/>', "<summary>An abstract</summary>"):
            with self.subTest(remove=remove), self.assertRaises(ValueError):
                parse_metadata_response(self.XML.replace(remove, "").encode())
        for document in ("<feed/>", self.XML.replace("2026-01-02T12:30:00Z", "invalid")):
            with self.assertRaises(ValueError):
                parse_metadata_response(document.encode())


if __name__ == "__main__":
    unittest.main()
