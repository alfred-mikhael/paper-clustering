"""Read filtered paper metadata from an arXiv snapshot or the Atom API."""

from datetime import date, datetime
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

import requests

from paper_clustering.data_models import PaperMetadata

ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_ARCHIVE_PATH = Path("data/math_cs_metadata.zip")


def read_metadata_archive(
    archive_path: str | Path = DEFAULT_ARCHIVE_PATH,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    categories: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    max_results: int | None = None,
) -> list[PaperMetadata]:
    """Return metadata from a filtered arXiv JSONL ZIP snapshot.

    The snapshot is read as a stream, so its multi-gigabyte JSONL member is
    never extracted to disk or loaded all at once. Date bounds are inclusive
    and apply to the first arXiv version (the original publication date).
    Categories and keywords use ``any`` matching: a paper passes when it has at
    least one selected category and at least one keyword in its title or
    abstract. Empty filters impose no restriction.
    """
    if max_results is not None and max_results < 0:
        raise ValueError("max_results must be non-negative or None")
    if start_date is not None and type(start_date) is not date:
        raise TypeError("start_date must be a date")
    if end_date is not None and type(end_date) is not date:
        raise TypeError("end_date must be a date")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if max_results == 0:
        return []

    selected_categories = frozenset(
        category.strip() for category in categories or () if category.strip()
    )
    selected_keywords = tuple(
        keyword.casefold().strip() for keyword in keywords or () if keyword.strip()
    )
    results: list[PaperMetadata] = []

    with zipfile.ZipFile(Path(archive_path)) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.endswith((".json", ".jsonl"))
        ]
        if len(members) != 1:
            raise ValueError(
                "archive must contain exactly one JSON or JSONL metadata member"
            )
        member = members[0]
        with archive.open(member) as records:
            for raw_record in records:
                if not raw_record.strip():
                    continue
                try:
                    record = json.loads(raw_record)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # The public snapshot has occasionally contained malformed
                    # records; one bad row should not abort a long scan.
                    continue

                record_categories = set(str(record["categories"]).split())
                if selected_categories and not record_categories.intersection(
                    selected_categories
                ):
                    continue
                searchable_text = f"{record['title']}\n{record['abstract']}".casefold()
                if selected_keywords and not any(
                    keyword in searchable_text for keyword in selected_keywords
                ):
                    continue

                if start_date or end_date:
                    versions = record.get("versions")
                    if not isinstance(versions, list) or not versions:
                        continue
                    first_version = versions[0]
                    created = (
                        first_version.get("created")
                        if isinstance(first_version, dict)
                        else None
                    )
                    publication_date = (
                        parsedate_to_datetime(str(created)) if created else None
                    )
                    if publication_date is None:
                        continue
                    publication_day = publication_date.date()
                    if start_date and publication_day < start_date:
                        continue
                    if end_date and publication_day > end_date:
                        continue

                try:
                    metadata = _record_to_metadata(record)
                except (TypeError, ValueError):
                    continue

                results.append(metadata)
                if max_results is not None and len(results) >= max_results:
                    break

    return results


def _record_to_metadata(record: dict[str, Any]) -> PaperMetadata:
    """Convert one Kaggle/arXiv snapshot record into the project data model."""
    arxiv_id = " ".join(str(record["id"]).split())
    versions = record.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError(f"arXiv record {arxiv_id} has no versions")
    created = versions[0].get("created") if isinstance(versions[0], dict) else None
    if not created:
        raise ValueError(f"arXiv record {arxiv_id} has no publication date")
    publication_date = parsedate_to_datetime(str(created))
    if publication_date is None:
        raise ValueError(f"arXiv record {arxiv_id} has an invalid publication date")

    categories = " ".join(str(record["categories"]).split()).split()
    authors = _record_authors(record)
    if not authors:
        raise ValueError(f"arXiv record {arxiv_id} has no authors")

    return PaperMetadata(
        authors=authors,
        publication_date=publication_date,
        arxiv_id=arxiv_id,
        title=" ".join(str(record["title"]).split()),
        abstract=" ".join(str(record["abstract"]).split()),
        primary_category=categories[0],
        url=f"https://arxiv.org/abs/{arxiv_id}",
        doi=" ".join(str(record.get("doi") or "").split()),
    )


def _record_authors(record: dict[str, Any]) -> tuple[str, ...]:
    parsed_authors = record.get("authors_parsed")
    if isinstance(parsed_authors, list):
        authors = tuple(
            " ".join(str(part).strip() for part in author[:2] if str(part).strip())
            for author in parsed_authors
            if isinstance(author, list)
        )
        authors = tuple(author for author in authors if author)
        if authors:
            return authors
    return tuple(
        author.strip()
        for author in str(record.get("authors", "")).split(",")
        if author.strip()
    )


def get_metadata(session: requests.Session, arxiv_id: str) -> PaperMetadata:
    """Fetch metadata for one arXiv identifier."""
    response = session.get(
        ARXIV_API_URL,
        params={"id_list": arxiv_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return parse_metadata_response(response.content)


def parse_metadata_response(response: bytes) -> PaperMetadata:
    """Parse the first paper in an arXiv Atom API response.

    Raises ``ValueError`` when the response is malformed or omits a field that
    is required by :class:`PaperMetadata`.
    """
    import feedparser

    feed = feedparser.parse(response)
    if not feed.entries:
        raise ValueError("arXiv metadata response contains no paper entry")
    entry = feed.entries[0]

    def required_text(key: str, field_name: str) -> str:
        value = " ".join(str(entry.get(key, "")).split())
        if not value:
            raise ValueError(f"arXiv paper entry has no {field_name}")
        return value

    identifier_url = required_text("id", "identifier")
    marker = "/abs/"
    arxiv_id = (
        identifier_url.partition(marker)[2]
        if marker in identifier_url
        else identifier_url.rsplit("/", 1)[-1]
    )
    if not arxiv_id:
        raise ValueError("arXiv paper entry has an invalid identifier")

    publication_text = required_text("published", "publication date")
    try:
        publication_date = datetime.fromisoformat(
            publication_text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"arXiv paper entry has an invalid publication date: {publication_text}"
        ) from exc

    authors = tuple(
        name
        for author in entry.get("authors", ())
        if (name := " ".join(str(author.get("name", "")).split()))
    )
    if not authors:
        raise ValueError("arXiv paper entry has no authors")

    primary_category = entry.get("arxiv_primary_category", {})
    category = str(primary_category.get("term", "")).strip()
    if not category:
        raise ValueError("arXiv paper entry has no primary category")

    alternate_link = next(
        (
            link.get("href", "").strip()
            for link in entry.get("links", ())
            if link.get("rel", "alternate") == "alternate" and link.get("href")
        ),
        identifier_url,
    )

    return PaperMetadata(
        authors=authors,
        publication_date=publication_date,
        arxiv_id=arxiv_id,
        title=required_text("title", "title"),
        abstract=required_text("summary", "abstract"),
        primary_category=category,
        url=alternate_link,
        doi=" ".join(str(entry.get("arxiv_doi", "")).split()),
    )
