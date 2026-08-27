"""Download and parse paper metadata from the arXiv Atom API."""

from datetime import datetime

import feedparser
import requests

from paper_clustering.data_models import PaperMetadata

ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_TIMEOUT_SECONDS = 60


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
    )
