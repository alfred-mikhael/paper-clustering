import time
from datetime import datetime, timedelta
import requests
import feedparser
from typing import Any, Optional
from .extract_intro import IntroExtractionError, get_intro_text, MathCoarseness
from .logger import IngestionLogger
from tqdm import tqdm
import logging

# base url for the arxiv api.
BASE_URL = "http://export.arxiv.org/api/query"

# Do not change these parameters. Arxiv specifically asks for a 3 second delay between API calls
ARXIV_MAX_PER_REQUEST = 1000
SLEEP_TIME = 3


class MetadataExtractionError(Exception):
    def __init__(self, arxiv_id: str, message: str):
        self.arxiv_id = arxiv_id
        self.message = message
        super().__init__(f"Failed to extract metadata from {arxiv_id}: {message}")


class ArxivFetchError(Exception):
    pass


def build_arxiv_query(categories: list[str], keywords: Optional[list[str]]) -> str:
    query = "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    if keywords:
        keywords = [f'abs:"{kw}"' for kw in keywords]
        query += f" AND (" + " OR ".join(keywords) + ")"
    return query


def download_from_arxiv(categories, start=0, keywords=None):
    """fetches Arxiv feed of papers in the specified `categories` up to
    `days_back` days ago, which contain any specified `keywords`.
    """

    query = build_arxiv_query(categories, keywords)
    params = {
        "search_query": query,
        "start": start,
        "max_results": ARXIV_MAX_PER_REQUEST,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        feed = feedparser.parse(response.text)
    except Exception as e:
        raise ArxivFetchError(str(e))

    if not feed.entries:
        raise ArxivFetchError("No response from Arxiv")

    return feed


def extract_entry_metadata(entry) -> dict[str, Any]:
    try:
        published = datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%SZ")
        return {
            "title": entry.title.strip(),
            "authors": [a.name for a in entry.authors],
            "published": published.isoformat(),
            "abstract": entry.summary.strip(),
            "id": entry.id.split("/abs/")[-1],
            "url": entry.link,
            "arxiv_category": entry.arxiv_primary_category["term"],
        }
    except Exception as e:
        raise MetadataExtractionError(getattr(entry, "link", ""), str(e))


def add_intro_text(
    session: requests.Session,
    paper: dict[str, Any],
    logger: IngestionLogger,
    coarseness: MathCoarseness = "coarse",
) -> bool:
    arxiv_id = paper.get("id", "")
    try:
        intro = get_intro_text(session, arxiv_id, coarseness)
        if not intro:
            raise IntroExtractionError(arxiv_id, "No introduction text found")
        paper["introduction"] = intro
        return True
    except Exception as e:
        logger.log_failure(
            e,
            "intro_extraction",
            {"arxiv_id": arxiv_id, "url": paper.get("url")},
        )
        return False


def fetch_arxiv_data(
    categories: list[str],
    max_results: int,
    logger: IngestionLogger,
    days_back: int = 7,
    keywords: Optional[list[str]] = None,
    coarseness: MathCoarseness = "coarse",
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    start = 0
    collected = 0
    skipped = 0
    hit_ratio = 1
    cutoff_date = datetime.now() - timedelta(days=days_back)
    collected_papers = []

    with requests.Session() as session:
        while collected < max_results:
            # EWMA of hit ratio to estimate how many batches are left. +1 to avoid division by 0
            hit_ratio = 0.8 * hit_ratio + 0.2 * (collected / (collected + skipped + 1))
            logging.info(
                f"Collected {collected} papers so far. Expect {hit_ratio * (max_results - collected) / ARXIV_MAX_PER_REQUEST} more batches."
            )
            try:
                feed = download_from_arxiv(categories, start, keywords)
            except ArxivFetchError as e:
                logger.log_failure(e, "arxiv_download")
                break

            for entry in tqdm(feed.entries, disable=not show_progress):
                try:
                    paper = extract_entry_metadata(entry)
                except MetadataExtractionError as e:
                    logger.log_failure(e, "metadata_extraction")
                    continue

                if datetime.fromisoformat(paper.get("published", "")) < cutoff_date:
                    return collected_papers

                # Ocassionally, a paper will be marked in many categories, but the primary
                # category will not be relevant (i.e. 'https://arxiv.org/abs/2607.06524v1')
                # we don't want to collect those papers, since they are unlikely to be relevant.
                if paper.get("arxiv_category") not in categories:
                    logger.log_skip(
                        paper["id"],
                        paper["title"],
                        paper["abstract"],
                        paper["published"],
                    )
                    skipped += 1
                    continue

                if add_intro_text(session, paper, logger, coarseness):
                    collected += 1
                    collected_papers.append(paper)
                    if collected >= max_results:
                        break
            start += ARXIV_MAX_PER_REQUEST
            time.sleep(SLEEP_TIME)

    return collected_papers


if __name__ == "__main__":
    from pprint import pp

    MAX_RESULTS = 100
    CATEGORIES = ["cs.DS", "cs.IT", "cs.CC", "math.CO"]
    DAYS_BACK = 7 * 365
    KEYWORDS = [
        "locally+decodable+code",
        "matrix+concentration",
        "coding+theory",
        "hypergraph",
    ]
    logger = IngestionLogger("../failure.jsonl", "../skipped_papers.txt")
    papers = fetch_arxiv_data(CATEGORIES, MAX_RESULTS, logger, DAYS_BACK, keywords=None)
    print(f"Collected {len(papers)} papers")

    # Example: print first paper
    pp(papers[0])
