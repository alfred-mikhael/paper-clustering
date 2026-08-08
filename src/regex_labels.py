"""Select important result, technique, and theorem text from introductions."""

import re
import tarfile
import zipfile
from typing import Iterable

import requests

from .extract_intro import (
    SECTION_RE,
    build_intro_tex_source_from_bytes,
    clean_latex_for_embedding,
    download_source,
    extract_intro_from_latex,
)

# Identify subsection titles whose prose deserves a small importance boost.
IMPORTANT_SECTION_TITLE_RE = re.compile(
    r"\b(?:results?|contributions?|techniques?|methods?|approach|"
    r"algorithms?|proof\s+(?:overview|outline|idea)|summary|overview)\b",
    re.IGNORECASE,
)

# Match complete theorem-like environments. The environment name may have a
# custom prefix, but must end in a standard theorem or definition suffix.
STATEMENT_ENV_RE = re.compile(
    r"""
    \\begin\s*\{(?P<env>
        [a-zA-Z@]*(?:theorem|thm|lemma|proposition|prop|corollary|cor|conjecture|
        claim|definition|defn|def)
    )\*?\}
    .*?
    \\end\s*\{(?P=env)\*?\}
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

ENRICHMENT_PATTERNS = (
    # Direct first-person claims of a result receive the strongest score.
    (
        re.compile(
            r"\b(?:we|this (?:paper|work)|our work)\s+"
            r"(?:shows?|proves?|establishs?|demonstrates?|obtains?|derives?|gives?|"
            r"present|resolves?|settles?|improves?|characterizes?|achieves?|answers?|"
            r"confirms?|refutes?|bounds?)\b",
            re.IGNORECASE,
        ),
        5,
    ),
    # Require content after a named result/contribution so an anaphoric phrase
    # such as "that is our main contribution" is not selected by itself.
    (
        re.compile(
            r"\b(?:our|the) (?:main |principal |key |new )?"
            r"(?:results?|contributions?|proofs?)..+\b|\bmain theorem\b",
            re.IGNORECASE,
        ),
        4,
    ),
    # First-person descriptions of methods introduced or used by this paper.
    (
        re.compile(
            r"\b(?:we|this (?:paper|work))\s+"
            r"(?:use|develop|introduce|design|construct|apply|combine|analy[sz]e|"
            r"reduce|exploit|invoke|establish|formulate)\b",
            re.IGNORECASE,
        ),
        4,
    ),
    # General method vocabulary is deliberately a weaker signal.
    (
        re.compile(
            r"\b(?:techniques?|argument|characteri[sz](?:e|ation)|methodology|"
            r"methods?|approach|algorithms?|constructions?|formulation|idea|strategy|"
            r"impl[ies|y]|proof (?:idea|strategy|overview))\b",
            re.IGNORECASE,
        ),
        2,
    ),
    # Phrases that normally introduce the central technical ingredient.
    (
        re.compile(
            r"\b(?:main|key|crucial|technical) (?:idea|ingredient|tool)\b|"
            r"\bour (?:proof|argument) (?:uses?|proceeds?|relies?)\b",
            re.IGNORECASE,
        ),
        4,
    ),
    # Definitions written in prose instead of a definition environment.
    (
        re.compile(
            r"\b(?:we (?:define|call|say)|is defined as|means that|definition of|"
            r"is called .+ if|by .+ we mean|.+ is the|refer|denote)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    # Weak connective phrases matter only alongside another signal.
    (re.compile(r"\b(?:using|via|based|building)\b", re.IGNORECASE), 2),
)

MAX_ENRICHMENT_SENTENCES = 12

# Match vocabulary that explicitly labels a statement as background or history.
HISTORICAL_CONTEXT_RE = re.compile(
    r"\b(?:previous(?:ly)?|prior (?:work|result)|earlier (?:work|result)|"
    r"best known|old (?:bound|result)|historical(?:ly)?|for context|history of|"
    r"discuss the history|their result|the result of which)\b",
    re.IGNORECASE,
)

# Match passive constructions that attribute a contribution to earlier work.
PASSIVE_ATTRIBUTION_RE = re.compile(
    r"\b(?:was|were|has been|have been)?\s*"
    r"(?:proved|shown|obtained|introduced|improved|reproved|established|developed|"
    r"proposed)\s+by\b",
    re.IGNORECASE,
)

# Match named authors, an optional citation, and a result verb.
ATTRIBUTED_RESULT_RE = re.compile(
    r"""
    \b(?!(?:We|This|It)\b)[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*
    (?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+\[[^\]]+\])?\s+
    (?:
        proved|showed|obtain(?:ed)?|gave|introduced|improved|reproved|established|
        developed|constructed|were\s+(?:the\s+)?first|was\s+(?:the\s+)?first
    )\b
    """,
    re.VERBOSE,
)

# Match a citation acting as the grammatical subject of an earlier-work claim.
CITED_WORK_RE = re.compile(
    r"(?:^|\b(?:on the other hand|in contrast)\s*,?\s*)\[[^\]]+\]\s+"
    r"(?:prove|show|obtain|give|introduce|develop|construct|consider|analy[sz]e)",
    re.IGNORECASE,
)

SECTION_POINTER_RE = re.compile(
    r"^(?:first|next|finally)?\s*,?\s*(?:in|see)\s+"
    r"(?:section|subsection|appendix)\b",
    re.IGNORECASE,
)

# Match theorem/result lead-ins whose actual content follows separately.
STATEMENT_LEAD_IN_RE = re.compile(
    r"\b(?:result|theorem|bound|corollary|improvement)\b.*\b(?:the following|below)\b|"
    r"\b(?:the following|below)\b.*\b(?:result|theorem|bound|corollary|improvement)\b",
    re.IGNORECASE,
)

FUTURE_OVERVIEW_RE = re.compile(
    r"\bwe\s+(?:will|shall)\s+(?:[a-z]+\s+){0,3}"
    r"(?:present|give|describe|outline|state)\b.*"
    r"\b(?:proof|overview|argument|theorem|result|ideas?)\b",
    re.IGNORECASE,
)

# A sentence is organizational only when it contains both patterns.
DOCUMENT_LOCATION_RE = re.compile(r"\b(?:section|subsection|appendix)\b", re.I)
ORGANIZATION_VERB_RE = re.compile(
    r"\b(?:contains?|present|proofs?|split|introduce|organization)\b", re.I
)

NOTATION_SETUP_RE = re.compile(
    r"\b(?:standard\b.*\bnotation|we\s+(?:write|denote|use)\b.*"
    r"(?:notation|to denote|denoted by))\b",
    re.IGNORECASE,
)

CITATION_LED_ORGANIZATION_RE = re.compile(
    r"^(?:finally\s*,?\s*)?in\s+(?:\[[^]]+\])?\s*,?\s*we\s+"
    r"(?:present|prove|define|introduce|discuss|describe|show|establish)\b",
    re.IGNORECASE,
)

# Reject claims whose subject has no meaning outside its original paragraph.
ANAPHORIC_RESULT_RE = re.compile(
    r"""
    ^(?:our|the)\s+(?:main\s+|principal\s+|key\s+)?(?:results?|theorem)\s+
    (?:is|shows?|proves?|establishes?)\s+(?:that\s+)?
    (?:it|they|this|these|those)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

PRIOR_CITATION_RE = re.compile(
    r"\b(?:construction|result|bound|method|argument)\s+in\s+\[[^]]+\]",
    re.IGNORECASE,
)

IMPROVEMENT_RE = re.compile(r"\bimprovements?\b", re.IGNORECASE)
MODAL_RE = re.compile(r"\b(?:would|could|might|may)\b", re.IGNORECASE)
CURRENT_WORK_RE = re.compile(
    r"\b(?:we|our|this (?:paper|work)|present)\b", re.IGNORECASE
)

THEOREM_TITLE_CITATION_RE = re.compile(r"\\cite\w*\b")
THEOREM_TITLE_HISTORY_RE = re.compile(
    r"\b(?:known|classical|previous|earlier|due to)\b", re.IGNORECASE
)

# Remove outer theorem markup and labels while retaining the statement body.
STATEMENT_MARKUP_RE = re.compile(
    r"^\s*\\begin\s*\{[^{}]+\}|\\end\s*\{[^{}]+\}\s*$|"
    r"\\label\s*\{[^{}]*\}",
    re.IGNORECASE,
)

# Split at terminal punctuation followed by a likely sentence-starting token.
SENTENCE_BOUNDARY_RE = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’]))\s+(?=[A-Z0-9])"
)


def get_enrichment_text(session: requests.Session, arxiv_id: str) -> str:
    """Download an introduction and return its important formatted excerpts."""
    try:
        source_bytes = download_source(session, arxiv_id)
        source = build_intro_tex_source_from_bytes(source_bytes, arxiv_id)
        if not source:
            return ""
        introduction = extract_intro_from_latex(source)
        return _extract_enrichment_excerpts(introduction) if introduction else ""
    except (
        requests.RequestException,
        tarfile.TarError,
        zipfile.BadZipFile,
        OSError,
        UnicodeError,
    ):
        return ""


def _extract_enrichment_excerpts(introduction: str) -> str:
    """Select current contributions and complete original formal statements."""
    all_statements = list(STATEMENT_ENV_RE.finditer(introduction))
    statements = [
        statement
        for statement in all_statements
        if not _is_historical_statement(statement)
    ]

    # Blanking keeps source offsets meaningful and prevents theorem text from
    # being selected a second time as ordinary prose.
    prose = list(introduction)
    for statement in all_statements:
        prose[statement.start() : statement.end()] = " " * (
            statement.end() - statement.start()
        )
    prose_text = "".join(prose)

    candidates: list[tuple[int, int, str]] = []
    sections = list(SECTION_RE.finditer(prose_text))
    regions: list[tuple[int, int, str]] = []
    if sections:
        regions.append((0, sections[0].start(), ""))
        for index, section in enumerate(sections):
            end = (
                sections[index + 1].start()
                if index + 1 < len(sections)
                else len(prose_text)
            )
            regions.append((section.end(), end, section.group("title")))
    else:
        regions.append((0, len(prose_text), ""))

    for start, end, title in regions:
        cleaned = clean_latex_for_embedding(prose_text[start:end])
        section_bonus = 2 if IMPORTANT_SECTION_TITLE_RE.search(title) else 0
        for sentence_index, sentence in enumerate(_split_sentences(cleaned)):
            if _should_exclude_sentence(sentence):
                print(sentence, _enrichment_sentence_score(sentence))
                continue
            score = section_bonus + _enrichment_sentence_score(sentence)
            print(sentence, score)
            if score >= 3:
                candidates.append((score, start + sentence_index, sentence))

    # Rank first to enforce a useful size bound, then restore document order.
    selected = sorted(
        sorted(candidates, key=lambda item: (-item[0], item[1]))[
            :MAX_ENRICHMENT_SENTENCES
        ],
        key=lambda item: item[1],
    )

    techniques = _dedupe_preserving_order(text for _, _, text in selected)
    theorems = _dedupe_preserving_order(
        _strip_statement_environment(match.group(0)) for match in statements
    )
    return (
        "Techniques:\n"
        + "\n\n".join(techniques)
        + "\n\nTheorems:\n"
        + "\n\n".join(theorems)
    )


def _strip_statement_environment(statement: str) -> str:
    """Remove the outer environment and internal label metadata."""
    return STATEMENT_MARKUP_RE.sub("", statement).strip()


def _split_sentences(text: str) -> list[str]:
    """Split prose after punctuation, including punctuation inside quotes."""
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_RE.split(text)
        if sentence.strip()
    ]


def _enrichment_sentence_score(sentence: str) -> int:
    """Return the sum of result, technique, and definition signals."""
    return sum(
        weight for pattern, weight in ENRICHMENT_PATTERNS if pattern.search(sentence)
    )


def _should_exclude_sentence(sentence: str) -> bool:
    """Return whether prose is historical, organizational, or uninformative."""
    if HISTORICAL_CONTEXT_RE.search(sentence):
        return True
    if PASSIVE_ATTRIBUTION_RE.search(sentence):
        return True
    if ATTRIBUTED_RESULT_RE.search(sentence) or CITED_WORK_RE.search(sentence):
        return True
    if ANAPHORIC_RESULT_RE.search(sentence):
        return True
    if PRIOR_CITATION_RE.search(sentence):
        return True
    if IMPROVEMENT_RE.search(sentence) and MODAL_RE.search(sentence):
        return True
    if IMPROVEMENT_RE.search(sentence) and not CURRENT_WORK_RE.search(sentence):
        return True
    if SECTION_POINTER_RE.search(sentence) or STATEMENT_LEAD_IN_RE.search(sentence):
        return True
    if FUTURE_OVERVIEW_RE.search(sentence) or NOTATION_SETUP_RE.search(sentence):
        return True
    if CITATION_LED_ORGANIZATION_RE.search(sentence):
        return True
    return bool(
        DOCUMENT_LOCATION_RE.search(sentence) and ORGANIZATION_VERB_RE.search(sentence)
    )


def _is_historical_statement(statement: re.Match[str]) -> bool:
    """Identify theorem environments explicitly attributed to earlier work."""
    environment = statement.group("env").lower()
    if environment == "def" or environment.endswith(("definition", "defn")):
        return False

    opening = statement.group(0).split("}", 1)[1].lstrip()
    if not opening.startswith("["):
        return False
    title = opening[1 : opening.find("]")]
    return bool(
        THEOREM_TITLE_CITATION_RE.search(title)
        or THEOREM_TITLE_HISTORY_RE.search(title)
    )


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """Return unique strings in their first-occurrence order."""
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
