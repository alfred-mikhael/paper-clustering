"""Extract binary regex features from paper paragraphs."""

import re
from typing import Iterable

from paper_clustering.data_models import ArxivSection

# Identify subsection titles whose prose deserves a small importance boost.
IMPORTANT_SECTION_TITLE_RE = re.compile(
    r"\b(?:results?|contributions?|techniques?|methods?|approach|"
    r"algorithms?|proof\s+(?:overview|outline|idea)|summary|overview|analysis)\b",
    re.IGNORECASE,
)

ENRICHMENT_PATTERNS = (
    # Direct first-person claims of a result.
    re.compile(
        r"\b(?:we|this (?:paper|work)|our work)\s+"
        r"(?:shows?|proves?|establishs?|demonstrates?|obtains?|derives?|gives?|"
        r"present|resolves?|settles?|improves?|characterizes?|achieves?|answers?|"
        r"confirms?|refutes?|bounds?|observes?|leverages?)\b",
        re.IGNORECASE,
    ),
    # Require content after a named result/contribution so an anaphoric phrase
    # such as "that is our main contribution" is not selected by itself.
    re.compile(
        r"\b(?:our|the) (?:main |principal |key |new )?"
        r"(?:results?|contributions?|proofs?)..+\b|\bmain theorem\b",
        re.IGNORECASE,
    ),
    # First-person descriptions of methods introduced or used by this paper.
    re.compile(
        r"\b(?:we|this (?:paper|work))\s+"
        r"(?:use|develop|introduce|design|construct|apply|combine|analy[sz]e|"
        r"reduce|exploit|invoke|establish|formulate)\b",
        re.IGNORECASE,
    ),
    # General method vocabulary is deliberately a weaker signal.
    re.compile(
        r"\b(?:techniques?|argument|characteri[sz](?:e|ation)|methodology|"
        r"methods?|approach|algorithms?|constructions?|formulation|idea|strategy|"
        r"impl[ies|y]|proof (?:idea|strategy|overview))\b",
        re.IGNORECASE,
    ),
    # Phrases that normally introduce the central technical ingredient.
    re.compile(
        r"\b(?:main|key|crucial|technical) (?:idea|ingredient|tool)\b|"
        r"\bour (?:proof|argument) (?:uses?|proceeds?|relies?)\b",
        re.IGNORECASE,
    ),
    # Definitions written in prose instead of a definition environment.
    re.compile(
        r"\b(?:we (?:define|call|say)|is defined as|means that|definition of|"
        r"is called .+ if|by .+ we mean|.+ is the|refer|denote)\b",
        re.IGNORECASE,
    ),
    # A catch-all category for useful words not covered by the other rules.
    re.compile(
        r"\b(?:rel(?:y|ies)|illustrates?|intuit(?:ion|ively|ive)|start with|"
        r"informal|inspired|we first|high(?:-|\s)?level)\b",
        re.IGNORECASE,
    ),
    # Weak connective phrases.
    re.compile(r"\b(?:using|via|based|building)\b", re.IGNORECASE),
)

# Match vocabulary that explicitly labels a statement as background or history.
HISTORICAL_CONTEXT_RE = re.compile(
    r"\b(?:previous(?:ly)?|prior (?:work|result)|earlier (?:work|result)|"
    r"best known|old (?:bound|result)|historical(?:ly)?|for context|history of|"
    r"discuss the history|their result|the result of which|recent(?:ly)?|introduced)\b",
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

# The feature-vector API deliberately exposes only aggregate regex evidence.
# Enrichment counts distinct positive patterns, while negative counts distinct
# exclusion rules that matched somewhere in the paragraph.
FEATURE_NAMES = (
    "important_section_title",
    "enrichment_pattern_hits",
    "negative_pattern_hits",
)

# Split at terminal punctuation followed by a likely sentence-starting token.
SENTENCE_BOUNDARY_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’]))\s+(?=[A-Z0-9])")


def label_paper(paper: Iterable[ArxivSection]) -> list[list[int]]:
    """Return one three-value regex-feature vector per paragraph in a paper."""
    return extract_paragraph_features(paper)


def extract_paragraph_features(
    sections: Iterable[ArxivSection],
) -> list[list[int]]:
    """Return aggregate regex features for each paragraph, in document order.

    Each output position is described by :data:`FEATURE_NAMES`. Enrichment
    counts the distinct positive patterns matched by non-excluded sentences.
    Negative evidence counts the distinct exclusion rules matched by any
    sentence. The section-header feature is binary and independent of the
    paragraph's sentence-level evidence.
    """
    paragraph_features: list[list[int]] = []
    for section in sections:
        important_section = bool(
            IMPORTANT_SECTION_TITLE_RE.search(section.section_header)
        )
        for paragraph in section.text:
            matched_enrichment_patterns = set()
            matched_negative_patterns = set()
            for sentence in _split_sentences(paragraph):
                sentence_negative_hits = _negative_pattern_hits(sentence)
                matched_negative_patterns.update(sentence_negative_hits)
                if sentence_negative_hits:
                    continue
                for index, pattern in enumerate(ENRICHMENT_PATTERNS):
                    if pattern.search(sentence):
                        matched_enrichment_patterns.add(index)

            paragraph_features.append(
                [
                    int(important_section),
                    len(matched_enrichment_patterns),
                    len(matched_negative_patterns),
                ]
            )

    return paragraph_features


def _split_sentences(text: str) -> list[str]:
    """Split prose after punctuation, including punctuation inside quotes."""
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_RE.split(text)
        if sentence.strip()
    ]


def _should_exclude_sentence(sentence: str) -> bool:
    """Return whether prose is historical, organizational, or uninformative."""
    return bool(_negative_pattern_hits(sentence))


def _negative_pattern_hits(sentence: str) -> set[str]:
    """Return the distinct exclusion rules matched by one sentence."""
    hits = set()
    if HISTORICAL_CONTEXT_RE.search(sentence):
        hits.add("historical_context")
    if PASSIVE_ATTRIBUTION_RE.search(sentence):
        hits.add("passive_attribution")
    if ATTRIBUTED_RESULT_RE.search(sentence):
        hits.add("attributed_result")
    if CITED_WORK_RE.search(sentence):
        hits.add("cited_work")
    if ANAPHORIC_RESULT_RE.search(sentence):
        hits.add("anaphoric_result")
    if PRIOR_CITATION_RE.search(sentence):
        hits.add("prior_citation")
    if IMPROVEMENT_RE.search(sentence) and MODAL_RE.search(sentence):
        hits.add("hypothetical_improvement")
    if IMPROVEMENT_RE.search(sentence) and not CURRENT_WORK_RE.search(sentence):
        hits.add("non_current_improvement")
    if SECTION_POINTER_RE.search(sentence):
        hits.add("section_pointer")
    if STATEMENT_LEAD_IN_RE.search(sentence):
        hits.add("statement_lead_in")
    if FUTURE_OVERVIEW_RE.search(sentence):
        hits.add("future_overview")
    if NOTATION_SETUP_RE.search(sentence):
        hits.add("notation_setup")
    if CITATION_LED_ORGANIZATION_RE.search(sentence):
        hits.add("citation_led_organization")
    if DOCUMENT_LOCATION_RE.search(sentence) and ORGANIZATION_VERB_RE.search(sentence):
        hits.add("document_organization")
    return hits


if __name__ == "__main__":
    from ..extract_text import get_paper
    import requests

    arxiv_id = "2608.20924v1"
    with requests.session() as session:
        paper = get_paper(session, arxiv_id)

    feature_vectors = label_paper(paper)
    texts = []
    for section in paper:
        texts.extend(section.text)

    print(FEATURE_NAMES)
    for features, text in zip(feature_vectors, texts):
        print(features, text)
