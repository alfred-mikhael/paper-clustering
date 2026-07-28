"""Download arXiv sources and extract introductions and enrichment text."""

import gzip
import io
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Optional

import requests

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
REQUEST_TIMEOUT_SECONDS = 60
MAX_INCLUDE_DEPTH = 8
TEX_SOURCE_SUFFIXES = {".tex", ".ltx"}


class IntroExtractionError(Exception):
    def __init__(self, arxiv_id: str, message: str):
        self.arxiv_id = arxiv_id
        self.message = message
        super().__init__(f"Failed to extract introduction from {arxiv_id}: {message}")


@dataclass(frozen=True)
class TexSource:
    name: str
    text: str


@dataclass(frozen=True)
class TexSection:
    kind: str
    title: str
    text: str


# Match a LaTeX section command, its level, and its brace-delimited title.
# One nested pair of braces is supported in titles; arbitrary nesting is not.
SECTION_RE = re.compile(
    r"""
    \\(?P<kind>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)
    \*?
    (?:\s*\[[^\]]*\])?
    \s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Recognize common names for an introduction section.
INTRO_TITLE_RE = re.compile(
    r"\b(?:introduction|intro|overview|background)\b", re.IGNORECASE
)

# Identify subsection titles whose prose deserves a small importance boost.
IMPORTANT_SECTION_TITLE_RE = re.compile(
    r"\b(?:results?|contributions?|techniques?|methods?|approach|"
    r"algorithms?|proof\s+(?:overview|outline|idea)|summary|overview)\b",
    re.IGNORECASE,
)

# Match complete theorem-like environments.  The environment name may have a
# custom prefix (for example, ``mtheorem`` or ``mainlemma``), but must end in a
# standard theorem/definition suffix.  The backreference requires the closing
# environment name to equal the opening name.
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
            r"(?:show|prove|establish|demonstrate|obtain|derive|give|present|"
            r"resolve|settle|improve|characterize|achieve|answer|confirm|refute|bound)\b",
            re.IGNORECASE,
        ),
        5,
    ),
    # Explicit names for a paper's result, contribution, proof, or theorem.
    (
        re.compile(
            r"\b(?:our|the) (?:main |principal |key |new )?"
            r"(?:results?|contributions?|proofs?)\b|\bmain theorem\b",
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
    # General method vocabulary; this is deliberately a weaker signal.
    (
        re.compile(
            r"\b(?:techniques?|argument|characteri[sz](?:e|ation)|methodology|methods?|approach|algorithms?|"
            r"constructions?|formulation|idea|strategy|proof (?:idea|strategy|overview))\b",
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
            r"\b(?:we (?:define|call|say)|is defined as|means that|"
            r"definition of|is called .+ if|by .+ we mean)\b",
            re.IGNORECASE,
        ),
        5,
    ),
    # Weak connective phrases that matter only alongside another signal.
    (
        re.compile(r"\b(?:using|via|based|building)\b", re.IGNORECASE),
        2,
    ),
)

MAX_ENRICHMENT_SENTENCES = 12

# Match vocabulary that explicitly labels a statement as background or history.
HISTORICAL_CONTEXT_RE = re.compile(
    r"\b(?:previous(?:ly)?|prior (?:work|result)|earlier (?:work|result)|"
    r"best known|old (?:bound|result)|historical(?:ly)?|for context|history of|"
    r"discuss the history|their result|the result of which)\b",
    re.IGNORECASE,
)

# Match passive constructions that attribute a contribution to earlier work,
# such as "was proved by" or "improved by".
PASSIVE_ATTRIBUTION_RE = re.compile(
    r"\b(?:was|were|has been|have been)?\s*"
    r"(?:proved|shown|obtained|introduced|improved|reproved|established|developed|proposed)"
    r"\s+by\b",
    re.IGNORECASE,
)

# Match one or more capitalized author surnames, an optional citation, and a
# result verb.  Pronouns are excluded so "We showed" is not an attribution.
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

# Match sentences whose grammatical subject is a citation to earlier work,
# optionally after a contrast transition such as "on the other hand".
CITED_WORK_RE = re.compile(
    r"(?:^|\b(?:on the other hand|in contrast)\s*,?\s*)\[[^\]]+\]\s+"
    r"(?:prove|show|obtain|give|introduce|develop|construct|consider|analy[sz]e)",
    re.IGNORECASE,
)

# Match a sentence that merely points the reader to another section.
SECTION_POINTER_RE = re.compile(
    r"^(?:first|next|finally)?\s*,?\s*(?:in|see)\s+"
    r"(?:section|subsection|appendix)\b",
    re.IGNORECASE,
)

# Match theorem/result lead-ins whose actual mathematical content follows in a
# separate environment, including "the following theorem" at either end.
STATEMENT_LEAD_IN_RE = re.compile(
    r"\b(?:result|theorem|bound|corollary|improvement)\b.*"
    r"\b(?:the following|below)\b|"
    r"\b(?:the following|below)\b.*"
    r"\b(?:result|theorem|bound|corollary|improvement)\b",
    re.IGNORECASE,
)

# Match future-tense meta prose announcing an upcoming proof or overview.
FUTURE_OVERVIEW_RE = re.compile(
    r"\bwe\s+(?:will|shall)\s+(?:[a-z]+\s+){0,3}"
    r"(?:present|give|describe|outline|state)\b.*"
    r"\b(?:proof|overview|argument|theorem|result|ideas?)\b",
    re.IGNORECASE,
)

# These two small patterns are used together: a sentence is organizational only
# when it contains both a document location and an organization verb.
DOCUMENT_LOCATION_RE = re.compile(r"\b(?:section|subsection|appendix)\b", re.I)
ORGANIZATION_VERB_RE = re.compile(
    r"\b(?:contains?|present|proofs?|split|introduce|organization)\b", re.I
)

# Match conventional notation setup rather than substantive definitions.
NOTATION_SETUP_RE = re.compile(
    r"\b(?:standard\b.*\bnotation|we\s+(?:write|denote|use)\b.*"
    r"(?:notation|to denote|denoted by))\b",
    re.IGNORECASE,
)

# Match organization sentences whose section reference has already been
# normalized to a bracketed key, e.g. "In [sec:proof], we present ...".
CITATION_LED_ORGANIZATION_RE = re.compile(
    r"^(?:finally\s*,?\s*)?in\s+(?:\[[^]]+\])?\s*,?\s*we\s+"
    r"(?:present|prove|define|introduce|discuss|describe|show|establish)\b",
    re.IGNORECASE,
)

# Match a result claim whose subject is an unresolved bare pronoun.  Such a
# sentence is not meaningful after extraction from its surrounding paragraph.
ANAPHORIC_RESULT_RE = re.compile(
    r"""
    ^(?:our|the)\s+(?:main\s+|principal\s+|key\s+)?(?:results?|theorem)\s+
    (?:is|shows?|proves?|establishes?)\s+(?:that\s+)?
    (?:it|they|this|these|those)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Match a named result or construction followed by a bracketed citation.  This
# is the normalized form of prose such as "the construction in \cite{...}".
PRIOR_CITATION_RE = re.compile(
    r"\b(?:construction|result|bound|method|argument)\s+in\s+\[[^]]+\]",
    re.IGNORECASE,
)

# Match singular or plural "improvement", and modal verbs used in speculative
# statements.  They are separate because the caller also checks current-work
# ownership before rejecting a concrete improvement claim.
IMPROVEMENT_RE = re.compile(r"\bimprovements?\b", re.IGNORECASE)
MODAL_RE = re.compile(r"\b(?:would|could|might|may)\b", re.IGNORECASE)
CURRENT_WORK_RE = re.compile(
    r"\b(?:we|our|this (?:paper|work)|present)\b", re.IGNORECASE
)

# Match bibliography commands in optional theorem titles, such as
# ``[Smith~\cite{smith}]``.
THEOREM_TITLE_CITATION_RE = re.compile(r"\\cite\w*\b")

# Match words that explicitly mark an optional theorem title as prior work.
THEOREM_TITLE_HISTORY_RE = re.compile(
    r"\b(?:known|classical|previous|earlier|due to)\b", re.IGNORECASE
)

# Remove an unescaped percent sign and everything after it on the same line.
COMMENT_RE = re.compile(r"(?<!\\)%.*")

# Match a simple environment whose body contains no nested environment of the
# same name.  DOTALL allows the body to span lines.
GENERIC_ENV_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}(.*?)\\end\{\1\}", re.DOTALL)

# Match a command with one non-nested brace argument and optional ``*``/``[]``.
# This is a fallback after commands needing balanced parsing have been handled.
COMMAND_WITH_BRACE_ARG_RE = re.compile(
    r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?\s*\{([^{}]*)\}"
)

# Match markup removed from a retained formal statement: either outer
# environment command at a string boundary, or a label command anywhere.
STATEMENT_MARKUP_RE = re.compile(
    r"^\s*\\begin\s*\{[^{}]+\}|\\end\s*\{[^{}]+\}\s*$|" r"\\label\s*\{[^{}]*\}",
    re.IGNORECASE,
)

# Split only when terminal punctuation (possibly followed by a closing quote)
# is followed by whitespace and a likely sentence-starting capital/digit.
SENTENCE_BOUNDARY_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’]))\s+(?=[A-Z0-9])")

# Match the body of a standard abstract environment as an introduction fallback.
ABSTRACT_RE = re.compile(
    r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
    re.IGNORECASE | re.DOTALL,
)

# Match \input{path} and \include{path}; the named group is resolved against
# the in-memory source archive by ``_expand_inputs``.
INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{(?P<path>[^{}]+)\}", re.IGNORECASE)

# Remove environments whose contents cannot usefully become embedding prose.
VERBATIM_ENV_RE = re.compile(
    r"\\begin\{(?:verbatim|lstlisting|minted)\}.*?"
    r"\\end\{(?:verbatim|lstlisting|minted)\}",
    re.IGNORECASE | re.DOTALL,
)

# Remove \newcommand-style declarations.  The final group supports one level
# of nested braces in the replacement body, which covers typical preambles.
NEWCOMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\s*"
    r"\{?\\[a-zA-Z@]+\}?"
    r"(?:\s*\[[^]]*\]){0,2}\s*"
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    re.DOTALL,
)

# Remove primitive \def-style declarations with optional numbered parameters
# and one level of nested braces in the definition body.
DEF_RE = re.compile(
    r"\\(?:def|gdef|edef|xdef)\s*\\[a-zA-Z@]+\s*"
    r"(?:#[0-9]\s*)*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    re.DOTALL,
)

# Remove common operator/delimiter declarations with a command-name argument
# followed by a simple brace-delimited definition.
DECLARE_COMMAND_RE = re.compile(
    r"\\(?:DeclareMathOperator|DeclarePairedDelimiter)\*?\s*"
    r"\{?\\[a-zA-Z@]+\}?\s*\{[^{}]*\}"
)

CITATION_COMMANDS = {
    "cite",
    "citealp",
    "citeauthor",
    "citep",
    "citet",
    "citeyear",
}

REFERENCE_COMMANDS = {"cref", "Cref", "eqref", "pageref", "ref", "subref"}

DROP_COMMANDS = {
    "addcontentsline",
    "bibliographystyle",
    "bibliography",
    "caption",
    "footnote",
    "footnotemark",
    "footnotetext",
    "index",
    "label",
    "thanks",
}

KEEP_CONTENT_COMMANDS = {
    "em",
    "emph",
    "mathbf",
    "mathcal",
    "mathfrak",
    "mathit",
    "mathrm",
    "mathsf",
    "mathtt",
    "mbox",
    "paragraph",
    "section",
    "subparagraph",
    "subsection",
    "subsubsection",
    "text",
    "textbf",
    "textit",
    "textnormal",
    "textrm",
    "textsc",
    "textsf",
    "texttt",
    "underline",
}

TEXTUAL_ENVS = {
    "abstract",
    "center",
    "description",
    "enumerate",
    "itemize",
    "quote",
    "quotation",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "conjecture",
    "remark",
    "example",
    "proof",
}


def download_source(session: requests.Session, arxiv_id: str) -> bytes:
    """Download the source archive for ``arxiv_id`` using the supplied session."""
    response = session.get(
        ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.content


def get_intro_text(session: requests.Session, arxiv_id: str) -> str:
    """Download and clean the introduction of an arXiv paper."""
    try:
        source_bytes = download_source(session, arxiv_id)
        source = build_intro_tex_source_from_bytes(source_bytes, arxiv_id)
        if not source:
            return ""
        intro = extract_intro_from_latex(source)
        return clean_latex_for_embedding(intro)
    except (
        requests.RequestException,
        tarfile.TarError,
        zipfile.BadZipFile,
        OSError,
        UnicodeError,
    ):
        return ""


def get_enrichment_text(session: requests.Session, arxiv_id: str) -> str:
    """Return formatted important excerpts from an arXiv introduction.

    Complete theorem-like and definition environments are always retained.  The
    remaining prose is ranked using signals for results, contributions,
    techniques, and definitions. Output is grouped under ``Techniques:`` and
    ``Theorems:`` headings.
    """
    try:
        source_bytes = download_source(session, arxiv_id)
        source = build_intro_tex_source_from_bytes(source_bytes, arxiv_id)
        if not source:
            return ""
        introduction = extract_intro_from_latex(source)
        if not introduction:
            return ""
        return _extract_enrichment_excerpts(introduction)
    except (
        requests.RequestException,
        tarfile.TarError,
        zipfile.BadZipFile,
        OSError,
        UnicodeError,
    ):
        return ""


def _extract_enrichment_excerpts(introduction: str) -> str:
    """Select current contributions and complete, original formal statements.

    Formal statements are handled separately so that their LaTeX remains
    verbatim. Prose is cleaned, split into sentences, filtered for historical
    or low-information material, and ranked. The highest-ranked prose excerpts
    and retained statements are returned in source order within their respective
    output sections.
    """
    all_statements = list(STATEMENT_ENV_RE.finditer(introduction))
    statements = [
        statement
        for statement in all_statements
        if not _is_historical_statement(statement)
    ]

    # Blanking instead of deleting keeps source offsets meaningful and prevents
    # theorem text from being selected a second time as ordinary prose.
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
            print(sentence)
            if _should_exclude_sentence(sentence):
                continue
            score = section_bonus + _enrichment_sentence_score(sentence)
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
    sentences = SENTENCE_BOUNDARY_RE.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _enrichment_sentence_score(sentence: str) -> int:
    """Return the sum of general result, technique, and definition signals."""
    return sum(
        weight for pattern, weight in ENRICHMENT_PATTERNS if pattern.search(sentence)
    )


def _should_exclude_sentence(sentence: str) -> bool:
    """Return whether prose is historical, organizational, or uninformative.

    These checks deliberately use general linguistic signals rather than
    topic-specific terms from any sample paper.
    """
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


def iter_tex_sources(
    source_bytes: bytes, arxiv_id: str = "source"
) -> Iterable[TexSource]:
    """Yield TeX files from tar, zip, gzip, or plain-text arXiv source data.

    Archives are read in memory and are never extracted to the filesystem.
    """
    buffer = io.BytesIO(source_bytes)

    if tarfile.is_tarfile(io.BytesIO(source_bytes)):
        with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not _is_latex_file(member.name):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                yield TexSource(member.name, decode_source(extracted.read()))
        return

    if zipfile.is_zipfile(buffer):
        with zipfile.ZipFile(buffer) as archive:
            for name in archive.namelist():
                if _is_latex_file(name):
                    yield TexSource(name, decode_source(archive.read(name)))
        return

    if source_bytes.startswith(b"\x1f\x8b"):
        decompressed = gzip.decompress(source_bytes)
        yield from iter_tex_sources(decompressed, arxiv_id)
        return

    yield TexSource(f"{arxiv_id}.tex", decode_source(source_bytes))


def decode_source(raw: bytes) -> str:
    """Decode TeX source as UTF-8, falling back to arXiv's common Latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def build_intro_tex_source(sources: list[TexSource]) -> str:
    """Select the most likely main TeX file and expand its includes."""
    source_map = {_normalize_path(source.name): source.text for source in sources}
    if not source_map:
        return ""
    main_name = max(
        source_map, key=lambda name: _main_file_score(name, source_map[name])
    )
    return _expand_inputs(main_name, source_map)


def build_intro_tex_source_from_bytes(source_bytes: bytes, arxiv_id: str) -> str:
    """Read an arXiv source payload and assemble its likely main document."""
    return build_intro_tex_source(list(iter_tex_sources(source_bytes, arxiv_id)))


def extract_intro_from_latex(source_text: str) -> str:
    """Extract the introduction section, with preface/abstract fallbacks.

    A matched introduction ends at the next top-level section rather than at a
    subsection, allowing result and technique subsections to remain available
    to enrichment extraction.
    """
    document = _document_body(strip_latex_comments(source_text))
    sections = list(SECTION_RE.finditer(document))

    for index, match in enumerate(sections):
        if not INTRO_TITLE_RE.search(match.group("title")):
            continue

        end = next(
            (
                section.start()
                for section in sections[index + 1 :]
                if section.group("kind").lower() in {"part", "chapter", "section"}
            ),
            len(document),
        )
        return document[match.end() : end]

    if sections:
        first_section_start = sections[0].start()
        preface = document[:first_section_start]
        if len(preface.split()) >= 100:
            return preface

    abstract_match = ABSTRACT_RE.search(document)
    return abstract_match.group(1) if abstract_match else ""


def extract_sections_from_latex(source_text: str) -> list[TexSection]:
    """Return every sectioning command and its contents in document order.

    Each section ends at the next sectioning command, so returned text does not
    overlap. The section kind distinguishes parts, chapters, sections,
    subsections, paragraphs, and their supported variants.
    """
    document = _document_body(strip_latex_comments(source_text))
    matches = list(SECTION_RE.finditer(document))
    return [
        TexSection(
            kind=match.group("kind").lower(),
            title=match.group("title").strip(),
            text=document[
                match.end() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(document)
                )
            ].strip(),
        )
        for index, match in enumerate(matches)
    ]


def clean_latex_for_embedding(text: str) -> str:
    """Remove LaTeX markup and normalize prose for embedding."""
    text = strip_latex_comments(text)
    text = _drop_latex_definitions(text)
    text = _drop_verbatim_blocks(text)
    text = _normalize_citation_commands(text)
    text = _drop_commands_with_arguments(text, DROP_COMMANDS)
    text = _preserve_href_text(text)
    text = _unwrap_environments(text)
    text = _unwrap_commands(text, KEEP_CONTENT_COMMANDS)
    text = _decode_latex_escapes(text)
    text = _drop_remaining_commands(text)
    text = _normalize_embedding_whitespace(text)
    return text


def strip_latex_comments(text: str) -> str:
    return "\n".join(COMMENT_RE.sub("", line) for line in text.splitlines())


def _is_latex_file(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in TEX_SOURCE_SUFFIXES


def _normalize_path(path: str) -> str:
    return str(PurePosixPath(path.replace("\\", "/")))


def _main_file_score(name: str, text: str) -> tuple[int, int]:
    lower_name = PurePosixPath(name).name.lower()
    score = 0
    if "\\documentclass" in text:
        score += 100
    if "\\begin{document}" in text:
        score += 100
    if lower_name in {"main.tex", "paper.tex", "article.tex", "ms.tex"}:
        score += 10
    if SECTION_RE.search(text):
        score += 20
    if "\\input" in text:
        score += 15
    return score, len(text)


def _expand_inputs(name: str, sources: dict[str, str], depth: int = 0) -> str:
    """Recursively inline TeX input/include commands up to a safe depth."""
    text = sources.get(name, "")
    if depth >= MAX_INCLUDE_DEPTH:
        return text

    base = PurePosixPath(name).parent

    def replace(match: re.Match[str]) -> str:
        include_name = match.group("path").strip()
        if not include_name or include_name.startswith("|"):
            return " "

        candidate = PurePosixPath(include_name)
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        if not candidate.is_absolute():
            candidate = base / candidate
        normalized = _normalize_path(str(candidate))
        fallback = _normalize_path(str(PurePosixPath(candidate.name)))
        target = normalized if normalized in sources else fallback
        if target not in sources:
            return " "
        return _expand_inputs(target, sources, depth + 1)

    return INCLUDE_RE.sub(replace, text)


def _document_body(text: str) -> str:
    begin = text.find(r"\begin{document}")
    if begin != -1:
        text = text[begin + len(r"\begin{document}") :]
    end = text.find(r"\end{document}")
    if end != -1:
        text = text[:end]
    return text


def _drop_latex_definitions(text: str) -> str:
    for pattern in (NEWCOMMAND_RE, DEF_RE, DECLARE_COMMAND_RE):
        text = pattern.sub(" ", text)
    return text


def _drop_verbatim_blocks(text: str) -> str:
    return VERBATIM_ENV_RE.sub(" ", text)


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_citation_commands(text: str) -> str:
    """Convert citation and cross-reference commands to visible bracketed text."""

    def preserve(match: re.Match[str]) -> str:
        # Collect optional citation notes (the contents of ``[...]``).
        notes = [note.strip() for note in re.findall(r"\[([^]]*)\]", match.group(0))]
        keys = [key.strip() for key in match.group("keys").split(",")]
        content = ", ".join(item for item in notes + keys if item)
        return f" [{content}] " if content else " "

    citation_commands = "|".join(
        re.escape(command)
        for command in sorted(CITATION_COMMANDS, key=len, reverse=True)
    )
    # Match a citation command, up to two optional notes, and its key list.
    text = re.sub(
        rf"\\(?:{citation_commands})\*?(?:\s*\[[^]]*\]){{0,2}}"
        rf"\s*\{{(?P<keys>[^{{}}]*)\}}",
        preserve,
        text,
    )

    reference_commands = "|".join(
        re.escape(command)
        for command in sorted(REFERENCE_COMMANDS, key=len, reverse=True)
    )
    # Cross-references use the same shape but allow only one optional note.
    return re.sub(
        rf"\\(?:{reference_commands})\*?(?:\s*\[[^]]*\])?"
        rf"\s*\{{(?P<keys>[^{{}}]*)\}}",
        preserve,
        text,
    )


def _drop_commands_with_arguments(text: str, commands: set[str]) -> str:
    for command in sorted(commands, key=len, reverse=True):
        text = _replace_balanced_command(text, command, lambda _args: " ")
    return text


def _preserve_href_text(text: str) -> str:
    # Keep the visible text (second argument) of \href{url}{text}.
    text = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", text)
    # A bare \url has no useful prose label, so remove it completely.
    text = re.sub(r"\\url\s*\{[^{}]*\}", " ", text)
    return text


def _unwrap_environments(text: str) -> str:
    # Turn list items, including optionally labelled items, into prose breaks.
    text = re.sub(r"\\item(?:\s*\[[^]]*\])?", ". ", text)

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1).rstrip("*")
        content = match.group(2)
        return f" {content} " if env_name in TEXTUAL_ENVS else " "

    previous = None
    while previous != text:
        previous = text
        text = GENERIC_ENV_RE.sub(replace, text)
    return text


def _unwrap_commands(text: str, commands: set[str]) -> str:
    for command in sorted(commands, key=len, reverse=True):
        text = _replace_balanced_command(
            text,
            command,
            lambda args: f" {args[-1]} " if args else " ",
        )

    previous = None
    while previous != text:
        previous = text
        text = COMMAND_WITH_BRACE_ARG_RE.sub(r" \1 ", text)
    return text


def _replace_balanced_command(text: str, command: str, replacement) -> str:
    # Match exactly one requested command name, with an optional star, without
    # consuming a longer command that merely shares the same prefix.
    command_re = re.compile(rf"\\{re.escape(command)}\*?(?![A-Za-z@])")
    pieces: list[str] = []
    cursor = 0

    while True:
        match = command_re.search(text, cursor)
        if not match:
            pieces.append(text[cursor:])
            break

        args_start = _skip_optional_args(text, match.end())
        args: list[str] = []
        idx = args_start
        while idx < len(text) and text[idx].isspace():
            idx += 1
        while idx < len(text) and text[idx] == "{":
            parsed = _read_balanced_group(text, idx)
            if parsed is None:
                break
            arg, idx = parsed
            args.append(arg)
            idx = _skip_optional_args(text, idx)
            while idx < len(text) and text[idx].isspace():
                idx += 1

        pieces.append(text[cursor : match.start()])
        pieces.append(replacement(args))
        cursor = idx if args else match.end()

    return "".join(pieces)


def _skip_optional_args(text: str, idx: int) -> int:
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text) or text[idx] != "[":
            return idx
        depth = 1
        idx += 1
        while idx < len(text) and depth:
            if text[idx] == "[":
                depth += 1
            elif text[idx] == "]":
                depth -= 1
            idx += 1
    return idx


def _read_balanced_group(text: str, idx: int) -> Optional[tuple[str, int]]:
    if idx >= len(text) or text[idx] != "{":
        return None

    depth = 1
    cursor = idx + 1
    start = cursor
    while cursor < len(text):
        char = text[cursor]
        escaped = cursor > 0 and text[cursor - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
        elif char == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return text[start:cursor], cursor + 1
        cursor += 1
    return None


def _decode_latex_escapes(text: str) -> str:
    replacements = {
        "``": '"',
        "''": '"',
        "---": " ",
        "--": " ",
        "~": " ",
        r"\&": "&",
        r"\%": "%",
        r"\$": "$",
        r"\#": "#",
        r"\_": "_",
        r"\{": " ",
        r"\}": " ",
        r"\/": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    accent_replacements = {
        # Match a TeX accent command and capture its ASCII base letter.
        r"\\['`^\"~=.uvHtcbdkr]\s*\{?([A-Za-z])\}?": r"\1",
        # The remaining patterns match common single-command Latin letters.
        r"\\[ij]": "i",
        r"\\AA": "A",
        r"\\aa": "a",
        r"\\AE": "AE",
        r"\\ae": "ae",
        r"\\O": "O",
        r"\\o": "o",
        r"\\OE": "OE",
        r"\\oe": "oe",
        r"\\ss": "ss",
    }
    for pattern, repl in accent_replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def _drop_remaining_commands(text: str) -> str:
    # Remove any unhandled named command and its optional ``*``/``[...]``.
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\s*\[[^]]*\])?", " ", text)
    # Remove one-character commands such as escaped spacing or punctuation.
    text = re.sub(r"\\.", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return text


def _normalize_embedding_whitespace(text: str) -> str:
    # Protect bracketed citation/reference payloads so keys such as
    # ``thm:main`` are not changed to ``thm: main`` by prose punctuation rules.
    bracketed: list[str] = []

    def stash_brackets(match: re.Match[str]) -> str:
        bracketed.append(match.group(0))
        return f"CITATIONPLACEHOLDER{len(bracketed) - 1}TOKEN"

    # Stash non-nested bracketed content while normalizing prose punctuation.
    text = re.sub(r"\[[^][]*\]", stash_brackets, text)
    # Collapse horizontal whitespace without erasing paragraph/newline structure yet.
    text = re.sub(r"[ \t\f\v]+", " ", text)
    # Normalize spacing after clause punctuation and sentence punctuation.
    text = re.sub(r"\s*([,;:])\s*", r"\1 ", text)
    text = re.sub(r"\s*([.!?])\s*", r"\1 ", text)
    # Remove dangling prepositions left behind when a citation/reference vanished.
    text = re.sub(
        r"\b(?:see|in|by|from|using|via)\s*[.!?](?:\s|$)", " ", text, flags=re.I
    )
    text = re.sub(
        r"\b(?:and|or|see|in|by|from|with|using|via)\s*[.!?]\s*$", " ", text, flags=re.I
    )
    text = re.sub(
        r"\b(?:and|or|see|in|by|from|with|using|via)\s*$", " ", text, flags=re.I
    )
    # Remove whitespace before punctuation, then collapse all remaining whitespace.
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,\n\t")
    for index, original in enumerate(bracketed):
        text = text.replace(f"CITATIONPLACEHOLDER{index}TOKEN", original)
    return text


if __name__ == "__main__":
    import requests

    with requests.Session() as session:
        # This is a hard example, since it has a lot of .tex files and
        # they are in a different folder than the main.tex
        # print(get_intro_text(session, "1607.04703v3"))
        print(get_enrichment_text(session, "2605.28793"))
        print("--------------------------------------------------------")
        print(get_enrichment_text(session, "2404.05864"))
        print("--------------------------------------------------------")
        print(get_enrichment_text(session, "2404.06513v2"))
        print("--------------------------------------------------------")
        print(get_enrichment_text(session, "2607.19346v1"))
        print("--------------------------------------------------------")
        print(get_enrichment_text(session, "2607.14068v2"))
