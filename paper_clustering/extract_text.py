"""Download arXiv sources and extract cleaned paper sections."""

import gzip
import io
import logging
import re
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Optional

import requests

from paper_clustering.data_models import ArxivSection, Paper
from paper_clustering.extract_metadata import get_metadata

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_REQUEST_WAIT_TIME = 3
REQUEST_TIMEOUT_SECONDS = 60
MAX_INCLUDE_DEPTH = 8
TEX_SOURCE_SUFFIXES = {".tex", ".ltx"}


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
    \\(?P<kind>part|chapter|section|subsection|subsubsection|paragraph|subparagraph|parhead)
    \*?
    (?:\s*\[[^\]]*\])?
    \s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Remove an unescaped percent sign and everything after it on the same line.
COMMENT_RE = re.compile(r"(?<!\\)%.*")

# Match a simple environment whose body contains no nested environment of the
# same name.  DOTALL allows the body to span lines.
GENERIC_ENV_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}(.*?)\\end\{\1\}", re.DOTALL)

# Locate proof environment boundaries. A token-based pass is used instead of
# one broad expression so nested proof/proof* environments are handled safely.
PROOF_ENV_TOKEN_RE = re.compile(
    r"\\(?:(?P<begin>begin)\s*\{proof\*?\}(?:\s*\[[^]]*\])?"
    r"|(?P<end>end)\s*\{proof\*?\})",
    re.IGNORECASE,
)

# Match a command with one non-nested brace argument and optional ``*``/``[]``.
# This is a fallback after commands needing balanced parsing have been handled.
COMMAND_WITH_BRACE_ARG_RE = re.compile(
    r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?\s*\{([^{}]*)\}"
)

# Match the four standard inline/display math delimiter forms.  Each alternative
# captures only the mathematical body so the delimiters can be discarded.  The
# single-dollar body permits escaped characters (for example, ``\{``) without
# mistaking them for delimiters.
MATH_SPAN_RE = re.compile(
    r"""
    \$\$(?P<double_dollar>.*?)\$\$
    |
    (?<!\\)\$(?!\$)(?P<single_dollar>(?:\\.|[^$])*?)(?<!\\)\$(?!\$)
    |
    \\\[(?P<bracket>.*?)\\\]
    |
    \\\((?P<parenthesis>.*?)\\\)
    """,
    re.DOTALL | re.VERBOSE,
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

NEW_PARAGRAPH_RE = re.compile(r"(?:\r\n|\n|\r){2}|\\par")

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
    "mtheorem",
    "thrm",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "conjecture",
    "remark",
    "example",
    "proof",
}

# Preserve the bodies of display-math environments. These are normalized as
# math rather than wrapped in start/end markers like theorem-like prose.
MATH_ENVS = {
    "align",
    "alignat",
    "aligned",
    "alignedat",
    "array",
    "bmatrix",
    "cases",
    "displaymath",
    "equation",
    "eqnarray",
    "flalign",
    "gather",
    "gathered",
    "math",
    "matrix",
    "multline",
    "pmatrix",
    "smallmatrix",
    "split",
    "vmatrix",
}


def download_source(session: requests.Session, arxiv_id: str) -> bytes:
    """Download the source archive for ``arxiv_id`` using the supplied session."""
    response = session.get(
        ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.content


def get_paper(
    session: requests.Session,
    arxiv_id: str,
    include_proofs: bool = False,
) -> Paper:
    """Download and clean the sections of an arXiv paper.

    When ``include_proofs`` is false, proof and proof* environments are removed
    in their entirety before the remaining LaTeX is converted to plain text.
    """
    metadata = get_metadata(session, arxiv_id)
    time.sleep(ARXIV_REQUEST_WAIT_TIME)
    try:
        source_bytes = download_source(session, arxiv_id)
        source = build_tex_source_from_bytes(source_bytes, arxiv_id)
        if not source:
            return Paper(metadata=metadata, sections=())
        if not include_proofs:
            source = _drop_proof_environments(source)
        sections = extract_sections_from_latex(source)
        results: list[ArxivSection] = []
        major_section_header = ""
        for section_index, section in enumerate(sections):
            section_header = clean_latex_for_embedding(section.title)
            if section.kind in {"part", "chapter", "section"}:
                major_section_header = section_header
            results.append(
                ArxivSection(
                    arxiv_id=metadata.arxiv_id,
                    section_index=section_index,
                    section_header=section_header,
                    major_section_header=major_section_header,
                    text=tuple(
                        _normalize_embedding_whitespace(p)
                        for p in NEW_PARAGRAPH_RE.split(
                            clean_latex_for_embedding(section.text)
                        )
                        if p.strip()
                    ),
                ),
            )
        return Paper(metadata=metadata, sections=tuple(results))
    except (
        requests.RequestException,
        tarfile.TarError,
        zipfile.BadZipFile,
        OSError,
        UnicodeError,
    ) as exc:
        logging.error(
            f"{time.localtime()}: Could not download or extract {arxiv_id}: {exc}"
        )
        return Paper(metadata=metadata, sections=())


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


def build_tex_source(sources: list[TexSource]) -> str:
    """Select the most likely main TeX file and expand its includes."""
    source_map = {_normalize_path(source.name): source.text for source in sources}
    if not source_map:
        return ""
    main_name = max(
        source_map, key=lambda name: _main_file_score(name, source_map[name])
    )
    return _expand_inputs(main_name, source_map)


def build_tex_source_from_bytes(source_bytes: bytes, arxiv_id: str) -> str:
    """Read an arXiv source payload and assemble its likely main document."""
    return build_tex_source(list(iter_tex_sources(source_bytes, arxiv_id)))


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
    text = _normalize_math_spans(text)
    text = _decode_latex_escapes(text)
    text = _drop_remaining_commands(text)
    # text = _normalize_embedding_whitespace(text)
    return text


def strip_latex_comments(text: str) -> str:
    return "\n".join(COMMENT_RE.sub("", line) for line in text.splitlines())


def _drop_proof_environments(text: str) -> str:
    """Remove proof/proof* environments, including their complete contents."""
    text = strip_latex_comments(text)
    pieces: list[str] = []
    cursor = 0
    depth = 0

    for match in PROOF_ENV_TOKEN_RE.finditer(text):
        if match.group("begin"):
            if depth == 0:
                pieces.append(text[cursor : match.start()])
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                cursor = match.end()

    # An unmatched opening proof conservatively removes everything following
    # it; otherwise retain all text after the final complete proof environment.
    if depth == 0:
        pieces.append(text[cursor:])
    return " ".join(pieces)


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


def _normalize_citation_commands(text: str) -> str:
    """Preserve citations as bracketed text and omit cross-references."""

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
    # Internal cross-references (including references to theorems and lemmas)
    # are document-local identifiers that add no useful prose after extraction.
    # Drop them rather than exposing labels such as ``[thm:main]``.
    return re.sub(
        rf"\\(?:{reference_commands})\*?(?:\s*\[[^]]*\])?"
        rf"\s*\{{(?P<keys>[^{{}}]*)\}}",
        " ",
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
        env_key = env_name.lower()
        content = match.group(2)
        if env_key in TEXTUAL_ENVS:
            # A theorem-like environment is one semantic block. Preserve outer
            # paragraph boundaries, but prevent blank lines or \par commands in
            # its body from separating the start/end markers into malformed
            # fragments such as ``We let end-of-definition``.
            content = NEW_PARAGRAPH_RE.sub(" ", content)
            return f" start-of-{env_key}:{content} end-of-{env_key} "
        if env_key in MATH_ENVS:
            # Displayed formulas often carry the entire conclusion of a lemma.
            # Treat their bodies exactly like delimiter-based math instead of
            # dropping them as unknown non-prose environments.
            content = NEW_PARAGRAPH_RE.sub(" ", content)
            return f" {_normalize_math_body(content)} "
        return " "

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


def _normalize_math_body(body: str) -> str:
    """Render the contents of a math span as readable plain text."""
    # Strip nested math-environment wrappers before normalizing their contents.
    body = re.sub(r"\\(?:begin|end)\s*\{[^{}]+\}", " ", body)
    body = body.replace(r"\{", "(").replace(r"\}", ")")
    body = body.replace("{", "(").replace("}", ")")
    # Retain readable command names such as ``rightarrow``, ``in``, and ``leq``.
    return re.sub(r"\\([A-Za-z@]+)\*?", r"\1", body)


def _normalize_math_spans(text: str) -> str:
    """Remove math delimiters and render mathematical braces as parentheses.

    Brace replacement is limited to recognized math spans so braces used as
    LaTeX grouping elsewhere remain available to the command-cleaning passes.
    Escaped dollar signs and unmatched delimiters are removed in the final two
    replacements, which guarantees that no dollar sign reaches the output.
    """

    def normalize(match: re.Match[str]) -> str:
        body = next(group for group in match.groups() if group is not None)
        return f" {_normalize_math_body(body)} "

    text = MATH_SPAN_RE.sub(normalize, text)
    return text.replace(r"\$", "").replace("$", "")


def _decode_latex_escapes(text: str) -> str:
    """Decode common LaTeX punctuation and accent commands.

    Alphabetic accent commands require a braced argument.  Without that
    boundary, the ``\r`` accent pattern would also match the beginning of
    commands such as ``\rightarrow`` and silently turn it into ``ightarrow``.
    """
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
        # Punctuation accents are one-character commands, so their base letter
        # may be bare or enclosed in braces.
        r"\\['`^\"~=.]\s*\{?([A-Za-z])\}?": r"\1",
        # Letter-named accents must use braces here to distinguish ``\r{a}``
        # from longer named commands such as ``\rightarrow``.
        r"\\[uvHtcbdkr]\s*\{([A-Za-z])\}": r"\1",
        # These patterns require a command boundary so, for example, ``\O``
        # cannot consume the start of the longer command ``\Omega``.
        r"\\i(?![A-Za-z@])": "i",
        r"\\j(?![A-Za-z@])": "j",
        r"\\AA(?![A-Za-z@])": "A",
        r"\\aa(?![A-Za-z@])": "a",
        r"\\AE(?![A-Za-z@])": "AE",
        r"\\ae(?![A-Za-z@])": "ae",
        r"\\O(?![A-Za-z@])": "O",
        r"\\o(?![A-Za-z@])": "o",
        r"\\OE(?![A-Za-z@])": "OE",
        r"\\oe(?![A-Za-z@])": "oe",
        r"\\ss(?![A-Za-z@])": "ss",
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
    # Remove whitespace before punctuation, then collapse all remaining whitespace.
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,\n\t")
    for index, original in enumerate(bracketed):
        text = text.replace(f"CITATIONPLACEHOLDER{index}TOKEN", original)
    return text


if __name__ == "__main__":
    from pprint import pp

    with requests.session() as session:
        pp(get_paper(session, "2308.15403"))
