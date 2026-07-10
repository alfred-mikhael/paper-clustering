"""Parses latex documents and extracts introduction section. Cleans up math notation in the introduction in
3 levels of coarseness. Generated almost entirely by Codex using GPT 5.5."""

import gzip
import io
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterable, Literal, Optional

import requests

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
REQUEST_TIMEOUT_SECONDS = 60
MAX_INCLUDE_DEPTH = 8
MathCoarseness = Literal["no_math", "coarse", "fine"]
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


SECTION_RE = re.compile(
    r"""
    \\(?P<kind>part|chapter|section|subsection|subsubsection)
    \*?
    (?:\s*\[[^\]]*\])?
    \s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}
    """,
    re.IGNORECASE | re.VERBOSE,
)

INTRO_TITLE_RE = re.compile(
    r"\b(introduction|intro|overview|background)\b", re.IGNORECASE
)

COMMENT_RE = re.compile(r"(?<!\\)%.*")
MATH_ENV_RE = re.compile(
    r"""
    \\begin\{
    (?P<env>
        equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|
        displaymath|eqnarray\*?|split|cases|array|pmatrix|bmatrix|vmatrix
    )
    \}
    (?P<content>.*?)
    \\end\{(?P=env)\}
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
GENERIC_ENV_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}(.*?)\\end\{\1\}", re.DOTALL)
COMMAND_WITH_BRACE_ARG_RE = re.compile(
    r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?\s*\{([^{}]*)\}"
)

DROP_COMMANDS = {
    "addcontentsline",
    "bibliographystyle",
    "bibliography",
    "caption",
    "cite",
    "citealp",
    "citeauthor",
    "citep",
    "citet",
    "citeyear",
    "cref",
    "Cref",
    "eqref",
    "footnote",
    "footnotemark",
    "footnotetext",
    "index",
    "label",
    "pageref",
    "ref",
    "subref",
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

LATEX_MATH_SYMBOLS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\varepsilon": "epsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\vartheta": "theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\varrho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\varphi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
    r"\Gamma": "gamma",
    r"\Delta": "delta",
    r"\Theta": "theta",
    r"\Lambda": "lambda",
    r"\Xi": "xi",
    r"\Pi": "pi",
    r"\Sigma": "sigma",
    r"\Phi": "phi",
    r"\Psi": "psi",
    r"\Omega": "omega",
    r"\mathbb{F}": "finite_field",
    r"\mathbb F": "finite_field",
    r"\mathbb{R}": "real",
    r"\mathbb R": "real",
    r"\mathbb{N}": "natural",
    r"\mathbb N": "natural",
    r"\mathbb{Z}": "integer",
    r"\mathbb Z": "integer",
    r"\mathbb{Q}": "rational",
    r"\mathbb Q": "rational",
    r"\mathbb{C}": "complex",
    r"\mathbb C": "complex",
    r"\mathcal{P}": "power_set",
    r"\emptyset": "empty_set",
    r"\varnothing": "empty_set",
    r"\infty": "infinity",
    r"\log": "log",
    r"\ln": "ln",
    r"\exp": "exp",
    r"\poly": "poly",
    r"\Pr": "probability",
    r"\mathop": " ",
    r"\operatorname": " ",
    r"\mathbf": " ",
    r"\mathrm": " ",
    r"\mathsf": " ",
    r"\mathit": " ",
    r"\mathcal": " ",
    r"\text": " ",
}

LATEX_MATH_OPERATORS = {
    r"\leq": "leq",
    r"\le": "leq",
    r"\geq": "geq",
    r"\ge": "geq",
    r"\neq": "neq",
    r"\ne": "neq",
    r"\approx": "approx",
    r"\sim": "sim",
    r"\simeq": "simeq",
    r"\equiv": "equiv",
    r"\in": "in",
    r"\notin": "not_in",
    r"\subseteq": "subseteq",
    r"\subset": "subset",
    r"\supseteq": "supseteq",
    r"\cup": "union",
    r"\cap": "intersection",
    r"\times": "times",
    r"\cdot": "times",
    r"\ast": "star",
    r"\star": "star",
    r"\oplus": "xor",
    r"\otimes": "tensor",
    r"\wedge": "and",
    r"\vee": "or",
    r"\land": "and",
    r"\lor": "or",
    r"\to": "to",
    r"\rightarrow": "to",
    r"\leftarrow": "from",
    r"\mapsto": "maps_to",
    r"\Rightarrow": "implies",
    r"\implies": "implies",
    r"\iff": "iff",
    r"\forall": "for_all",
    r"\exists": "exists",
    r"\sum": "sum",
    r"\prod": "product",
    r"\min": "min",
    r"\max": "max",
    r"\argmin": "argmin",
    r"\argmax": "argmax",
}


def download_source(session: requests.Session, arxiv_id: str) -> bytes:
    response = session.get(
        ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.content


def get_intro_text(
    session: requests.Session, arxiv_id: str, coarseness: MathCoarseness = "fine"
) -> str:
    """Download an arXiv source package and return embedding-ready introduction text."""
    try:
        source_bytes = download_source(session, arxiv_id)
        expanded_source = build_intro_tex_source_from_bytes(source_bytes, arxiv_id)
        if not expanded_source:
            return ""
        intro = extract_intro_from_latex(expanded_source)
        return clean_latex_for_embedding(intro, coarseness=coarseness)
    except (
        requests.RequestException,
        tarfile.TarError,
        zipfile.BadZipFile,
        OSError,
        UnicodeError,
    ):
        return ""


def iter_tex_sources(
    source_bytes: bytes, arxiv_id: str = "source"
) -> Iterable[TexSource]:
    """Yield TeX-like files from common arXiv source formats without extracting to disk."""
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
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def build_main_tex_source(sources: list[TexSource]) -> str:
    source_map = {_normalize_path(source.name): source.text for source in sources}
    candidates = [
        name
        for name in source_map
        if PurePosixPath(name).suffix.lower() in TEX_SOURCE_SUFFIXES
    ]
    main_name = max(
        candidates or source_map,
        key=lambda name: _main_file_score(name, source_map[name]),
    )
    return _expand_inputs(main_name, source_map)


def build_intro_tex_source(sources: list[TexSource]) -> str:
    source_map = {_normalize_path(source.name): source.text for source in sources}
    candidates = [
        name
        for name in source_map
        if PurePosixPath(name).suffix.lower() in TEX_SOURCE_SUFFIXES
    ]
    main_name = max(
        candidates or source_map,
        key=lambda name: _main_file_score(name, source_map[name]),
    )

    main_text = source_map[main_name]
    if extract_intro_from_latex(main_text):
        return main_text

    intro_expanded = _expand_inputs(
        main_name,
        source_map,
        include_filter=_is_likely_intro_include,
    )
    if extract_intro_from_latex(intro_expanded):
        return intro_expanded

    return _expand_inputs(main_name, source_map)


def build_intro_tex_source_from_bytes(source_bytes: bytes, arxiv_id: str) -> str:
    if tarfile.is_tarfile(io.BytesIO(source_bytes)):
        return _build_intro_tex_source_from_tar(source_bytes)

    buffer = io.BytesIO(source_bytes)
    if zipfile.is_zipfile(buffer):
        return _build_intro_tex_source_from_zip(source_bytes)

    if source_bytes.startswith(b"\x1f\x8b"):
        decompressed = gzip.decompress(source_bytes)
        return build_intro_tex_source_from_bytes(decompressed, arxiv_id)

    return decode_source(source_bytes)


def _build_intro_tex_source_from_tar(source_bytes: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:*") as tar:
        members = {
            _normalize_path(member.name): member
            for member in tar.getmembers()
            if member.isfile() and _is_latex_file(member.name)
        }
        if not members:
            return ""

        cache: dict[str, str] = {}

        def read_source(name: str) -> str:
            normalized = _normalize_path(name)
            if normalized not in cache:
                extracted = tar.extractfile(members[normalized])
                cache[normalized] = (
                    decode_source(extracted.read()) if extracted is not None else ""
                )
            return cache[normalized]

        return _build_intro_tex_source_lazy(list(members), read_source)


def _build_intro_tex_source_from_zip(source_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        name_map = {
            _normalize_path(name): name
            for name in archive.namelist()
            if _is_latex_file(name)
        }
        names = list(name_map)
        if not names:
            return ""

        cache: dict[str, str] = {}

        def read_source(name: str) -> str:
            normalized = _normalize_path(name)
            if normalized not in cache:
                cache[normalized] = decode_source(archive.read(name_map[normalized]))
            return cache[normalized]

        return _build_intro_tex_source_lazy(names, read_source)


def _build_intro_tex_source_lazy(
    source_names: list[str], read_source: Callable[[str], str]
) -> str:
    main_name = _select_main_tex_name(source_names, read_source)
    if not main_name:
        return ""

    main_text = read_source(main_name)
    if extract_intro_from_latex(main_text):
        return main_text

    intro_expanded = _expand_inputs_lazy(
        main_name,
        set(source_names),
        read_source,
        include_filter=_is_likely_intro_include,
    )
    if extract_intro_from_latex(intro_expanded):
        return intro_expanded

    return _expand_inputs_lazy(main_name, set(source_names), read_source)


def _select_main_tex_name(
    source_names: list[str], read_source: Callable[[str], str]
) -> str:
    candidates = [
        name
        for name in source_names
        if PurePosixPath(name).suffix.lower() in TEX_SOURCE_SUFFIXES
    ]
    if not candidates:
        return ""

    def priority(name: str) -> tuple[int, int, str]:
        path = PurePosixPath(name)
        preferred = path.name.lower() in {
            "main.tex",
            "paper.tex",
            "article.tex",
            "ms.tex",
        }
        return (0 if preferred else 1, len(path.parts), name)

    best_name = candidates[0]
    best_score = (-1, -1)
    for name in sorted(candidates, key=priority):
        text = read_source(name)
        score = _main_file_score(name, text)
        if score > best_score:
            best_name = name
            best_score = score
        if "\\begin{document}" in text:
            return name

    return best_name


def extract_intro_from_latex(source_text: str) -> str:
    text = strip_latex_comments(source_text)
    document = _document_body(text)
    sections = list(SECTION_RE.finditer(document))

    for idx, match in enumerate(sections):
        if not INTRO_TITLE_RE.search(match.group("title")):
            continue

        start = match.end()
        end = len(document)
        for next_match in sections[idx + 1 :]:
            if next_match.group("kind").lower() in {"part", "chapter", "section"}:
                end = next_match.start()
                break
        return document[start:end]

    if sections:
        first_section_start = sections[0].start()
        preface = document[:first_section_start]
        if len(preface.split()) >= 100:
            return preface

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        document,
        re.IGNORECASE | re.DOTALL,
    )
    return abstract_match.group(1) if abstract_match else ""


def clean_latex_for_embedding(text: str, coarseness: MathCoarseness = "fine") -> str:
    """Normalize LaTeX into semantic prose suitable for vector embeddings."""
    text = strip_latex_comments(text)
    text = _drop_latex_definitions(text)
    text = _drop_verbatim_blocks(text)
    text = _replace_math(text, coarseness)
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
        score += 20
    if SECTION_RE.search(text):
        score += 10
    return score, len(text)


def _expand_inputs(
    name: str,
    sources: dict[str, str],
    depth: int = 0,
    include_filter=None,
) -> str:
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
        if include_filter is not None and not include_filter(target):
            return " "
        return _expand_inputs(target, sources, depth + 1, include_filter)

    return re.sub(
        r"\\(?:input|include)\s*\{(?P<path>[^{}]+)\}",
        replace,
        text,
        flags=re.IGNORECASE,
    )


def _expand_inputs_lazy(
    name: str,
    source_names: set[str],
    read_source: Callable[[str], str],
    depth: int = 0,
    include_filter=None,
) -> str:
    text = read_source(name)
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
        target = normalized if normalized in source_names else fallback
        if target not in source_names:
            return " "
        if include_filter is not None and not include_filter(target):
            return " "
        return _expand_inputs_lazy(
            target,
            source_names,
            read_source,
            depth + 1,
            include_filter,
        )

    return re.sub(
        r"\\(?:input|include)\s*\{(?P<path>[^{}]+)\}",
        replace,
        text,
        flags=re.IGNORECASE,
    )


def _is_likely_intro_include(path: str) -> bool:
    name = PurePosixPath(path).stem.lower()
    return bool(INTRO_TITLE_RE.search(name))


def _document_body(text: str) -> str:
    begin = text.find(r"\begin{document}")
    if begin != -1:
        text = text[begin + len(r"\begin{document}") :]
    end = text.find(r"\end{document}")
    if end != -1:
        text = text[:end]
    return text


def _drop_latex_definitions(text: str) -> str:
    patterns = [
        (
            r"\\(?:newcommand|renewcommand|providecommand)\s*"
            r"\{?\\[a-zA-Z@]+\}?"
            r"(?:\s*\[[^\]]*\]){0,2}\s*"
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        ),
        r"\\(?:def|gdef|edef|xdef)\s*\\[a-zA-Z@]+\s*(?:#[0-9]\s*)*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        r"\\(?:DeclareMathOperator|DeclarePairedDelimiter)\*?\s*\{?\\[a-zA-Z@]+\}?\s*\{[^{}]*\}",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.DOTALL)
    return text


def _drop_verbatim_blocks(text: str) -> str:
    return re.sub(
        r"\\begin\{(?:verbatim|lstlisting|minted)\}.*?\\end\{(?:verbatim|lstlisting|minted)\}",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _replace_math(text: str, coarseness: MathCoarseness) -> str:
    _validate_math_coarseness(coarseness)

    def replacement(match: re.Match[str]) -> str:
        return f" {_latex_math_to_text(match.group('content'), coarseness)} "

    def positional_replacement(match: re.Match[str]) -> str:
        return f" {_latex_math_to_text(match.group(1), coarseness)} "

    text = MATH_ENV_RE.sub(
        replacement,
        text,
    )
    text = re.sub(
        r"\\\[(.*?)\\\]",
        positional_replacement,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        positional_replacement,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\$\$(.*?)\$\$",
        positional_replacement,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?<!\\)\$((?:\\.|[^$])*)(?<!\\)\$",
        positional_replacement,
        text,
        flags=re.DOTALL,
    )
    return text


def _validate_math_coarseness(coarseness: str) -> None:
    if coarseness not in {"no_math", "coarse", "fine"}:
        raise ValueError("coarseness must be one of 'no_math', 'coarse', or 'fine'")


def _latex_math_to_text(math_text: str, coarseness: MathCoarseness) -> str:
    if coarseness == "no_math":
        return "math expression"

    if coarseness == "coarse":
        return _coarse_latex_math_to_text(math_text)

    math_text = _normalize_common_math_domains(math_text)
    math_text = _replace_latex_math_symbols(math_text)
    math_text = _unwrap_math_command_arguments(math_text)
    math_text = _replace_latex_math_operators(math_text)
    math_text = _normalize_math_syntax(math_text)
    return _compact_math_token(math_text)


def _coarse_latex_math_to_text(math_text: str) -> str:
    coarse_terms = _extract_coarse_math_terms(math_text)
    return " ".join(coarse_terms or ["math expression"])


def _extract_coarse_math_terms(math_text: str) -> list[str]:
    raw = math_text.lower()
    normalized = _unwrap_math_command_arguments(math_text)
    normalized = normalized.lower()
    search_text = f"{raw} {normalized}"
    terms: list[str] = []

    coarse_patterns = [
        (
            r"(\\[A-za-z]*|.).s?(:|\\colon|\\Colon).*\\(?:to|rightarrow)",
            "function to",
        ),
        (
            r"..s?\\in\s*(?:\\mathbb\s*\{\s*f\s*\}|\\mathbb\s+f|\\mathbf\s*\{\s*f\s*\}|f|\\(bits|Bits|nbits)(\^(\{?\\[A-za-z]*\}?|.))?)",
            "finite field vector",
        ),
        (r"(\\?\{\s*0\s*,\s*1\s*\\?\})|\\(bits|Bits|nbits)", "boolean cube"),
        (
            r"(?:\\mathbb\s*\{\s*f\s*\}|\\mathbb\s+f|\\mathbf\s*\{\s*f\s*\}|f)"
            r"\s*_\s*\{?\s*(?:2|q|p|[0-9]+)\s*\}?",
            "finite field",
        ),
        (r"\\mathbb\s*\{\s*r\s*\}|\\mathbb\s+r", "real space"),
        (r"\\mathbb\s*\{\s*c\s*\}|\\mathbb\s+c", "complex space"),
        (r"\\mathbb\s*\{\s*z\s*\}|\\mathbb\s+z", "integer lattice"),
        (r"\\mathbb\s*\{\s*n\s*\}|\\mathbb\s+n", "natural numbers"),
        (r"\\(?:pr|prob|probability|Pr)\b", "probability"),
        (r"\\(?:mathbb\s*\{\s*e\s*\}|mathbb\s+e|expectation|ex)\b", "expectation"),
        (r"\\(?:sum|prod)\b", "aggregate"),
        (r"\\(?:min|max|argmin|argmax)\b", "optimization"),
        (r"\\(?:otimes|tensor)\b", "tensor"),
        (r"\\(?:oplus|xor)\b", "xor"),
        (r"\\(?:forall|exists)\b", "quantifier"),
        (r"\\(?:leq|le|geq|ge|neq|ne|approx|sim|simeq|equiv)\b|[<>=]", "relations"),
        (r"\\(?:frac|binom|sqrt)\b", "formula"),
    ]

    for pattern, term in coarse_patterns:
        if re.search(pattern, search_text):
            terms.append(term)

    return _dedupe_preserving_order(terms)


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_common_math_domains(math_text: str) -> str:
    math_text = re.sub(
        r"\\?\{\s*0\s*,\s*1\s*\\?\}\s*\^\s*\{?\s*([A-Za-z0-9]+)\s*\}?",
        r" finite_field_2_\1 ",
        math_text,
    )
    math_text = re.sub(
        r"\\?\{\s*0\s*,\s*1\s*\\?\}",
        " finite_field_2 ",
        math_text,
    )
    math_text = re.sub(
        r"\\?\{\s*1\s*,\s*\\(?:ldots|dots|cdots)\s*,\s*([A-Za-z0-9]+)\s*\\?\}",
        r" interval_\1 ",
        math_text,
    )
    math_text = re.sub(
        r"\[\s*([A-Za-z0-9]+)\s*\]",
        r" interval_\1 ",
        math_text,
    )
    math_text = re.sub(
        r"(?:\\mathbb\s*\{\s*F\s*\}|\\mathbb\s+F|F)\s*_\s*\{?\s*([A-Za-z0-9]+)\s*\}?"
        r"\s*\^\s*\{?\s*([A-Za-z0-9]+)\s*\}?",
        r" finite_field_\1_\2 ",
        math_text,
    )
    math_text = re.sub(
        r"(?:\\mathbb\s*\{\s*F\s*\}|\\mathbb\s+F|F)\s*_\s*\{?\s*([A-Za-z0-9]+)\s*\}?",
        r" finite_field_\1 ",
        math_text,
    )
    return math_text


def _unwrap_math_command_arguments(math_text: str) -> str:
    replacements = {
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}": r" \1 over \2 ",
        r"\\binom\s*\{([^{}]+)\}\s*\{([^{}]+)\}": r" choose \1 \2 ",
        r"\\sqrt\s*\{([^{}]+)\}": r" sqrt \1 ",
        r"\\operatorname\s*\{([^{}]+)\}": r" \1 ",
        r"\\text\s*\{([^{}]+)\}": r" \1 ",
        r"\\(?:mathbb|mathcal|mathbf|mathrm|mathsf|mathit)\s*\{([^{}]+)\}": r" \1 ",
    }
    previous = None
    while previous != math_text:
        previous = math_text
        for pattern, repl in replacements.items():
            math_text = re.sub(pattern, repl, math_text)
    return math_text


def _replace_latex_math_symbols(math_text: str) -> str:
    symbols = sorted(
        LATEX_MATH_SYMBOLS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for latex, token in symbols:
        math_text = math_text.replace(latex, f" {token} ")
    return math_text


def _replace_latex_math_operators(math_text: str) -> str:
    operators = sorted(
        LATEX_MATH_OPERATORS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for latex, token in operators:
        math_text = math_text.replace(latex, f" {token} ")
    return math_text


def _normalize_math_syntax(math_text: str) -> str:
    replacements = {
        "^": "_",
        "_": "_",
        "=": " equals ",
        "+": " plus ",
        "-": " minus ",
        "/": " over ",
        "<": " lt ",
        ">": " gt ",
        "|": " given ",
        "&": " and ",
        ",": " ",
        ";": " ",
        ":": " ",
    }
    for old, new in replacements.items():
        math_text = math_text.replace(old, new)
    math_text = re.sub(r"\\[a-zA-Z@]+", " ", math_text)
    return math_text


def _compact_math_token(math_text: str) -> str:
    math_text = math_text.lower()
    math_text = re.sub(r"[^a-z0-9]+", "_", math_text)
    math_text = re.sub(r"_+", "_", math_text).strip("_")
    return math_text


def _normalize_citation_commands(text: str) -> str:
    cite_commands = "|".join(sorted(DROP_COMMANDS))
    return re.sub(
        rf"\\(?:{cite_commands})\*?(?:\s*\[[^\]]*\]){{0,2}}\s*\{{[^{{}}]*\}}",
        " ",
        text,
    )


def _drop_commands_with_arguments(text: str, commands: set[str]) -> str:
    for command in sorted(commands, key=len, reverse=True):
        text = _replace_balanced_command(text, command, lambda _args: " ")
    return text


def _preserve_href_text(text: str) -> str:
    text = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\url\s*\{[^{}]*\}", " ", text)
    return text


def _unwrap_environments(text: str) -> str:
    text = re.sub(r"\\item(?:\s*\[[^\]]*\])?", ". ", text)

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
        r"\\['`^\"~=.uvHtcbdkr]\s*\{?([A-Za-z])\}?": r"\1",
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
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\s*\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\.", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return text


def _normalize_embedding_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s*([,;:])\s*", r"\1 ", text)
    text = re.sub(r"\s*([.!?])\s*", r"\1 ", text)
    text = re.sub(
        r"\b(?:see|in|by|from|using|via)\s*[.!?](?:\s|$)", " ", text, flags=re.I
    )
    text = re.sub(
        r"\b(?:and|or|see|in|by|from|with|using|via)\s*[.!?]\s*$", " ", text, flags=re.I
    )
    text = re.sub(
        r"\b(?:and|or|see|in|by|from|with|using|via)\s*$", " ", text, flags=re.I
    )
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,\n\t")


if __name__ == "__main__":
    import requests

    with requests.Session() as session:
        # This is a hard example, since it has a lot of .tex files and
        # they are in a different folder than the main.tex
        print(get_intro_text(session, "1607.04703v3", coarseness="coarse"))
