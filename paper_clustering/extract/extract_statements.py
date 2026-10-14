"""Extract theorem-like statements and dependencies mentioned in their proofs."""

from bisect import bisect_right
from dataclasses import dataclass
import re

from paper_clustering.data_models import ArxivSection, Statement
from paper_clustering.extract.extract_text import COMMENT_RE, VERBATIM_ENV_RE


STATEMENT_NAMES = {
    "claim", "theorem", "mtheorem", "thm", "thrm", "lemma", "lem",
    "proposition", "prop", "corollary", "cor", "definition", "defn",
    "conjecture", "remark", "example", "assumption", "axiom", "fact",
    "observation", "result",
}
TOKEN_RE = re.compile(
    r"\\(?P<boundary>begin|end)\s*\{(?P<name>[^{}]+)\}"
    r"|\\label\s*(?:\[[^\]]*\]\s*)?\{(?P<label>[^{}]*)\}"
)
REFERENCE_RE = re.compile(
    r"\\(?:ref|eqref|autoref|cref|Cref)\*?\s*\{([^{}]+)\}"
)


@dataclass(frozen=True)
class _Environment:
    name: str
    start: int
    body_start: int
    end: int
    labels: tuple[str, ...]


def _mask_ignored(latex: str) -> str:
    """Mask comments and verbatim blocks while preserving source offsets."""
    def mask(match: re.Match[str]) -> str:
        return re.sub(r"[^\r\n]", " ", match.group())

    return VERBATIM_ENV_RE.sub(mask, COMMENT_RE.sub(mask, latex))


def _statement_names(source: str) -> set[str]:
    return STATEMENT_NAMES | set(
        re.findall(r"\\newtheorem\*?\s*\{([^{}]+)\}", source)
    )


def _environments(source: str) -> list[_Environment]:
    """Find complete environments and labels directly inside each one."""
    stack: list[tuple[str, int, int, list[str]]] = []
    environments: list[_Environment] = []
    for event in TOKEN_RE.finditer(source):
        if event.group("label") is not None:
            if stack:
                stack[-1][3].append(event.group("label"))
            continue
        name = event.group("name")
        if event.group("boundary") == "begin":
            stack.append((name, event.start(), event.end(), []))
        elif stack and stack[-1][0] == name:
            _, start, body_start, labels = stack.pop()
            environments.append(
                _Environment(name, start, body_start, event.start(), tuple(labels))
            )
    return sorted(environments, key=lambda environment: environment.start)


def extract_statements(section: ArxivSection) -> list[Statement]:
    r"""Extract statements from ``section.raw_latex`` with the section's headers.

    Recognize common names, starred variants and locally declared
    ``\newtheorem`` names. Preserve exact bodies without the enclosing
    begin/end commands. Labels include direct ``\label{key}`` and
    ``\label[type]{key}`` commands, excluding nested environment labels.
    Comments and verbatim blocks are ignored when parsing. Nested statements
    are returned in opening order; unclosed environments are skipped.
    """
    source = _mask_ignored(section.raw_latex)
    names = _statement_names(source)
    return [
        Statement(
            major_section_header=section.major_section_header,
            section_header=section.section_header,
            statement_type=environment.name,
            raw_latex=section.raw_latex[environment.body_start:environment.end],
            labels=environment.labels,
        )
        for environment in _environments(source)
        if environment.name.rstrip("*") in names
    ]


def _proof_heading(source: str, start: int, end: int) -> tuple[str, int]:
    """Read an optional bracketed title and return where the proof body starts."""
    position = start
    while position < end and source[position].isspace():
        position += 1
    if position == end or source[position] != "[":
        return "", start
    title_start = position + 1
    brackets = 1
    braces = 0
    position += 1
    while position < end:
        char = source[position]
        if char == "\\":
            position += 2
            continue
        if char == "{":
            braces += 1
        elif char == "}":
            braces = max(0, braces - 1)
        elif not braces:
            if char == "[":
                brackets += 1
            elif char == "]":
                brackets -= 1
                if brackets == 0:
                    return source[title_start:position], position + 1
        position += 1
    return "", start


def _references(text: str) -> list[str]:
    return [
        label.strip()
        for match in REFERENCE_RE.finditer(text)
        for label in match.group(1).split(",")
    ]


def build_statement_dependency(
    statements: list[Statement], sections: list[ArxivSection]
) -> dict[str, list[str]]:
    r"""Map each labelled statement to statements referenced in its proofs.

    Sections must belong to one paper and be in document order. The first
    label of each statement is its key; other labels are aliases. Unlabelled
    statements and references to unknown labels are omitted. Dependencies
    are unique, in encounter order, with self-references excluded.

    The first reference in a proof's optional heading identifies its owner.
    All explicit assignments take priority, even over earlier unnamed proofs.
    Otherwise a proof belongs to the immediately preceding completed statement,
    unless that statement has an explicit proof. An unknown explicit reference
    does not fall back to positional assignment. Proof headings do not add
    dependency edges. Recognizes ref, eqref, autoref, cref and Cref commands,
    including starred forms, and proof/proof* environments.
    """
    dependencies: dict[str, list[str]] = {
        statement.labels[0]: [] for statement in statements if statement.labels
    }
    aliases = {
        label: statement.labels[0]
        for statement in statements
        for label in statement.labels
    }
    source = _mask_ignored("\n\n".join(section.raw_latex for section in sections))
    environments = _environments(source)
    names = _statement_names(source) | {
        statement.statement_type.rstrip("*") for statement in statements
    }
    preceding = sorted(
        (
            environment for environment in environments
            if environment.name.rstrip("*") in names
        ),
        key=lambda environment: environment.end,
    )
    ends = [environment.end for environment in preceding]
    proofs: list[tuple[_Environment, list[str], int]] = []
    explicit_owners: set[str] = set()
    for environment in environments:
        if environment.name.rstrip("*") != "proof":
            continue
        heading, body_start = _proof_heading(
            source, environment.body_start, environment.end
        )
        references = _references(heading)
        proofs.append((environment, references, body_start))
        if references and references[0] in aliases:
            explicit_owners.add(aliases[references[0]])

    for proof, references, body_start in proofs:
        if references:
            owner = aliases.get(references[0])
        else:
            index = bisect_right(ends, proof.start) - 1
            if index < 0:
                continue
            owner = next(
                (aliases[label] for label in preceding[index].labels if label in aliases),
                None,
            )
            if owner in explicit_owners:
                continue
        if owner is None:
            continue
        for label in _references(source[body_start:proof.end]):
            dependency = aliases.get(label)
            if (
                dependency is not None
                and dependency != owner
                and dependency not in dependencies[owner]
            ):
                dependencies[owner].append(dependency)
    return dependencies
