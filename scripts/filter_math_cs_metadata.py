"""Create a JSONL subset of the Kaggle arXiv metadata snapshot.

Example:
    python scripts/filter_math_cs_metadata.py \
        data/archive.zip data/math_cs_metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter math.* and cs.* papers from an arXiv metadata ZIP into JSONL."
    )
    parser.add_argument("archive", type=Path, help="Kaggle arXiv metadata ZIP file")
    parser.add_argument("output", type=Path, help="Destination JSONL file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    return parser.parse_args()


def is_math_or_cs(record: dict[str, object]) -> bool:
    """Return whether any arXiv category belongs to computer science or math."""
    return any(
        category.startswith(("cs.", "math."))
        for category in str(record.get("categories", "")).split()
    )


def write_subset(
    archive_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Stream a ZIP snapshot and write matching records to a JSONL file."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()

    matching_records = 0

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.endswith((".json", ".jsonl"))
        ]
        if len(members) != 1:
            raise ValueError("archive must contain exactly one JSON or JSONL metadata member")

        with temporary_path.open("wb") as output:
            with archive.open(members[0]) as records:
                for raw_record in tqdm(records, desc="Filtering metadata", unit="records"):
                    try:
                        record = json.loads(raw_record)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict) or not is_math_or_cs(record):
                        continue

                    output.write(raw_record if raw_record.endswith(b"\n") else raw_record + b"\n")
                    matching_records += 1

    temporary_path.replace(output_path)
    return matching_records


def main() -> None:
    args = parse_args()
    count = write_subset(
        args.archive,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count:,} math/cs papers to {args.output}")


if __name__ == "__main__":
    main()
