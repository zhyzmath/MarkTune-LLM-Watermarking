"""Sample records and exclusive-create result files."""

import json
from pathlib import Path


def read_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        from datasets import load_dataset

        yield from load_dataset("parquet", data_files=str(path), split="train")
    elif path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("Sample input must be a .jsonl or .parquet file")


def write_json(path, value):
    # Validate before creating the output, including non-finite metrics.
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(serialized + "\n")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
