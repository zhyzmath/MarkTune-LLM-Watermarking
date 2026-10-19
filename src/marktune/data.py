"""Prepare document-disjoint training data and fixed evaluation prompts."""

import hashlib
import json
import random
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError("Input must be a .parquet or .jsonl file")


def document_id(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def split_documents(rows, tokenizer, counts, seed, min_tokens=512):
    """Deduplicate complete documents before splitting and truncating."""
    eligible = []
    seen = set()
    for row in rows:
        text = row["text"].strip()
        identity = document_id(text)
        if identity in seen:
            continue
        seen.add(identity)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= min_tokens:
            eligible.append({"text": text, "document_id": identity})
    random.Random(seed).shuffle(eligible)
    total = sum(counts.values())
    if len(eligible) < total:
        raise ValueError(f"Need {total} unique eligible documents, found {len(eligible)}")
    result = {}
    offset = 0
    for name, count in counts.items():
        if type(count) is not int or count <= 0:
            raise ValueError("Every split needs a positive integer document count")
        result[name] = eligible[offset : offset + count]
        offset += count
    return result


def prepare_training_data(config, input_path, counts, source):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"], trust_remote_code=False
    )
    splits = split_documents(
        read_rows(input_path), tokenizer, counts, config["data"]["seed"], config["training"]["ce_sequence_length"]
    )
    output = Path(config["data"]["directory"])
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(config["data"]["seed"])
    for split, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), output / f"{split}.parquet")
        if split in ("train", "validation"):
            prompts = []
            for row in rows:
                length = rng.randint(config["data"]["min_prompt_tokens"], config["data"]["max_prompt_tokens"])
                ids = tokenizer.encode(row["text"], add_special_tokens=False)[:length]
                prompts.append(
                    {
                        "prompt_text": tokenizer.decode(ids, skip_special_tokens=True),
                        "data_source": "openwebtext",
                        "document_id": row["document_id"],
                    }
                )
            pq.write_table(pa.Table.from_pylist(prompts), output / f"prompts-{split}.parquet")
    manifest = {
        "source": source,
        "input_sha256": file_sha256(input_path),
        "seed": config["data"]["seed"],
        "model": config["model"]["name"],
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or config["model"]["revision"],
        "counts": counts,
        "document_ids": {name: [r["document_id"] for r in rows] for name, rows in splits.items()},
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)


def build_evaluation_rows(rows, tokenizer, task, num_samples, prompt_tokens=200, response_tokens=200):
    result = []
    seen = set()
    for row in rows:
        if task == "c4":
            text = row["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= prompt_tokens + response_tokens:
                continue
            prompt = tokenizer.decode(ids[:prompt_tokens], skip_special_tokens=True)
            reference = tokenizer.decode(ids[prompt_tokens : prompt_tokens + response_tokens], skip_special_tokens=True)
            identity = document_id(text)
        elif task == "eli5":
            question, answer = row["question"], row["answer"]
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) <= response_tokens:
                continue
            prompt = question
            reference = tokenizer.decode(ids[:response_tokens], skip_special_tokens=True)
            identity = document_id(question + "\n" + answer)
        else:
            raise ValueError(f"Unknown evaluation task: {task}")
        if identity in seen:
            continue
        seen.add(identity)
        result.append({"id": identity, "task": task, "prompt": prompt, "reference": reference})
        if len(result) == num_samples:
            return result
    raise ValueError(f"Need {num_samples} eligible {task} samples, found {len(result)}")


def prepare_evaluation_data(config, input_path, output, task, source):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"], trust_remote_code=False
    )
    evaluation = config["evaluation"]
    rows = build_evaluation_rows(
        read_rows(input_path),
        tokenizer,
        task,
        evaluation["num_samples"],
        evaluation["prompt_tokens"],
        evaluation["response_tokens"],
    )
    output = Path(output)
    metadata = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or metadata.exists():
        raise FileExistsError(output if output.exists() else metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with metadata.open("x") as handle:
        json.dump(
            {
                "source": source,
                "input_sha256": file_sha256(input_path),
                "model": config["model"]["name"],
                "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or config["model"]["revision"],
                "task": task,
                "samples": len(rows),
            },
            handle,
            indent=2,
        )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
