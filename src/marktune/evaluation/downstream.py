"""MMLU, GSM8K, and MBPP through an installed lm-evaluation-harness."""

import os
from pathlib import Path
import subprocess
import sys


def build_command(model, dtype, output, tasks, *, allow_code=False, seed=23333):
    if not tasks or not set(tasks) <= {"mmlu", "gsm8k", "mbpp"}:
        raise ValueError("Supported downstream tasks are mmlu, gsm8k, and mbpp")
    if "mbpp" in tasks and not allow_code:
        raise ValueError("MBPP executes generated code; pass --allow-code-execution in an isolated environment")
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model},dtype={dtype},trust_remote_code=False",
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        "auto:1",
        "--max_batch_size",
        "4",
        "--seed",
        f"{seed},{seed},{seed},{seed}",
        "--output_path",
        str(output),
    ]
    if tasks == ["mmlu"]:
        command += ["--num_fewshot", "5"]
    if "mbpp" in tasks:
        command += ["--confirm_run_unsafe_code"]
    return command


def run(args, config):
    output = Path(args.output)
    model = args.model or config["model"]["name"]
    tasks = args.tasks.split(",")
    environment = dict(os.environ)
    if "mbpp" in tasks:
        if not args.allow_code_execution:
            raise ValueError("MBPP requires --allow-code-execution")
        environment["HF_ALLOW_CODE_EVAL"] = "1"
    commands = [
        build_command(
            model,
            config["model"]["dtype"],
            output / task,
            [task],
            allow_code=args.allow_code_execution,
            seed=config["evaluation"]["seed"],
        )
        for task in tasks
    ]
    output.mkdir(parents=True, exist_ok=False)
    from .io import write_json

    write_json(
        output / "evaluation.json",
        {"model": model, "tasks": tasks, "commands": commands, "seed": config["evaluation"]["seed"]},
    )
    for command in commands:
        subprocess.run(command, env=environment, check=True)
