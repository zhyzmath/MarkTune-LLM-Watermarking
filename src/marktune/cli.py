"""Command-line interface for MarkTune training and evaluation."""

import argparse

from .config import add_config_arguments, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="marktune", description="MarkTune watermarking and paper evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize", help="Create a Gaussian-initialized Hugging Face actor and key")
    add_config_arguments(initialize)
    initialize.add_argument("--output", required=True)
    initialize.add_argument("--method", choices=("marktune", "gaussmark"), default="marktune")
    initialize.set_defaults(func=_initialize)
    train = subparsers.add_parser("train", help="Run CE-regularized GRPO")
    add_config_arguments(train)
    train.add_argument("--initialized", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--dry-run", action="store_true", help="Print the verl overlay without loading models or data")
    train.set_defaults(func=_train)
    prepare = subparsers.add_parser("prepare-data", help="Prepare disjoint OpenWebText splits from local documents")
    add_config_arguments(prepare)
    prepare.add_argument("--input", required=True, help="Parquet/JSONL with a text column")
    prepare.add_argument("--source", required=True, help="Dataset name, split and revision")
    prepare.add_argument("--train-documents", required=True, type=int)
    prepare.add_argument("--validation-documents", required=True, type=int)
    prepare.add_argument("--attack-documents", required=True, type=int)
    prepare.set_defaults(func=_prepare_data)
    evaluation = subparsers.add_parser("prepare-eval", help="Prepare C4 or ELI5 prompts and human references")
    add_config_arguments(evaluation)
    evaluation.add_argument("--input", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--source", required=True, help="Dataset name, split and revision")
    evaluation.add_argument("--task", required=True, choices=("c4", "eli5"))
    evaluation.set_defaults(func=_prepare_eval)
    export = subparsers.add_parser("export", help="Export a verl FSDP actor checkpoint to Hugging Face")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=_export)
    from .evaluation.cli import add_subparsers

    add_subparsers(subparsers)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"marktune: {error}\n")


def _initialize(args):
    from .training import initialize_actor

    initialize_actor(load_config(args.config, args.set), args.output, args.method)


def _train(args):
    from .training import run_training

    run_training(load_config(args.config, args.set), args.initialized, args.output, args.dry_run)


def _prepare_data(args):
    from .data import prepare_training_data

    counts = {
        "train": args.train_documents,
        "validation": args.validation_documents,
        "attack-train": args.attack_documents,
    }
    prepare_training_data(load_config(args.config, args.set), args.input, counts, args.source)


def _prepare_eval(args):
    from .data import prepare_evaluation_data

    prepare_evaluation_data(load_config(args.config, args.set), args.input, args.output, args.task, args.source)


def _export(args):
    from pathlib import Path
    import subprocess
    import sys

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            args.checkpoint,
            "--target_dir",
            str(output),
        ],
        check=True,
    )
