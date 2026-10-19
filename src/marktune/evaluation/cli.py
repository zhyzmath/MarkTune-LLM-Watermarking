"""Evaluation subcommands; heavyweight libraries load only when a command runs."""

from pathlib import Path

METHODS = ["marktune", "gaussmark", "kgw-1", "kgw-2", "gumbel", "synthid", "kgw-d"]


def add_subparsers(subparsers):
    from marktune.config import add_config_arguments

    def command(name, help_text, handler):
        parser = subparsers.add_parser(name, help=help_text)
        add_config_arguments(parser)
        parser.set_defaults(func=handler)
        return parser

    def models(parser, *, detection=False, benchmark=False):
        methods = [method for method in METHODS if method != "kgw-d"] if benchmark else METHODS
        parser.add_argument("--method", choices=methods if detection else ["clean"] + methods, default="marktune")
        parser.add_argument("--model", help="Generation model or checkpoint directory")
        parser.add_argument("--tokenizer", help="Tokenizer name; defaults to the recipe base model")
        if detection:
            parser.add_argument("--reference-model", help="Fixed clean reference stored in the watermark key")
            parser.add_argument("--key", help="Persisted Gaussian watermark key")

    parser = command("generate", "Generate response samples", run_generate)
    models(parser)
    parser.add_argument("--input", required=True, help="Prepared prompt/reference JSONL or Parquet")
    parser.add_argument("--output", required=True, help="New response JSONL")

    parser = command("detect", "Score visible response text without prompts", run_detect)
    models(parser, detection=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--clean-input", help="Clean generations with matched prompt IDs for null scores")
    parser.add_argument("--output", required=True)

    parser = command("metrics", "Compute detection and generation quality metrics", run_metrics)
    parser.add_argument("--input", required=True, help="Scored JSONL, with clean_score for detectability")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality", action="store_true", help="Load the oracle and compute response perplexity")
    parser.add_argument("--mauve", action="store_true", help="Compute MAUVE against human references")
    parser.add_argument("--oracle", help="Override the recipe oracle checkpoint")

    parser = command("attack-text", "Apply deletion, substitution, DIPPER, or roundtrip translation", run_attack_text)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--attack", choices=["delete", "substitute", "dipper", "translate"], required=True)
    parser.add_argument("--ratio", type=float, help="Word-level deletion probability or substitution fraction")
    parser.add_argument("--diversity", type=int, help="DIPPER lexical and order diversity (paper: 20 or 60)")

    parser = command("attack-lora", "Fine-tune with LoRA on the disjoint attack split", run_attack_lora)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--input", help="Prepared attack-train.parquet")
    parser.add_argument("--output", required=True, help="New adapter and merged-checkpoint directory")
    parser.add_argument("--micro-batch-size", type=int, default=1)

    parser = command("downstream", "Evaluate MMLU, GSM8K, and MBPP using lm_eval", run_downstream)
    parser.add_argument("--model")
    parser.add_argument("--tasks", default="mmlu,gsm8k", help="Comma-separated subset of mmlu,gsm8k,mbpp")
    parser.add_argument("--output", required=True, help="New evaluation directory")
    parser.add_argument(
        "--allow-code-execution", action="store_true", help="Enable MBPP execution in an isolated environment"
    )

    parser = command("benchmark", "Benchmark batch-one generation and detection latency", run_benchmark)
    models(parser, detection=True, benchmark=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)


def configuration(args, *, check_output=True):
    from marktune.config import load_config

    if check_output and Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    return load_config(args.config, args.set)


def run_generate(args):
    from .generation import run

    run(args, configuration(args))


def run_detect(args):
    from .detection import run

    run(args, configuration(args))


def run_attack_text(args):
    from .attacks import run

    run(args, configuration(args))


def run_attack_lora(args):
    from .lora import run

    run(args, configuration(args, check_output=False))


def run_downstream(args):
    from .downstream import run

    run(args, configuration(args))


def run_benchmark(args):
    from .benchmark import run

    run(args, configuration(args))


def run_metrics(args):
    from .generation import load_model_tokenizer
    from .io import read_records, write_json
    from .metrics import detection_metrics, quality_metrics

    config = configuration(args)
    records = list(read_records(args.input))
    oracle = tokenizer = None
    if args.quality:
        oracle, tokenizer = load_model_tokenizer(
            args.oracle or config["model"]["oracle"], config, args.oracle or config["model"]["oracle"]
        )
    result = {
        "quality": quality_metrics(
            records,
            oracle,
            tokenizer,
            mauve=args.mauve,
            device=config["runtime"]["device"],
            seed=config["evaluation"]["seed"],
        )
    }
    if any("clean_score" in row for row in records):
        if any("clean_score" not in row or "score" not in row for row in records):
            raise ValueError("Detection metrics require score and clean_score for every sample")
        settings = config["evaluation"]
        result["detection"] = detection_metrics(
            [row["clean_score"] for row in records],
            [row["score"] for row in records],
            fpr=settings["fpr"],
            bootstrap_samples=settings["bootstrap_samples"],
            confidence=settings["confidence"],
            seed=settings["seed"],
        )
    write_json(args.output, result)
