"""LoRA removal attack on the disjoint OpenWebText attack split."""

from pathlib import Path


def training_options(config, *, world_size=1, micro_batch_size=1):
    options = config["attacks"]["lora"]
    denominator = world_size * micro_batch_size
    if denominator < 1 or options["batch_size"] % denominator:
        raise ValueError("LoRA batch size must be divisible by world size times micro batch size")
    return dict(
        max_steps=options["steps"],
        learning_rate=options["learning_rate"],
        warmup_steps=options["warmup_steps"],
        lr_scheduler_type="cosine",
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=options["batch_size"] // denominator,
        save_strategy="steps",
        save_steps=options["save_steps"],
        optim="adamw_torch",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


def reserve_output(output, state):
    """Reserve a shared output directory once and propagate failures to all ranks."""
    import torch

    status = [None]
    if state.is_main_process:
        try:
            Path(output).mkdir(parents=True, exist_ok=False)
        except OSError as error:
            status[0] = (type(error).__name__, str(error))
    if state.num_processes > 1:
        torch.distributed.broadcast_object_list(status, src=0)
    if status[0]:
        name, message = status[0]
        if name == "FileExistsError":
            raise FileExistsError(message)
        raise OSError(message)


def run(args, config):
    import torch
    from accelerate import PartialState
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    from .io import write_json
    from .generation import seed_everything

    output = Path(args.output)
    state = PartialState(cpu=config["runtime"]["device"] == "cpu")
    reserve_output(output, state)
    seed_everything(config["evaluation"]["seed"])
    data_path = Path(args.input or Path(config["data"]["directory"]) / "attack-train.parquet")
    options = config["attacks"]["lora"]
    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or config["model"]["name"],
        trust_remote_code=False,
        revision=config["model"].get("revision") if not args.tokenizer else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=options["sequence_length"], padding="max_length")
        encoded["labels"] = [
            [token if valid else -100 for token, valid in zip(ids, mask)]
            for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
        ]
        return encoded

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    local_rank = state.local_process_index
    world_size = state.num_processes
    if config["runtime"]["device"].startswith("cuda"):
        torch.cuda.set_device(local_rank)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        trust_remote_code=False,
        revision=config["model"].get("revision") if args.model == config["model"]["name"] else None,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=options["rank"],
            lora_alpha=options["alpha"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=options["target_modules"],
        ),
    )
    kwargs = training_options(config, world_size=world_size, micro_batch_size=args.micro_batch_size)
    kwargs.update(
        output_dir=str(output),
        seed=config["evaluation"]["seed"],
        data_seed=config["data"]["seed"],
        bf16=config["model"]["dtype"] == "bfloat16",
        fp16=config["model"]["dtype"] == "float16",
        use_cpu=config["runtime"]["device"] == "cpu",
    )
    if state.is_main_process:
        write_json(
            output / "attack.json",
            {"model": args.model, "data": str(data_path), "lora": options, "trainer": kwargs, "world_size": world_size},
        )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "adapter")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output / "merged")
        tokenizer.save_pretrained(output / "merged")
