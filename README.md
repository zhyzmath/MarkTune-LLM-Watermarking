# MarkTune: Improving the Quality-Detectability Trade-off in Model-Embedded LLM Watermarking

[![arXiv](https://img.shields.io/badge/arXiv-2512.04044-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2512.04044)

MarkTune initializes an actor with an RMS-scaled Gaussian watermark and optimizes it with GRPO and cross-entropy on human text. The watermark reward and detector share a fixed clean reference model. Generation uses standard Hugging Face decoding.

```text
configs/                 Model-specific YAML recipes
src/marktune/            Watermarking, data preparation, training, and CLI
src/marktune/baselines/  Inference-time and distilled watermark detectors
src/marktune/evaluation/ Generation, metrics, attacks, downstream tasks, timing
third_party/verl/        CE/GRPO training backend and reward/dataset adapters
tests/                  CPU regression tests
```

## Installation

Use Python 3.10–3.12. Training requires Linux, CUDA, and four GPUs for the supplied recipes.

```bash
pip install torch==2.6.0
pip install -r requirements/train.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e '.[data,evaluation,attacks]' -e third_party/verl
```

The training stack pairs [vLLM 0.8.5 with PyTorch 2.6.0](https://github.com/vllm-project/vllm/blob/v0.8.5/requirements/cuda.txt) and [Transformers 4.51.3 with Qwen3 support](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen3).

For generation and detection alone, use `pip install -e .`. Add the `evaluation` extra for quality metrics and the `downstream` extra for LM Evaluation Harness. Llama model access requires the corresponding model-provider approval.

## Configuration

Edit `configs/qwen3.yaml` or `configs/llama2.yaml`, or pass repeatable typed overrides:

```bash
marktune train --config configs/qwen3.yaml --initialized outputs/qwen3-init \
  --output outputs/qwen3 --dry-run --set runtime.micro_batch_size=4
```

Paths are relative to the working directory. `--dry-run` prints the training configuration without loading models, accessing data, or starting workers. Existing output files and run directories are never reused by the public commands.

| Setting | Qwen3-4B-Base | Llama2-7B |
| --- | --- | --- |
| Watermarked block | layer 28 MLP up-projection | layer 30 MLP up-projection |
| MarkTune initialization σ | 0.6 | 0.8 |
| GaussMark baseline σ | 1.0 | 1.2 |
| Training steps | 200 | 300 |
| CE coefficient | 0.01 | 0.005 |
| Peak learning rate | 5e-6 | 4e-6 |
| Linear warmup steps | 20 | 30 |
| Sampling temperature | 0.7 | 1.0 |

Both recipes use AdamW with β=(0.9, 0.97), 64 prompts per batch, 8 responses per prompt, 3 GRPO inner steps, mini-batches of 64 prompts, and 256-token rollouts. Each update also uses a 128-document human-text CE batch truncated to 512 tokens. All actor parameters remain trainable. After warmup, cosine decay spans 40% of the total training steps; the learning rate remains at `min_learning_rate_ratio` afterwards. Its default is zero.

The quality regularizer uses OpenWebText, independently of the stronger oracle used to evaluate PPL. Evaluation defaults to 500 prompts and 200 response tokens, with prompt-agnostic detection, AUC, TPR at 1% FPR, and 500 paired bootstrap resamples at 90% confidence.

## Data

Obtain [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), [C4 RealNewsLike](https://huggingface.co/datasets/allenai/c4), and [ELI5](https://github.com/facebookresearch/ELI5) separately. Record the dataset revision and split when preparing each input. Local inputs are JSONL or Parquet:

- OpenWebText and C4: `{"text": "..."}`.
- ELI5: `{"question": "...", "answer": "..."}`, with one selected long-form answer per question.

Choose the training, validation, and attack document counts explicitly. For example:

```bash
marktune prepare-data --config configs/qwen3.yaml --input openwebtext.parquet \
  --source 'OpenWebText, train, YOUR_DATASET_REVISION' \
  --train-documents 20000 --validation-documents 1000 --attack-documents 20000

marktune prepare-eval --config configs/qwen3.yaml --task c4 \
  --input c4-realnewslike.parquet --output data/qwen3/c4.jsonl \
  --source 'C4 realnewslike, validation, YOUR_DATASET_REVISION'

marktune prepare-eval --config configs/qwen3.yaml --task eli5 \
  --input eli5.jsonl --output data/qwen3/eli5.jsonl \
  --source 'ELI5, test, YOUR_DATASET_REVISION'
```

Complete OpenWebText documents are deduplicated before seeded assignment to the three disjoint splits. Training prompts contain 64–256 tokens. Manifests retain input hashes, document identities, counts, seeds, and tokenizer revisions.

C4 inputs must exceed 400 tokens: the first 200 form the prompt and the next 200 the human reference. ELI5 answers must exceed 200 tokens and are truncated to 200. Each model needs its own tokenized preparation. Keep evaluation corpora separate from all training inputs.

## Training and model export

```bash
marktune initialize --config configs/qwen3.yaml --output outputs/qwen3-init
marktune train --config configs/qwen3.yaml --initialized outputs/qwen3-init \
  --output outputs/qwen3

marktune export --checkpoint outputs/qwen3/global_step_200/actor \
  --output outputs/qwen3-model
```

Initialization writes an `actor/` checkpoint, `watermark-key.pt`, and a manifest identifying the clean reference and its resolved revision. Retain the exact key; random seeds alone do not identify Gaussian draws across devices and dtypes. The key is needed for both reward computation and detection and should remain private.

The training output contains the resolved recipe, backend configuration, and checkpoints. Its bundled `verl` runtime implements the joint CE/GRPO update and the GPU gradient reward.

## Generation, detection, and quality

```bash
marktune generate --config configs/qwen3.yaml --method clean \
  --input data/qwen3/c4.jsonl --output outputs/qwen3-clean.jsonl
marktune generate --config configs/qwen3.yaml --method marktune \
  --model outputs/qwen3-model --input data/qwen3/c4.jsonl \
  --output outputs/qwen3-watermarked.jsonl
marktune detect --config configs/qwen3.yaml --method marktune \
  --key outputs/qwen3-init/watermark-key.pt \
  --input outputs/qwen3-watermarked.jsonl --clean-input outputs/qwen3-clean.jsonl \
  --output outputs/qwen3-scored.jsonl
marktune metrics --config configs/qwen3.yaml --input outputs/qwen3-scored.jsonl \
  --quality --mauve --output outputs/qwen3-metrics.json
```

Use the same prepared prompts and clean generations for each comparison. Detection scores only visible response text. The clean scores are paired by sample identity and prompt. Reports distinguish interpolated ROC TPR from attainable empirical ROC points and include the null sample count. PPL is measured on response tokens conditioned on the prompt; Seq-rep-3 averages duplicate 3-gram rates across responses; MAUVE compares responses with their human references.

The generation and detection methods are `marktune`, `gaussmark`, `kgw-1`, `kgw-2`, `gumbel`, `synthid`, and `kgw-d`. For GaussMark, initialize with `--method gaussmark` and generate from its `actor/` directory. KGW-1/2 use γ=0.5, δ=1/2 and a four-token prefix. Gumbel-max and SynthID also use four-token prefixes; SynthID uses 30 tournaments. Each response starts with a fresh sampler state.

KGW-D requires a distilled checkpoint with matching tokenizer, teacher PRF, and teacher hash key. The Llama recipe supplies the public one-token-prefix KGW-D checkpoint (γ=0.25, δ=1). Its detector follows the [released teacher implementation](https://github.com/chenchenygu/watermark-learnability/tree/f84f70f9fc30eba5e39bf1889376d61de5f2ee59): CPU `simple_1` green lists, key 15485863, and special-token exclusions. Teacher settings are under `baselines.kgw_distilled`. For other compatible distilled checkpoints, set `baselines.kgw_distilled_model` or pass `--model`.

The trade-off sweeps use these values with the same initialization, training, generation, and metric commands:

| Variable | Qwen3 | Llama2 |
| --- | --- | --- |
| KGW δ | 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5 | same |
| GaussMark σ | 0.6, 0.8, 1.0, 1.1, 1.2, 1.5, 1.8 | 0.8, 1.0, 1.2, 1.3, 1.4, 1.6, 2.0 |
| MarkTune training steps | 50, 100, 150, 200, 250, 300, 500 | 50, 100, 200, 300, 400, 500, 600 |

Override `baselines.kgw.delta` with `--method kgw-1` for a KGW sweep; `kgw-2` always uses δ=2. Override `watermark.gaussmark_sigma` for Gaussian initializations or `training.total_steps` for training runs. Use a separate output path for every setting.

## Text-editing attacks

```bash
python -m nltk.downloader wordnet punkt punkt_tab
marktune attack-text --config configs/qwen3.yaml --input outputs/qwen3-watermarked.jsonl \
  --attack delete --ratio 0.2 --output outputs/qwen3-delete-02.jsonl
marktune attack-text --config configs/qwen3.yaml --input outputs/qwen3-watermarked.jsonl \
  --attack substitute --ratio 0.5 --output outputs/qwen3-substitute-05.jsonl
marktune attack-text --config configs/qwen3.yaml --input outputs/qwen3-watermarked.jsonl \
  --attack dipper --diversity 20 --output outputs/qwen3-dipper-20.jsonl
marktune attack-text --config configs/qwen3.yaml --input outputs/qwen3-watermarked.jsonl \
  --attack translate --output outputs/qwen3-translate.jsonl
```

Evaluate deletion/substitution at 0.2 and 0.5, and DIPPER lexical/order diversity at 20 and 60. Ratios refer to words; each output also records realized tokenizer-level insertion, deletion, and substitution counts. Translation runs English→French→English with MarianMT. DIPPER and translation obtain their pretrained attack models when invoked. Score attacked responses with `detect`, using the same clean null set and watermark key, then run `metrics`.

## LoRA fine-tuning attack

```bash
marktune attack-lora --config configs/qwen3.yaml --model outputs/qwen3-model \
  --input data/qwen3/attack-train.parquet --output outputs/qwen3-lora
```

The attack uses the disjoint OpenWebText split, 1500 AdamW steps, peak LR 1e-5, 300 warmup steps, cosine decay, batch size 64, and sequence length 512. LoRA rank/alpha are 8/16, applied to MLP projections and `lm_head`. Adapter checkpoints are saved every 100 steps; the final output also contains `adapter/` and `merged/`. Evaluate intermediate adapters or the merged model using the original key and clean reference. Run this command under `accelerate launch -m marktune` to use multiple GPUs, keeping the effective batch size fixed.

## Downstream tasks and latency

```bash
pip install -e '.[downstream]'
marktune downstream --config configs/qwen3.yaml --model outputs/qwen3-model \
  --tasks mmlu,gsm8k --output outputs/qwen3-downstream
marktune benchmark --config configs/qwen3.yaml --method marktune \
  --model outputs/qwen3-model --key outputs/qwen3-init/watermark-key.pt \
  --input data/qwen3/c4.jsonl --output outputs/qwen3-latency.json
```

To include MBPP, use `--tasks mmlu,gsm8k,mbpp --allow-code-execution` in an isolated execution environment. Run all compared checkpoints with the same LM Evaluation Harness version and task settings. The wrapper uses 5-shot MMLU and the installed task definitions for GSM8K and MBPP.

Latency uses batch size one, the first 100 C4 prompts with 200-token prefixes, 200-token responses, and two additional warmup prompts. CUDA synchronization brackets each timed operation. Generation/detection times and individual measurements are written as JSON.

## License

Third-party attribution and licensing are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

MarkTune is released under the [MIT License](LICENSE). Bundled third-party components and derived baseline modules retain their Apache-2.0 licenses.
