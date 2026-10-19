# Third-party software

| Component | Source | License | Included scope |
| --- | --- | --- | --- |
| verl | [volcengine/verl](https://github.com/volcengine/verl/tree/b8c6d132a8a6a4b3152851b5deafabf001ac3339) | Apache-2.0 | FSDP/GRPO and vLLM runtime, with MarkTune CE loss, gradient reward, text dataset, and learning-rate schedule integration |
| MarkLLM | [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM/tree/112fa7419b115efc82f229a59a86fb14c5528c80) | Apache-2.0 | KGW, exponential/Gumbel sampling, and SynthID algorithms adapted in `src/marktune/baselines/` |
| KGW-D teacher | [chenchenygu/watermark-learnability](https://github.com/chenchenygu/watermark-learnability/tree/f84f70f9fc30eba5e39bf1889376d61de5f2ee59) | Apache-2.0 | One-token `simple_1` detector matching the released KGW-D teacher; copyright and source notice in `src/marktune/baselines/kgw_distilled.py` |

License texts are in `third_party/verl/LICENSE` and `third_party/markllm/LICENSE`. Original copyright notices are retained in derived source files.

The main `marktune` source is licensed under the [MIT License](LICENSE). The derived baseline modules and bundled `verl` retain their Apache-2.0 terms. Model weights and datasets are obtained separately under their respective licenses and access conditions.
