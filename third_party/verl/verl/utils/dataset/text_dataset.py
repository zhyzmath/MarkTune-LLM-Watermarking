import os
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


class RLTextDataset(Dataset):
    """
    A minimal RL dataset that reads parquet with a plain text column (prompt_text)
    and tokenizes it directly with the target tokenizer (e.g., Llama-2 sentencepiece),
    avoiding chat templates for base models.

    Expected parquet columns:
      - prompt_text: str  (the raw prompt)
      - data_source: str  (optional; default 'owt')
      - ability: str      (optional; default 'generic')
      - reward_model: dict (optional; default {'style':'none','ground_truth': None})
      - extra_info: dict  (optional)
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        self.data_files = list(data_files)
        self.tokenizer = tokenizer
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt_text")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from verl.utils.fs import copy_to_local

        for i, parquet_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        frames = [datasets.load_dataset("parquet", data_files=f)["train"] for f in self.data_files]
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(frames)
        self.dataframe = self._maybe_filter_out_long_prompts(self.dataframe)

    def _maybe_filter_out_long_prompts(self, df: datasets.Dataset):
        if not self.filter_overlong_prompts:
            return df
        prompt_key = self.prompt_key
        tok = self.tokenizer

        def _len_ok(ex):
            text = ex.get(prompt_key, "") or ""
            ids = tok(text, add_special_tokens=False)["input_ids"]
            return len(ids) <= self.max_prompt_length

        df = df.filter(_len_ok, num_proc=self.num_workers, desc="Filtering overlong prompts")
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe[idx]
        text = row.get(self.prompt_key, "") or ""
        model_inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        out = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "data_source": row.get("data_source", "owt"),
            "ability": row.get("ability", "generic"),
            "reward_model": row.get("reward_model", {"style": "none", "ground_truth": None}),
            "extra_info": row.get("extra_info", {"index": idx}),
        }
        return out
