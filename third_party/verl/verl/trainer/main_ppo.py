# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed CE-regularized GRPO using the FSDP and synchronous vLLM backend."""

import ray
import torch
from omegaconf import OmegaConf
from transformers import set_seed

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.dataset.text_dataset import RLTextDataset
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def run_ppo(config):
    if config.actor_rollout_ref.actor.strategy != "fsdp" or config.actor_rollout_ref.rollout.mode != "sync":
        raise ValueError("MarkTune training requires FSDP and synchronous rollouts")
    if config.algorithm.adv_estimator != "grpo" or config.reward_model.enable:
        raise ValueError("MarkTune uses GRPO and its gradient reward manager")
    if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        raise ValueError("The main-body recipes use human-text CE regularization")
    if not ray.is_initialized():
        ray.init(runtime_env=get_ppo_ray_runtime_env())
    ray.get(TaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        OmegaConf.resolve(config)
        set_seed(config.data.seed)
        validate_config(config, use_reference_policy=False, use_critic=False)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=False)
        workers = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
        resources = ResourcePoolManager(
            resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping={Role.ActorRollout: "global_pool"},
        )
        reward = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.reward_kwargs)
        validation_reward = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.reward_kwargs)
        train_dataset = RLTextDataset(config.data.train_files, tokenizer, config.data)
        val_dataset = RLTextDataset(config.data.val_files, tokenizer, config.data)
        generator = torch.Generator().manual_seed(config.data.seed)
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
        trainer = RayPPOTrainer(
            config=config, tokenizer=tokenizer, processor=None,
            role_worker_mapping=workers, resource_pool_manager=resources,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward, val_reward_fn=validation_reward,
            train_dataset=train_dataset, val_dataset=val_dataset,
            collate_fn=collate_fn, train_sampler=sampler,
        )
        trainer.init_workers()
        reward.bind_actor_worker_group(trainer.actor_rollout_wg)
        validation_reward.bind_actor_worker_group(trainer.actor_rollout_wg)
        trainer.fit()
