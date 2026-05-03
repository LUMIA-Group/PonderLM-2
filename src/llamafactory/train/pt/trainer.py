# Copyright 2024 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

# 添加必要的导入
if is_sagemaker_mp_enabled():
    import sagemaker_pytorch_training_toolkit.smp_utils as smp_utils
    from sagemaker_pytorch_training_toolkit.smp_utils import smp_forward_backward

try:
    from apex import amp
except ImportError:
    amp = None


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Fixes the loss value. See https://github.com/huggingface/transformers/pull/35438 for details.
        """
        inputs["global_step"] = self.state.global_step
        loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
        if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
            if return_outputs:
                loss = (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                loss = loss / self.args.gradient_accumulation_steps

        return loss

    # @override
    # def training_step(
    #     self, model: "PreTrainedModel", inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    # ) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.
    #     This override skips steps with NaN/Inf loss to prevent training crashes.
    #     """
    #     model.train()
    #     if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #         self.optimizer.train()

    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    #     # 立即检查原始损失是否为 NaN/Inf
    #     if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
    #         self.state.nan_loss_count = getattr(self.state, "nan_loss_count", 0) + 1
    #         if self.is_world_process_zero():
    #             print(
    #                 f"WARNING: Skipping step {self.state.global_step} due to NaN/Inf loss. "
    #                 f"Total skipped steps: {self.state.nan_loss_count}"
    #             )
            
    #         # 直接返回零损失，跳过所有后续处理
    #         return torch.tensor(0.0, device=self.args.device, dtype=torch.float32).detach()

    #     # 正常的训练步骤处理
    #     del inputs
    #     if (
    #         self.args.torch_empty_cache_steps is not None
    #         and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #     ):
    #         if is_torch_xpu_available():
    #             torch.xpu.empty_cache()
    #         elif is_torch_mlu_available():
    #             torch.mlu.empty_cache()
    #         elif is_torch_musa_available():
    #             torch.musa.empty_cache()
    #         elif is_torch_npu_available():
    #             torch.npu.empty_cache()
    #         elif is_torch_mps_available(min_version="2.0"):
    #             torch.mps.empty_cache()
    #         else:
    #             torch.cuda.empty_cache()

    #     kwargs = {}
    #     # For LOMO optimizers you need to explicitly use the learning rate
    #     if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         kwargs["learning_rate"] = self._get_learning_rate()

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     try:
    #         if self.use_apex:
    #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             loss *= self.args.gradient_accumulation_steps                       
    #             self.accelerator.backward(loss, **kwargs)
    #     except AssertionError as e:
    #         # if "all_groups_norm > 0" in str(e):
    #         self.state.nan_loss_count = getattr(self.state, "nan_loss_count", 0) + 1
    #         if self.is_world_process_zero():
    #             print(
    #                 f"WARNING: Skipping step {self.state.global_step} due to zero gradient norm (AssertionError). "
    #                 f"Total skipped steps: {self.state.nan_loss_count}"
    #             )
    #         return loss.detach() / self.args.gradient_accumulation_steps
    #         # else:
    #         #     raise e
    #     return loss.detach() / self.args.gradient_accumulation_steps