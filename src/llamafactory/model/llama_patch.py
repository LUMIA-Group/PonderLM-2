# Copyright 2024 the LlamaFactory team and the PonderLM-2 authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Monkey-patch transformers' built-in Llama / GPT-NeoX model classes with
PonderLM-2 (`ours`) and the three baselines (`pause`, `loop`, `ponderlm`).

PonderLM-2 method ↔ patch_method ↔ modeling file mapping
--------------------------------------------------------
| method     | --patch_method | Llama file                   | GPT-NeoX file                                  |
|------------|----------------|------------------------------|------------------------------------------------|
| ours       | (default)      | modeling_llama.py            | modeling_gpt_neox.py                           |
| ponderlm   | ``ponderlm``   | modeling_llama_orin.py       | modeling_gpt_neox_addhidden.py                 |
| loop       | ``loop``       | modeling_llama_loop.py       | modeling_gpt_neox_addhidden_weightshare.py     |
| pause      | ``pause``      | modeling_llama_pause.py      | modeling_gpt_neox_addpausetoken.py             |
| vanilla    | ``vanilla`` (or unset) | (transformers stock)         | (transformers stock)                           |

The vanilla path leaves the stock ``transformers`` implementation in place.
``ponderlm`` reuses the ``orin`` modeling files with ``--recurrent_model true``
(softmax-weighted-embedding pondering, the algorithm of the original PonderLM
paper).
"""

from .modeling.modeling_llama import LlamaForCausalLM as PonderLM2LlamaForCausalLM
from .modeling.modeling_llama_orin import LlamaForCausalLM as PonderLMLlamaForCausalLM
from .modeling.modeling_llama_loop import LlamaForCausalLM as LoopLlamaForCausalLM
from .modeling.modeling_llama_pause import LlamaForCausalLM as PauseLlamaForCausalLM

from .modeling.modeling_gpt_neox import GPTNeoXForCausalLM as PonderLM2GPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_addhidden import GPTNeoXForCausalLM as PonderLMGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_addhidden_weightshare import GPTNeoXForCausalLM as LoopGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_addpausetoken import GPTNeoXForCausalLM as PauseGPTNeoXForCausalLM


def _swap_llama(cls):
    import transformers.models.llama.modeling_llama as m
    m.LlamaForCausalLM = cls


def _swap_gpt_neox(cls):
    import transformers.models.gpt_neox.modeling_gpt_neox as m
    m.GPTNeoXForCausalLM = cls


def patch_llama_ponderlm2() -> None:
    """Use the PonderLM-2 Llama implementation (ours)."""
    _swap_llama(PonderLM2LlamaForCausalLM)


def patch_llama_ponderlm() -> None:
    """Use the original PonderLM Llama implementation (baseline)."""
    _swap_llama(PonderLMLlamaForCausalLM)


def patch_llama_loop() -> None:
    """Use the looped Llama implementation (baseline)."""
    _swap_llama(LoopLlamaForCausalLM)


def patch_llama_pause() -> None:
    """Use the pause-token Llama implementation (baseline)."""
    _swap_llama(PauseLlamaForCausalLM)


def patch_gpt_neox_ponderlm2() -> None:
    """Use the PonderLM-2 GPT-NeoX implementation (ours)."""
    _swap_gpt_neox(PonderLM2GPTNeoXForCausalLM)


def patch_gpt_neox_ponderlm() -> None:
    """Use the original PonderLM GPT-NeoX implementation (baseline)."""
    _swap_gpt_neox(PonderLMGPTNeoXForCausalLM)


def patch_gpt_neox_loop() -> None:
    """Use the looped GPT-NeoX implementation (baseline)."""
    _swap_gpt_neox(LoopGPTNeoXForCausalLM)


def patch_gpt_neox_pause() -> None:
    """Use the pause-token GPT-NeoX implementation (baseline)."""
    _swap_gpt_neox(PauseGPTNeoXForCausalLM)
