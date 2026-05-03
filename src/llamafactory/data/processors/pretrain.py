# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ...hparams import DataArguments


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    # If the dataset already provides ``input_ids`` (e.g. a pre-tokenised corpus
    # loaded via ``--tokenized_path``) just pass it through untouched.
    if "input_ids" in examples.keys() and "_prompt" not in examples.keys():
        return examples

    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

    if not data_args.packing:
        if data_args.template == "gemma":
            text_examples = [tokenizer.bos_token + example for example in text_examples]
        return tokenizer(text_examples, add_special_tokens=False, truncation=True, max_length=data_args.cutoff_len)

    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = data_args.cutoff_len
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if data_args.template == "gemma":
        for i in range(len(result["input_ids"])):
            result["input_ids"][i][0] = tokenizer.bos_token_id
    return result
