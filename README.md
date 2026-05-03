<div align="center">

# PonderLM-2

### Pretraining Language Models with Latent Thoughts

[![Paper](https://img.shields.io/badge/arXiv-2509.23184-b31b1b.svg)](https://arxiv.org/abs/2509.23184)
[![ICML 2026](https://img.shields.io/badge/ICML-2026%20Spotlight-1d4ed8.svg)](https://icml.cc)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![HF Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoints-orange)](https://huggingface.co/zeng123)

</div>

> **TL;DR.** Chain-of-Thought scales test-time compute by generating extra
> *tokens*. PonderLM-2 does the same at **pretraining** time, but in
> **continuous space**: before predicting each next token the model first
> emits a few **latent thoughts** — extra last-hidden-state vectors — and
> feeds them back into itself. Result: **PonderLM-2-Pythia-1.4B (300 B Pile
> tokens) beats vanilla Pythia-2.8B at equal inference flops**, on language
> modeling and a range of downstream tasks. More latent thoughts, lower
> loss.

```
   vanilla:      x₁ ──► x₂ ──► x₃ ──► x₄

   PonderLM-2:   x₁ ──► z₁ ──► x₂ ──► z₂ ──► x₃ ──► z₃ ──► x₄ ──► z₄
                       z_i = latent thought emitted before predicting x_{i+1}
```

---

## What's in this repo

A slim fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
covering Llama and GPT-NeoX. PonderLM-2, three baselines, and a vanilla
reference all share the same CLI and switch with one flag:

| `--patch_method` | Method | Reference |
|---|---|---|
| `ours`     | **PonderLM-2** (this paper)             | Zeng et al., ICML 2026 |
| `ponderlm` | PonderLM                                | [LUMIA-Group/PonderingLM](https://github.com/LUMIA-Group/PonderingLM) |
| `loop`     | Loop (weight-shared layer stack)        | folklore |
| `pause`    | Pause Token                             | Goyal et al., 2024 |
| `vanilla`  | Stock Llama / GPT-NeoX                  | — |

Released checkpoints on the Hugging Face Hub:

- 🤗 [`zeng123/PonderLM-2-Pythia-1.4b`](https://huggingface.co/zeng123/PonderLM-2-Pythia-1.4b)
- 🤗 [`zeng123/PonderLM-2-Pythia-410m`](https://huggingface.co/zeng123/PonderLM-2-Pythia-410m)

---

## Install

```bash
git clone https://github.com/hzby/PonderLM-2.git
cd PonderLM-2
conda create -n ponderlm2 python=3.10 -y
conda activate ponderlm2
pip install -e ".[torch,metrics,deepspeed]"
pip install wandb
conda install cuda=12.1.0 -c nvidia -y
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

Tested on PyTorch 2.4 + CUDA 12.1, 8×A100 / 8×H100 / 8×RTX4090.

---

## Quickstart — Pythia-70m on minipile

A small pretraining run on 8 GPUs to confirm the install works before
committing to the larger Table 3 runs.

```bash
bash scripts/quickstart_ponderlm2_pythia_70m.sh
```

Behind the scenes:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
    --model_name_or_path EleutherAI/pythia-70m \
    --stage pt --do_train --finetuning_type full --train_from_scratch true \
    --dataset minipile --template default --cutoff_len 2048 \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
    --learning_rate 1.0e-3 --weight_decay 0.01 --warmup_ratio 0.01 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --bf16 --flash_attn fa2 --disable_gradient_checkpointing true \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --patch_method ours --recurrent_model true --scale_embeds true \
    --interpolation false \
    --num_latent_thoughts 1 \
    --num_jacobi_iterations 3 \
    --random_jacobi_iterations true \
    --output_dir saves/minipile/ponderlm2_pythia_70m
```

For a vanilla baseline next to it, replace `--patch_method ours` with
`--patch_method vanilla` and drop `--recurrent_model true`; the rest of the
ponder flags become no-ops. **Side by side, PonderLM-2's training loss is
visibly below the vanilla curve from very early on.**

---

## Reproducing Table 3 (Llama-1.4b on smallpile)

Table 3 of the paper compares PonderLM-2 against Pause Token, Loop, and
PonderLM, all 1.4 B-parameter Llamas trained on **smallpile** for 50 000
steps at global batch 256, seq-len 2048 (≈ 26 B tokens) on an 8-GPU node.
The repo ships one script per row, plus a vanilla reference.

The corpus is pre-tokenised as `uint16` and lives at
[`hyq718/uint16smallpile`](https://huggingface.co/datasets/hyq718/uint16smallpile):

```bash
mkdir -p data/tokenized_data
huggingface-cli download hyq718/uint16smallpile \
    --repo-type dataset --local-dir data/tokenized_data/smallpile
```

Then launch one row at a time:

```bash
bash scripts/train_ponderlm2_llama_1_4b.sh   # ours
bash scripts/train_ponderlm_llama_1_4b.sh    # PonderLM
bash scripts/train_loop_llama_1_4b.sh        # Loop
bash scripts/train_pause_llama_1_4b.sh       # Pause Token
bash scripts/train_vanilla_llama_1_4b.sh     # vanilla reference
```

Each script accepts three environment variables:

```bash
MODEL_CFG=llama_config/410m \
TOKENIZED=data/tokenized_data/smallpile \
OUTPUT_DIR=runs/ponderlm2_llama_410m \
bash scripts/train_ponderlm2_llama_1_4b.sh
```

For Pythia / GPT-NeoX, point `MODEL_CFG` at `pythia_config/410m` (or `1.4b`).
Multi-node works out of the box through the LLaMA-Factory launcher — just
prefix with `torchrun --nnodes ...` or your scheduler's launcher.

---

## Hyperparameters

| Flag | Default | Method | Meaning |
|---|---|---|---|
| `--patch_method`               | unset | all       | Which modeling implementation to swap in. |
| `--recurrent_model`            | False | all       | Turn pondering on. Always `true` except for `vanilla`. |
| `--num_latent_thoughts`        | 0     | all       | For `ours`: latent-thought iterations per token (*K* in the paper). For `loop`: extra full passes through the layer stack. For `pause`: number of pause tokens after each input token. For `ponderlm`: number of pondering forward passes. |
| `--num_jacobi_iterations`      | 3     | ours      | Number of Jacobi-iteration steps used to solve the latent thoughts. Jacobi lets all positions update in parallel each step instead of strictly left-to-right; a few steps are enough to converge. |
| `--random_jacobi_iterations`   | False | ours      | At training time, sample the iteration count uniformly each step (curriculum used in the paper). |
| `--scale_embeds`               | False | all       | Multiply token embeddings by √d before pondering. Recommended on. |
| `--interpolation`              | True  | ours / ponderlm | Map hidden → embedding via a softmax over the vocab. PonderLM-2 turns this **off**; PonderLM keeps it on. |
| `--train_from_scratch`         | False | all       | Random-init from the model config rather than loading pretrained weights. |

Everything else (optimiser, scheduler, DeepSpeed, FlashAttention, data flags)
is plain LLaMA-Factory; see the scripts for the values used in the paper.

---

## Inference

The released GPT-NeoX checkpoints are wired up via `auto_map`, so a normal
`from_pretrained` does the right thing:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "zeng123/PonderLM-2-Pythia-1.4b"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
).cuda()

prompt = "The mitochondria is "
out = model.generate(
    **tok(prompt, return_tensors="pt").to(model.device),
    max_new_tokens=64, use_cache=True,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

For Llama checkpoints trained from this repo, patch the class first:

```python
from llamafactory.model.llama_patch import patch_llama_ponderlm2
patch_llama_ponderlm2()
# from_pretrained as usual
```

The Llama path supports the standard KV-cache, so `model.generate` works
the same way as for any HuggingFace transformer.

---

## Repository tour

```
PonderLM-2/
├── scripts/                      # one script per row of Table 3 + quickstart
├── llama_config/                 # Llama configs (125m, 410m, 834m, 1.4b, 2.8b)
├── pythia_config/                # Pythia configs (410m)
├── examples/deepspeed/           # ZeRO configs used in the paper
├── data/dataset_info.json        # minipile, smallpile, testpile, uint16smallpile
└── src/llamafactory/model/
    ├── llama_patch.py                                   # method swap entry point
    └── modeling/
        ├── modeling_llama.py                            # PonderLM-2 (Llama, w/ KV-cache)
        ├── modeling_llama_orin.py                       # PonderLM   (Llama)
        ├── modeling_llama_loop.py                       # Loop       (Llama)
        ├── modeling_llama_pause.py                      # Pause      (Llama)
        ├── modeling_gpt_neox.py                         # PonderLM-2 (GPT-NeoX)
        ├── modeling_gpt_neox_addhidden.py               # PonderLM   (GPT-NeoX)
        ├── modeling_gpt_neox_addhidden_weightshare.py   # Loop       (GPT-NeoX)
        └── modeling_gpt_neox_addpausetoken.py           # Pause      (GPT-NeoX)
```

---

## Citation

```bibtex
@inproceedings{zeng2026ponderlm2,
  title     = {{PonderLM-2}: Pretraining Language Models with Latent Thoughts},
  author    = {Zeng, Boyi and others},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {Spotlight}
}
```

## Acknowledgements

Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
(Apache-2.0). The PonderLM baseline is adapted from
[LUMIA-Group/PonderingLM](https://github.com/LUMIA-Group/PonderingLM).

## Contact

Open a GitHub issue, or email
[boyizeng@sjtu.edu.cn](mailto:boyizeng@sjtu.edu.cn).
