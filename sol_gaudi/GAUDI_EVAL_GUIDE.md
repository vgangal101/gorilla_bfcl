# Gaudi2 reference notes (ASU SOL)

Companion to `README.md`. The README covers **how to run** the BFCL sweep; this file covers **what the hardware is, what the ASU workshop deck documented, and a shortcut path that skips SLURM entirely**. If you're here to submit a job, use `README.md` + `quickstart.sh`.

## Hardware on SOL

- **10 Gaudi2 nodes:** `gaudi001`–`gaudi010` (part of a larger 200-node / 1600-HPU Intel donation; only 10 are wired into SOL today).
- **Per node:** 8 HPUs, 152 CPUs, 96 GB HBM per HPU.
- AIP devices appear in `hl-smi` as `HL-225` and report `98304 MiB` per card.
- Also on SOL (NVIDIA, for reference): 61 nodes × A100 = 254 A100s.
- `hl-smi` is the Habana analogue of `nvidia-smi` — use it inside allocations to confirm HPU health before launching vLLM.

## Authoritative SLURM values

These are what `config.env.example` ships with — don't rederive them from older NVIDIA scripts or the workshop slide deck:

| Field | Value |
|---|---|
| `SLURM_ACCOUNT` | `class_cse59827694spring2026` |
| `SLURM_QOS` | `class_gaudi` (the deck shows `public` — that's the general-access QoS, not ours) |
| `SLURM_PARTITION` | `gaudi` |
| `SLURM_GRES_PREFIX` | `gpu:hl225` (e.g. `--gres=gpu:hl225:4`) |

## Shared assets on SOL

- `/data/sse/gaudi` — container images and ASU RC guides.
- `/data/sse/gaudi/guides` — walkthroughs.
- `/data/sse/gaudi/notebooks` — MNIST in lazy and eager modes (useful sanity check when Habana env breaks).
- `/data/sse/gaudi/diffusion` — Optimum-Habana diffusion examples.
- **JupyterHub kernel:** `gaudi-pytorch` (pre-built SynapseAI PyTorch env; good for interactive debugging before committing to SLURM).

## PyTorch on HPU (if you ever need to patch BFCL itself)

Three code deltas relative to a CUDA script:

```python
import habana_frameworks.torch.core as htcore     # loads Habana op kernels
device = torch.device("hpu")                       # was "cuda"
# In Lazy mode only (we default Eager via PT_HPU_LAZY_MODE=0), after
# loss.backward() and optimizer.step():
htcore.mark_step()
```

For drop-in GPU→HPU conversion of existing `.cuda()` / `torch.device("cuda")` code, set `PT_HPU_GPU_MIGRATION=1` and import `habana_frameworks`; Synapse's GPU Migration Toolkit rewrites CUDA calls to HPU.

The existing sweep runs BFCL **unmodified** (just `--skip-server-setup` against the Apptainer-wrapped vLLM server), so you shouldn't need these changes unless you're extending BFCL itself.

## LLM support — what the workshop deck explicitly listed

The ASU Gaudi2 workshop deck ("Introducing the Gaudi2 and Demystifying AI Processors", Jeevesh Choudhury) explicitly names these as vLLM-on-Gaudi supported: **DeepSeek-R1, Llama 3.x, Qwen 2.5, Qwen/Qwen2.5-VL-7B-Instruct**.

- **Qwen3-32B is not in that list.** Qwen3 shares the Qwen2 architecture family so HabanaAI's vLLM fork generally handles it, but treat the smoke test (`manage_bfcl_gaudi.sh submit qwen3_4b`) as gating before submitting the full sweep — if Qwen3-4B works, 8B/14B/32B will almost certainly follow.
- **Gemma 4** (`google/gemma-4-31B-it`) requires the BFCL patch noted in `README.md:92` (`supported_models.py` + `model_config.py` entries). The Habana vLLM fork's handling of Gemma 4 is also unverified by the deck.

