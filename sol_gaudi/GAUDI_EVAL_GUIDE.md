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

| Field |/Users/hectorhernandez/Desktop/repo/gorilla_bfcl/sol_gaudi/GAUDI_EVAL_GUIDE.md Value |
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

## Shortcut: the Voyager managed API (no SLURM)

The deck's *API Access* slides show ASU runs an **OpenAI-compatible inference endpoint on Gaudi2**, managed via the Voyager portal. For any model that's already served there, you can skip SLURM entirely and hit the endpoint from a login node (or anywhere else).

- Portal: **https://voyager.rc.asu.edu/** → *LLM Access* tab → *Create Key*.
- Base URL: **`https://openai.rc.asu.edu/v1`**.
- Models shown in the deck screenshot (all tagged *Gaudi2*):

  | Model | Context |
  |---|---|
  | `llama4-scout-17b` | 68K |
  | `qwen3-coder-30b-a3b-instruct` | 131K |
  | `qwen3-30b-a3b-thinking-2507` | 131K |
  | `qwen3-30b-a3b-instruct-2507` | 131K |
  | `qwen3-235b-a22b-instruct-2507` | 262K |
  | `qwen3-235b-a22b-thinking-2507` | 262K |

These are Qwen3 **MoE** variants (30B-A3B, 235B-A22B), not the dense 32B we're benchmarking in this sweep. If the goal is "any capable Qwen3 on Gaudi2", they're a 15-minute path:

```bash
# in $BFCL_PROJECT_ROOT/.env
REMOTE_OPENAI_BASE_URL=https://openai.rc.asu.edu/v1
REMOTE_OPENAI_API_KEY=<key from Voyager>

bfcl generate \
  --model qwen3-30b-a3b-instruct-2507 \
  --test-category simple_python \
  --skip-server-setup --num-threads 2
bfcl evaluate \
  --model qwen3-30b-a3b-instruct-2507 \
  --test-category simple_python
```

Caveat: `qwen3-30b-a3b-instruct-2507` is not in BFCL's shipped `MODEL_CONFIG_MAPPING` — add an entry in `bfcl_eval/constants/model_config.py` pointing at an OpenAI-compatible handler before the first run.

## Decision: Voyager API vs. self-hosted vLLM

| Path | Effort | Use when |
|---|---|---|
| Voyager API (`https://openai.rc.asu.edu/v1`) | Low — Voyager key + one `MODEL_CONFIG_MAPPING` entry, no SLURM | Model you want is on the Voyager menu. Best for quick iteration. |
| Self-hosted sweep (`sol_gaudi/quickstart.sh`) | Higher — SIF build + SLURM queue + per-model scripts | Model isn't on the menu (Qwen3-32B dense, Gemma 4), or you want reproducible runs with pinned Habana/vLLM versions. |

The `sol_gaudi/` sweep targets dense Qwen3 sizes and Gemma 4 specifically because those aren't available via Voyager. For the MoE variants, use Voyager and save the HPU quota.
