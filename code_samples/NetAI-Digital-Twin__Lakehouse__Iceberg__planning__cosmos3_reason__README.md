# cosmos3_reason — Cosmos3-Edge (4B) driving-difficulty axis

Detachable candidate axis for the Gold difficulty union, and a re-test of the
question `planning/alpamayo/FINDINGS.md` left open: the 10B VLM scorer was shelved
partly because it only fit on this hardware in a degraded 1-frame / 64-token
config. Cosmos3-Edge is 4.87 GB of reasoner weights instead of 22 GB, so the full
config is now affordable — see [FEASIBILITY.md](FEASIBILITY.md) for why this tier
and not the augmentation half.

## Detaching
Delete this directory. Nothing else references it: the base lakehouse and the
`nvidia_*` / `planning` runners have no import of it, frames are read straight off
the NFS mp4s (no `physical_ai_av`, no vendored model repo), and the module writes
only dotfiles inside its own directory. The venv and weights are gitignored.

## Setup
```bash
uv venv --python 3.12 c3_venv
VIRTUAL_ENV=$PWD/c3_venv uv pip install \
  "transformers @ git+https://github.com/huggingface/transformers.git@main" \
  torch torchvision accelerate opencv-python-headless pyarrow numpy
./c3_venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('nvidia/Cosmos3-Edge', ignore_patterns=['assets/*','images/*','vae/*','*.md'])"
```
Notes:
- `transformers` must come from **main** — `cosmos3_edge` is not in 5.14.1 stable
  (which ships `cosmos3_omni` only, for Nano/Super).
- `vae/` is skipped: it belongs to the diffusion tower, which this module never
  touches. Download is ~7.3 GB.
- The model is **not** gated; no access request needed.

## Files
| file | role |
|---|---|
| `cosmos3_scorer.py` | model load, NFS frame sampling, logit-EV scorers (`score_clip`, `reasoned_score`) |
| `gate_runner.py` | validity battery over the 452-clip OOD label set |
| `FEASIBILITY.md` | why Edge for reasoning and not for augmentation |
| `analyze_axis.py` | AUC / noisy-OR union / bootstrap-CI analysis vs the production axes; `--compare` for cross-backend residual analysis |
| `spot_review.py` | builds the blind human review set + self-contained `review.html` |
| `score_review.py` | unblinds `ratings.json` and runs the A-vs-B contrast |
| `RESULTS.md` | measured outcome vs the Alpamayo baseline |

## Usage
```bash
./c3_venv/bin/python gate_runner.py 452 --mode cold        # single forward per clip
./c3_venv/bin/python gate_runner.py 452 --mode reasoned    # rationale, then rate
./c3_venv/bin/python analyze_axis.py .gate_cosmos3-edge_reasoned_452.json
```
Backends (`--model`, or `C3_MODEL`): `cosmos3-edge` (default), `gemma4-e4b`,
`gemma4-e2b`, or any HF id. Everything but the checkpoint is held constant so the
out-of-family runs are a valid control — see RESULTS.md.
```bash
./c3_venv/bin/python gate_runner.py 452 --mode reasoned --model gemma4-e4b
./c3_venv/bin/python analyze_axis.py --compare .gate_cosmos3-edge_reasoned_452.json \
                                               .gate_gemma4-e4b_reasoned_452.json
```
Env knobs: `C3_FRAMES` (default 4), `C3_LONG_SIDE` (640), `C3_DEV`, `C3_CLIPS`.

## Human spot-review
The label-free anchor. Every other number here is scored against `ood_reasoning`,
which is Alpamayo-lineage; human judgement is the only check that sidesteps it.
```bash
./c3_venv/bin/python spot_review.py            # -> review.html + .review_key.json
./c3_venv/bin/python score_review.py ratings.json
```
60 clips, shuffled (seed 0), no scores/labels/rationales on the page. The contrast is
stratum **A** (cosmos3-hard / conflict-easy, n=25) vs **B** (both-easy, n=25) — both
conflict-easy, so any difference is cosmos3's alone. Plus 5 both-hard clips as a
scale-validity guard: if those do not rate hardest, the run is void.

## Hardware
BF16-only, Ampere or newer. `pick_device()` selects the first sm_80+ GPU, because
this host pairs an A10 (Ampere, usable) with a Quadro RTX 6000 (Turing, **not**
usable — no BF16, and SDPA falls back to a flash kernel that refuses pre-Ampere),
and torch's default `CUDA_DEVICE_ORDER=FASTEST_FIRST` does not match `nvidia-smi`
index order. Measured: 4.87 GB weights, 5.28 GB peak at 4x 640x360 frames.

## Scoring
Both scorers read the next-token distribution over `'0'..'9'` after a steered
answer prefix and return `E[digit]/9` in `[0,1]`. One forward pass, so the score is
continuous, always numeric, and bit-exact reproducible. `enable_thinking=False`
closes the `<think></think>` block so the digit is genuinely the next token.
