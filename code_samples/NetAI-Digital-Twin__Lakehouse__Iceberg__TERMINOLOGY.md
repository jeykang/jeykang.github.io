# TERMINOLOGY — the words this project uses, and the ones it does not

Adopted 2026-09-21 after the professor twice flagged that "harness" and "agent" read as
LLM harnesses and LLM agents (calls of 2026-08-21 and 2026-09-16). Inside the project
"agent" was also doing two jobs at once: the driving policy under test *and* the other
road users. The replacements below borrow the vocabulary of the scenario-based
validation standards (ISO 34502, ASAM OpenSCENARIO, SOTIF), which resolves the
ambiguity and gives the papers the formal framing he asked for in the same move.

**Rule.** Prose, docs, figures, slides, papers and new code use the *use* column.
Existing code identifiers keep their names until the module is next touched (see
"identifiers still carrying old names"); a doc that must mention such an identifier
writes it in backticks and does not use the word in prose. When a term is introduced
in a paper, expand it once in a terminology box, not a footnote.

## The convention

| concept | use | do not use | precedent |
|---|---|---|---|
| the whole validation system built on the lakehouse | **proving ground** (in Korean, 검증장); "a data-driven / virtual proving ground" | harness, testbed (acceptable in systems venues, second choice), rig | "virtual proving ground" is an established automotive term; 검증장 is the professor's own word |
| the component that runs a policy over episodes and scores it (`evaluation/harness.py`) | **evaluator** — open-loop evaluator, closed-loop evaluator | harness | says what it does |
| the metric code (`evaluation/metrics.py`) | **scorer** | — | NAVSIM's PDM scorer |
| our AlpaSim plugin (`driver=harness`, package `alpasim_harness`) | **policy bridge** | harness plugin, harness driver | it bridges our policy contract into AlpaSim |
| the thing being evaluated | **driving policy**; on first use "the driving policy — the system under test (SUT)" | agent, model (alone), planner (alone) | SUT is the standards' term; the code already says `Policy` |
| the interface a driving policy implements | **policy contract** | agent interface | `evaluation/README.md` already calls it "the whole contract" |
| other road users (tracked cuboids, `obstacle.offline`) | **actors** in code and tables, **traffic participants** in prose on first use | agents | ISO/ASAM; CARLA and OpenSCENARIO use "actors"; NAVSIM's "reactive agents" is exactly the collision to avoid |
| the curation axis built from actor proximity | **traffic conflict** | agent-conflict | the traffic-engineering "conflict technique" (near-crash interaction) is what the axis measures |
| the 121-frame window Cosmos augments | **interaction window** | agent window | |
| deciding which clips earn a closed-loop rollout (`evaluation/skip.py`) | **rollout triage**; the predictor inside it is the **screen**; the budget is the **dial** | skip policy (collides with driving policy) | triage = deciding who gets the expensive resource |
| the six cheap policies scored on every clip | **reference ladder** / **baseline ladder** | — | |
| condition × augmentation × serving mode | **scenario class** (≈ ISO "logical scenario") | scenario (unqualified) | |
| one scored time window of one clip | **episode** (≈ ISO "concrete scenario" / test case) | — | Iceberg `nvidia_gold.episode` |
| the format a validator consumes, into which Gold is materialised | **serving mode** — NVIDIA mode (NuRec/NCore), open-loop mode, … ; the act is **serving** (진열) | compatibility mode, validator mode | 호환 모드로 진열 (professor, 2026-09-16) |
| the curation tiers | **Bronze / Silver / Gold**, "medallion" (금은동) | — | unchanged |
| the reconstructed scene | **twin** — "neural twin (NuRec)", "OpenUSD twin" | — | |
| storage tiers | **accumulation** (축적) vs **serving** (진열) | — | professor, 2026-09-16 |
| evaluation regimes | **open-loop** / **closed-loop**, expanded on first use | — | standard |

Acronyms to expand on first use, every document: MF-PDMS (map-free predictive driver
model score) and EPDMS (extended PDMS, NAVSIM) with their sub-terms NC (no collision),
TTC (time to collision), EP (ego progress), HC/EC (history/absolute comfort); NuRec
(NVIDIA neural reconstruction), USDZ/OpenUSD, NCore v4, SUT, VLA (vision-language-action),
OOD (out of distribution), AUC.

Do-not-use list, with the reason: **harness** (LLM harnesses), **agent** (LLM agents;
and ego-vs-others ambiguity), **rig** (NVIDIA's sensor rig, `T_sensor_rig`),
**test bench** (hardware-in-the-loop connotation), **skip policy** (policy is taken).

## Identifiers still carrying old names (rename when the module is next touched)

| identifier | where | target name |
|---|---|---|
| `harness.py`, `from harness import …` | `evaluation/` | `evaluator.py` with a one-line import shim |
| `alpasim_harness` package, `driver=harness`, `HARNESS_POLICY`, `HARNESS_DIR`, image `alpasim-harness-base` | `evaluation/alpasim/plugin`, `run_scene.sh` | `alpasim_bridge`, `driver=bridge` — only together with the next image rebuild |
| `Scenario.agents`, `agent_history`, `agent_span_us`, `AgentBox`, `n_tracks` | `evaluation/scenario.py`, `adapters.py`, `harness.py` | `actors`, `actor_history`, `actor_span_us`, `ActorBox` |
| `n_agents`, `window_agents` | `evaluation/episodes.py`, `cosmos_augmentation/batch_manifest.json` | `n_actors`, `window_actors` |
| `validator_mode` column | `nvidia_gold.episode`, `episodes.py`, `build_episode_tables.py` | `serving_mode` — at the next table rebuild (needs a migration otherwise) |
| `.conflict/`, `conflict_score`, `conflict_runner.py` | `planning/`, NFS shards | keep `conflict`; the prose name is "traffic conflict" |
| `camera_gated.parquet`, "agent-gated" in docstrings | `planning/write_camera_gated.py` | docstring: "actor-gated" |
| `skip.py` | `evaluation/` | keep the file name; the tool is "rollout triage" |

## Korean ↔ English

검증장 proving ground · 진열 serving (tier / mode) · 축적 accumulation (tier) · 금은동
medallion tiers · 호환 모드 serving mode · 에피소드 episode · 시나리오 scenario class ·
주행 정책 driving policy · 교통 참여자 traffic participants / actors.
