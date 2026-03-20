---
marp: true
title: QA Filtering via FOM Feedback (Step 2a/2b)
paginate: true
---

# QA Filtering via FOM Feedback
## Step 2a (plumbing) + Step 2b (sentence attribution)

Goal: keep only QA statements that *causally* improve downstream circuit optimization (FOM).

---

# Context / Problem

- We generate many MCQ QA items from circuit KGs (Step 1).
- Some QA sentences are wrong or irrelevant.
- Naive paragraph-level filtering fails: incorrect sentences can “free-ride” on correct ones.

Desired property: a kept sentence should be *individually useful* for achieving higher best-FOM.

---

# Step 2a: Make QA sentences influence optimization

**Idea:** Inject QA sentences as prompt context (instead of “related circuit KGs”) and measure impact via final best-FOM.

Implementation pieces:
- `LLMBO/core/QA_verification_proposer.py`
  - Injects a `[QA_PARAGRAPH circuit=...]` block into the proposer prompt.
- `LLMBO/QA_verification_llmbo.py`
  - Runs the existing optimization loop but with the QA paragraph injected.
  - Prints a machine-readable `FINAL_BEST_FOM=...`.

---

# Step 2a: What gets scored?

We score a paragraph $P$ of QA sentences by running the optimizer:

- Input context: the paragraph $P$
- Output score: best observed FOM over the run

We repeat runs and use the **mean** best-FOM to reduce run-to-run variance.

---

# Step 2a: Cost control (init-data reuse)

Optimization is expensive (sim + LLM calls).

To avoid re-simulating initialization points for every variant run:
- `LLMBO/QA_verification_llmbo.py` supports:
  - `--save_init_data_pkl path.pkl` (generate and save init data, then exit)
  - `--init_data_pkl path.pkl` (reuse init simulations)

This turns many variant evaluations into mostly “optimize-from-same-start”.

---

# Step 2a: Deterministic-ish LLM

True determinism isn’t guaranteed, but we reduce randomness:
- `LLMBO/QA_verification_llmbo.py` accepts `--llm_temperature` (default `0.0`)
- Passed into `LLMBO/backend/llm/gpt.py` backend.

Net effect: proposer suggestions vary less across repeats.

---

# Step 2b: Sentence attribution via ablate + flip

Given baseline paragraph $P$ containing $K$ sentences $s_1..s_K$:

For each sentence $s_i$ we evaluate three paragraphs:
- **Baseline**: $P$
- **Ablation**: $P \setminus \{s_i\}$
- **Flip**: $P$ with $s_i$ replaced by a contradictory version

Let $F(\cdot)$ be mean best-FOM over `--repeats` runs.

Define:
- $\Delta_{remove}(i)=F(P) - F(P\setminus\{s_i\})$
- $\Delta_{flip}(i)=F(P) - F(P_{flip(i)})$

Keep $s_i$ if both deltas are large enough.

---

# Step 2b: Keep criterion

Configurable threshold:

- Keep if:
  - $\Delta_{remove}(i) \ge \texttt{min_delta}$ AND
  - $\Delta_{flip}(i) \ge \texttt{min_delta}$

Defaults:
- `--min_delta 0.0` keeps sentences that do not improve FOM when removed/flipped.
- Increase `--min_delta` to require stronger evidence.

---

# Step 2b: Orchestrator script

Driver:
- `QA_Task/filtered_QA/filter_with_llmbo.py`

Responsibilities:
- Build baseline paragraph from QA JSON
- Run baseline / ablate / flip evaluations
- Compute deltas + keep decisions
- Write:
  - `QA_Task/filtered_QA/<circuit>_filtered_QA.json`
  - `QA_Task/filtered_QA/<circuit>_filtered_QA_scores.json`
  - `QA_Task/filtered_QA/<circuit>_init_data.pkl` (cached init)

---

# Key knobs (runtime vs confidence)

- `--paragraph_k`
  - Number of QA sentences in the baseline paragraph
  - Use `-1` or `0` to include **all** QA items
- `--repeats`
  - Mean across repeats reduces randomness
- `--n_itr`, `--n_proposal_llm`, `--n_proposal_bo`
  - Optimization budget per evaluation
- `--llm_temperature`
  - Lower = more stable LLM proposals

Cost scales roughly like: $(1 + 2K) * repeats$ verification runs.

---

# Recommended procedure (FC example)

## 1) Verify all QA items

```bash
cd /home/karthik/sim_clean/AMS-opt
/home/karthik/miniconda3/envs/analog-rep/bin/python filter_with_llmbo.py \
  --circuit FC --paragraph_k -1 --repeats 3 --n_itr 5 --n_proposal_llm 1 --n_proposal_bo 0 \
  --min_delta 0.0 --llm_temperature 0.0
```

## 2) Outputs

- `QA_Task/filtered_QA/FC_filtered_QA.json`
- `QA_Task/filtered_QA/FC_filtered_QA_scores.json`
- `QA_Task/filtered_QA/FC_init_data.pkl`

---

# Reading the results

- `*_filtered_QA.json`
  - Only the kept QA items (same schema as raw QA)
  - Adds a `verification` field with deltas + settings

- `*_scores.json`
  - Full per-id metrics:
    - baseline/ablated/flipped means
    - raw FOMs per repeat
    - `keep` boolean

Use `*_scores.json` to tune `min_delta`, `repeats`, and budgets.

---

# Notes / Known limitations

- “Flip” is heuristic (string edit / answer substitution). It’s a stress test, not a perfect logical negation.
- LLM calls + simulation can still be noisy; repeats help.
- Full-run verification is feasible now ($\sim$50–70 QA items), but scaling will need batching/active selection.
