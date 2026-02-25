# QA_Task

Minimal QA-task evaluation scaffold.

## What it does (currently)

- Loads raw reference GNN embeddings from `umap_results_full/gnn_embeddings.json`.
- For a target `(family, circuit_id)` present in that metadata, retrieves top-k similar circuits by cosine similarity.
- Loads each retrieved circuit's KG text from `netlists/<family>/<circuit_id>/fun_graph.json`.

This is the first step toward the full MCQ accuracy pipeline (baseline vs KG-augmented).

## Quick sanity run

```bash
cd /home/karthik/sim_clean/AMS-opt/AMS-opt
conda run -n analog-rep python QA_Task/eval_mcq.py --sanity --target_family diff_amps --target_id 69 --k 5
```

Outputs a JSON report under `QA_Task/out/`.

## Build prompts from MCQs

`eval_mcq.py` can also load MCQs from `mcq/<family>/*.json` and write a JSONL where each line contains:
- metadata (family, circuit_id, answer)
- `baseline_prompt`
- `augmented_prompt` (baseline + top-k related `fun_graph.json` KGs)

Example using the included sample MCQ:

```bash
cd /home/karthik/sim_clean/AMS-opt/AMS-opt
conda run -n analog-rep python QA_Task/eval_mcq.py \
	--build_prompts \
	--mcq_root QA_Task/sample_mcq \
	--k 3 \
	--limit 1 \
	--out_jsonl prompts_sample.jsonl
```

Output:
- `QA_Task/out/prompts_sample.jsonl`

## vLLM integration (baseline vs KG-augmented accuracy)

`eval_mcq.py` supports two ways to use vLLM:
- In-process (recommended here): import `vllm` and run `LLM(...).generate(...)` directly (no server).
- Optional: call a running vLLM server via the OpenAI-compatible HTTP API.

### A) In-process vLLM (no server)

Run evaluation with a local/HF model id or a local model path:

```bash
cd /home/karthik/sim_clean/AMS-opt/AMS-opt
conda run -n analog-rep python QA_Task/eval_mcq.py \
	--eval_vllm_local \
	--mcq_root mcqs \
	--k 5 \
	--vllm_model meta-llama/Meta-Llama-3-8B-Instruct \
	--apply_chat_template \
	--temperature 0 \
	--max_tokens 8 \
	--out_jsonl eval_results_local.jsonl \
	--summary_json summary_local.json
```

Optional knobs (local mode):
- `--tensor_parallel_size 2`
- `--gpu_memory_utilization 0.9`
- `--batch_size 8`
- `--dtype half` (or `bfloat16`)

Dry-run (does not load the model; writes one example line unless `--limit` is set):

```bash
cd /home/karthik/sim_clean/AMS-opt/AMS-opt
conda run -n analog-rep python QA_Task/eval_mcq.py \
	--eval_vllm_local \
	--dry_run \
	--mcq_root mcqs \
	--k 5 \
	--limit 1 \
	--vllm_model meta-llama/Meta-Llama-3-8B-Instruct \
	--out_jsonl eval_dryrun_local.jsonl \
	--summary_json summary_dryrun_local.json
```

### B) Optional: server-based vLLM (OpenAI-compatible HTTP)

#### 1) Start the vLLM server

Example (serving a local Llama 3 8B instruct model path):

```bash
# In a separate terminal / session
vllm serve /path/to/Meta-Llama-3-8B-Instruct \
	--host 0.0.0.0 \
	--port 8000 \
	--served-model-name llama3-8b
```

Notes:
- The `--served-model-name` is what you pass as `--vllm_model` when running eval.
- If your vLLM is behind auth, set `VLLM_API_KEY` or pass `--vllm_api_key`.

#### 2) (Optional) Dry-run to validate plumbing

This builds prompts and constructs the HTTP payload without sending requests:

```bash
cd /home/karthik/sim_clean/AMS-opt/AMS-opt
conda run -n analog-rep python QA_Task/eval_mcq.py \
	--eval_vllm \
	--dry_run \
	--mcq_root mcqs \
	--k 5 \
	--limit 1 \
	--vllm_base_url http://localhost:8000 \
	--vllm_model llama3-8b \
	--out_jsonl eval_dryrun.jsonl \
	--summary_json summary_dryrun.json
```

#### 3) Run evaluation

```bash
cd /home/karthik/sim_clean/AMS-opt/AMS-opt
conda run -n analog-rep python QA_Task/eval_mcq.py \
	--eval_vllm \
	--mcq_root mcqs \
	--k 5 \
	--vllm_base_url http://localhost:8000 \
	--vllm_model llama3-8b \
	--temperature 0 \
	--max_tokens 8 \
	--out_jsonl eval_results.jsonl \
	--summary_json summary.json
```

Outputs:
- Per-question JSONL with predictions: `QA_Task/out/eval_results.jsonl`
- Summary metrics: `QA_Task/out/summary.json`
