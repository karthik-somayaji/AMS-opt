import json
import os
import pickle
import sys

import numpy as np
import torch

from backend.llm import gpt
from core import task
from core import proposer
from core import bo
from core import sampler
from core import surrogate
import history_summary
import utils
import argparse


def _resolve_llmbo_relative_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    llmbo_root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(llmbo_root, path)
    return candidate


def _repo_root() -> str:
    # `llmbo.py` lives in `LLMBO/`; repo root is one more level up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _workspace_root() -> str:
    """Return the checked-out repository root (parent of `LLMBO/`)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _infer_gnn_model_cfg_from_state_dict(state_dict: dict) -> dict:
    """Infer model dims from a ContrastiveGINModel checkpoint state_dict."""

    def _shape(key: str):
        if key not in state_dict:
            raise KeyError(f"Missing key '{key}' in checkpoint state_dict")
        return tuple(state_dict[key].shape)

    input_dim = _shape("encoder.gin_layers.0.mlp.0.weight")[1]

    hidden_dims = []
    i = 0
    while f"encoder.gin_layers.{i}.mlp.0.weight" in state_dict:
        hidden_dims.append(_shape(f"encoder.gin_layers.{i}.mlp.0.weight")[0])
        i += 1

    embedding_dim = _shape("encoder.readout_mlp.0.weight")[0]
    projection_dim = _shape("projection_head.mlp.0.weight")[0]
    use_batch_norm = any(k.startswith("encoder.batch_norms.") for k in state_dict.keys())

    return {
        "input_dim": int(input_dim),
        "hidden_dims": [int(x) for x in hidden_dims],
        "embedding_dim": int(embedding_dim),
        "projection_dim": int(projection_dim),
        "use_batch_norm": bool(use_batch_norm),
    }


def _load_gnn_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained GNN model for embedding extraction."""
    repo_root = _workspace_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from gnn_training.models import ContrastiveGINModel  # local import to avoid impacting non-history runs

    checkpoint_abs = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(repo_root, checkpoint_path)
    ckpt = torch.load(checkpoint_abs, map_location=torch.device(device))
    state_dict = ckpt.get("model_state", ckpt)
    cfg = _infer_gnn_model_cfg_from_state_dict(state_dict)

    model = ContrastiveGINModel(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        embedding_dim=cfg["embedding_dim"],
        projection_dim=cfg["projection_dim"],
        dropout=0.0,
        use_batch_norm=cfg["use_batch_norm"],
    )
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model, cfg


def _encode_comb_graph_npz(model, npz_path: str, device: str = "cpu") -> np.ndarray:
    return _encode_comb_graph_npz_mode(model, npz_path=npz_path, device=device, embedding_mode="skg")


def _encode_comb_graph_npz_mode(
    model,
    *,
    npz_path: str,
    device: str = "cpu",
    embedding_mode: str = "skg",
) -> np.ndarray:
    """Encode a `comb_graph_gnn.npz` into the trained encoder embedding space.

    embedding_mode:
      - "skg": use the full graph (structural + knowledge nodes)
      - "sg": strip knowledge nodes first (RemoveKnowledgeNodes) and encode the structural-only graph
    """

    embedding_mode = str(embedding_mode or "skg").lower()
    if embedding_mode not in {"skg", "sg"}:
        raise ValueError(f"Invalid embedding_mode: {embedding_mode}. Expected 'skg' or 'sg'.")

    data = np.load(npz_path, allow_pickle=True)
    nodes = data.get("nodes", None)
    features = data["features"].astype(np.float32)
    adjacency = data["adjacency"].astype(np.float32)

    if embedding_mode == "sg":
        repo_root = _workspace_root()
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from gnn_training.perturbations import RemoveKnowledgeNodes  # local import

        if nodes is None:
            raise ValueError(f"SG mode requires 'nodes' in npz: {npz_path}")
        sg_graph = RemoveKnowledgeNodes({"nodes": nodes, "features": features, "adjacency": adjacency}).apply()
        features = sg_graph["features"]
        adjacency = sg_graph["adjacency"]

    with torch.no_grad():
        feat_t = torch.tensor(features, dtype=torch.float32, device=torch.device(device))
        adj_t = torch.tensor(adjacency, dtype=torch.float32, device=torch.device(device))
        emb = model.encode(feat_t, adj_t).detach().cpu().numpy().reshape(-1)
    return emb


def _ensure_reference_embeddings_json(
    *,
    model,
    out_path: str,
    reference_metadata_json: str,
    netlists_root: str,
    device: str = "cpu",
    embedding_mode: str = "skg",
) -> None:
    """Create (if missing) a JSON of reference embeddings in *raw GNN embedding space*.

    The metadata list is read from `reference_metadata_json` and must contain objects
    with `circuit_id` and `family`.
    """

    if os.path.exists(out_path):
        return

    repo_root = _workspace_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gnn_training.data import CircuitDataLoader  # local import

    with open(reference_metadata_json, "r") as f:
        ref = json.load(f)
    meta = ref.get("metadata")
    if not isinstance(meta, list) or len(meta) == 0:
        raise ValueError(f"Invalid reference metadata JSON (missing metadata list): {reference_metadata_json}")

    embeddings = []
    out_meta = []
    for m in meta:
        cid = str(m.get("circuit_id"))
        fam = str(m.get("family"))
        fam_dir = os.path.join(netlists_root, fam)
        loader = CircuitDataLoader(cid, fam_dir)
        graph = loader.get_graph()
        if str(embedding_mode).lower() == "sg":
            from gnn_training.perturbations import RemoveKnowledgeNodes  # local import

            graph = RemoveKnowledgeNodes(graph).apply()
        with torch.no_grad():
            feat_t = torch.tensor(graph["features"], dtype=torch.float32, device=torch.device(device))
            adj_t = torch.tensor(graph["adjacency"], dtype=torch.float32, device=torch.device(device))
            emb = model.encode(feat_t, adj_t).detach().cpu().numpy().reshape(-1)
        embeddings.append(emb.tolist())
        out_meta.append({"circuit_id": cid, "family": fam})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"embeddings": embeddings, "metadata": out_meta}, f)
    print(f"[LLMBO] Wrote raw GNN reference embeddings: {out_path} (N={len(out_meta)}, D={len(embeddings[0])})")


def _maybe_prepare_llmbo_anchor_embedding(args) -> None:
    """If `--target_id` is one of {amp2,FC,comp,ldo}, compute its GNN embedding as anchor.

    Stores it on `args.target_anchor_embedding` (list[float]) for `LLMBO/core/proposer.py`.
    """

    if not getattr(args, "history", 0):
        return

    target_id = getattr(args, "target_id", None)
    if target_id is None:
        return

    target_id = str(target_id)
    # Numeric ids are treated as netlists anchors (existing behavior)
    if target_id.isdigit():
        return

    alias_to_dir = {
        "amp2": "amp2_ati_new",
        "FC": "FC_ati_new",
        "comp": "comp_ati_new",
        "ldo": "ldo_ati_new",
    }
    if target_id not in alias_to_dir:
        raise ValueError(
            "Invalid --target_id. Expected a numeric netlists id (e.g., 77) or one of "
            f"{sorted(alias_to_dir.keys())}; got '{target_id}'."
        )

    repo_root = _workspace_root()
    device = getattr(args, "gnn_device", "cpu")
    checkpoint = getattr(args, "gnn_checkpoint", os.path.join(repo_root, "checkpoints_full_training", "best_model.pt"))
    model, cfg = _load_gnn_model(checkpoint_path=checkpoint, device=device)
    embedding_mode = str(getattr(args, "gnn_embedding_mode", "skg") or "skg").lower()

    ckt_dir = os.path.join(repo_root, "LLMBO", alias_to_dir[target_id])
    npz_path = os.path.join(ckt_dir, "comb_graph_gnn.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing anchor graph features: {npz_path}")

    anchor_emb = _encode_comb_graph_npz_mode(model, npz_path=npz_path, device=device, embedding_mode=embedding_mode)
    args.target_anchor_embedding = anchor_emb.tolist()

    # Ensure reference embeddings JSON exists (in the same raw embedding space)
    ref_meta_default = os.path.join(repo_root, "umap_results_full", "umap_embeddings.json")
    reference_metadata_json = getattr(args, "reference_metadata_json", ref_meta_default)
    reference_metadata_json = (
        reference_metadata_json
        if os.path.isabs(reference_metadata_json)
        else os.path.join(repo_root, reference_metadata_json)
    )

    # Default embeddings JSON depends on embedding_mode.
    embeddings_json = getattr(args, "embeddings_json", None)
    default_skg = os.path.join(repo_root, "umap_results_full", "gnn_embeddings.json")
    default_sg = os.path.join(repo_root, "umap_results_full", "gnn_embeddings_sg.json")
    if not embeddings_json or embeddings_json == "umap_results_full/gnn_embeddings.json":
        embeddings_json = default_sg if embedding_mode == "sg" else default_skg
        args.embeddings_json = embeddings_json
    embeddings_json = embeddings_json if os.path.isabs(embeddings_json) else os.path.join(repo_root, embeddings_json)
    args.embeddings_json = embeddings_json

    netlists_root = os.path.join(repo_root, "netlists")
    _ensure_reference_embeddings_json(
        model=model,
        out_path=embeddings_json,
        reference_metadata_json=reference_metadata_json,
        netlists_root=netlists_root,
        device=device,
        embedding_mode=embedding_mode,
    )

    print(
        f"[LLMBO] Using LLMBO anchor embedding for --target_id {target_id!r} "
        f"(D={anchor_emb.shape[0]}) with reference embeddings {os.path.relpath(embeddings_json, repo_root)}"
    )


def _is_within_dir(path: str, parent_dir: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(parent_dir)]) == os.path.abspath(parent_dir)
    except ValueError:
        return False


def _enforce_optimization_circuit_roots(task_setting: dict, allowed_subdirs: list[str]) -> None:
    """Fail fast if a task points optimization at a disallowed circuit directory.

    `netlists/<family>` should be used only for optional KG loading (prompt context),
    never as the HSPICE optimization target.
    """
    ckt_dir = task_setting.get("ckt_dir")
    if not ckt_dir:
        raise ValueError("Task setting missing required key 'ckt_dir'.")

    allowed_abs = [os.path.abspath(os.path.join(_repo_root(), "LLMBO", sd)) for sd in allowed_subdirs]
    ckt_dir_abs = os.path.abspath(ckt_dir)
    if not any(_is_within_dir(ckt_dir_abs, a) for a in allowed_abs):
        allowed_pretty = ", ".join([os.path.join("LLMBO", sd) for sd in allowed_subdirs])
        raise ValueError(
            "Refusing to optimize a circuit outside the allowed *_ati_new directories. "
            f"Got ckt_dir={ckt_dir_abs}. Allowed roots: {allowed_pretty}."
        )



class LLMBO(object):
    def __init__(self,
                 path_task_setting,
                 n_init_data=1,
                 init_method="zeroshot",
                 #n_sample=10,
                 n_sample=3,
                 sample_method="topk",
                 shuffle_sample=False,
                 n_proposal_llm=1,
                 n_proposal_bo=1,
                 n_itr=10,
                 gpt_version="3.5",
                 openai_api_seed=95,
                 # openai_api_seed=42,
                 path_checkpoints="./checkpoints",
                 rank_based_on_bo=True
                 ):
        """
        Args:
            n_init_data: number of datapoints for initialization;
            init_method: initialization method, should be ["zeroshot", "random", "fixed"]
            n_sample: number of high quality data to be sampled from data_collected;
            sample_method: sample method, should be ["topk", "random", "all"]
            shuffle_sample: whether to shuffle the sample in each iteration;
            n_proposal_llm: number of parameter settings proposed by LLM in each iteration, 0 means do not use llm proposer;
            n_proposal_bo: number of parameter settings proposed by BO (GP) in each iteration, 0 means do not use bo proposer;
        """
        assert init_method in ["zeroshot", "random", "fixed"]
        assert sample_method in ["topk", "random", "all", "mixed"]
        if gpt_version != "3.5" and gpt_version != "4":
            raise ValueError(f"gpt_version expect to be '3.5' or '4', got '{gpt_version}'.")

        self.gpt_version = gpt_version
        self.openai_api_seed = openai_api_seed
        if os.environ.get("OPENAI_API_KEY"):
            self.backend = gpt.GPT(model=gpt_version, seed=openai_api_seed, debug_mode=False)
        else:
            self.backend = None
            if init_method == "zeroshot":
                print("[LLMBO] OPENAI_API_KEY not set; falling back init_method='random'.")
                self.init_method = "random"

        self.data_dict_keys = ["params", "metrics", "targets", "aux_info", "params_numpy"]
        self.task = task.Task(path_task_setting, self.data_dict_keys)

        self.name_task = self.task.name_task
        self.ckt_name_description = self.task.ckt_name_description

        self.params_list = self.task.params_list
        self.n_params = self.task.n_params
        self.n_metrics = self.task.n_metrics
        self.ranges = self.task.ranges

        self.n_init_data = n_init_data
        if not hasattr(self, "init_method"):
            self.init_method = init_method
        self.n_sample = n_sample
        self.sample_method = sample_method
        self.shuffle_sample = shuffle_sample

        self.n_proposal_llm = min(5, n_proposal_llm) if rank_based_on_bo==True else 1 #n_proposal_llm
        self.n_proposal_bo = n_proposal_bo

        self.rank_based_on_bo = rank_based_on_bo

        self.n_itr = n_itr

        self.path_checkpoints = path_checkpoints + f"_{self.name_task}_gpt{gpt_version}"
        if not os.path.exists(self.path_checkpoints):
            os.makedirs(self.path_checkpoints)

        self.checkpoints = {
            "gpt_version": self.gpt_version,
            "openai_api_seed": openai_api_seed
        }

        self.data_collected = utils.create_empty_data_dict(self.data_dict_keys)
        self.data_collected_llm = utils.create_empty_data_dict(self.data_dict_keys)
        self.data_collected_bo = utils.create_empty_data_dict(self.data_dict_keys)

        self.params_proposed = []


        # Get initial data and add them to collected data;
        self.data_collected, self.data_collected_llm, self.data_collected_bo = self.task.initialize(self.n_init_data, self.init_method, log_info=True)


        self.llm_proposer = None
        if self.backend is not None and self.n_proposal_llm > 0:
            self.llm_proposer = proposer.LLMProposer(
                ckt_name_description=self.task.ckt_name_description,
                params_list=self.params_list,
                task_context=self.task.task_context,
                backend=self.backend,
                n_proposal=self.n_proposal_llm,
                ranges=self.ranges,
                example_keys=["params", "metrics", "targets", "aux_info"],
            )

        self.sampler = sampler.Sampler(
            n_sample=self.n_sample,
            method=self.sample_method,
            shuffle=self.shuffle_sample
        )

        self.bo_proposer = bo.GPBO(
            data_init=self.data_collected,
            params_list=self.params_list,
            n_proposal=self.n_proposal_bo,
            ranges=self.ranges
        )

        self.target_best = max(self.data_collected["targets"])
        print(f"Initial best target value: {self.target_best:.2f}")

    def optimize(self, args):
        print("Beginning Design Cycle")
        fom_array = []
        for itr in range(self.n_itr):
            print(f"Current iteration:{itr+1}.")
            # Sample high quality data for llm using sampler;
            data_pro = self.sampler.sample(self.data_collected)
            #print(data_pro['targets'])

            params_proposed_llm = []
            if self.llm_proposer is not None:
                params_proposed_llm = self.llm_proposer.propose_params(data_pro, args)

            print('************* IN HISTORY *************')            
            #responses_for_history = self.llm_proposer.generate_prompt_history_summary(data_pro) 
            print('************* OUT HISTORY *************')  

            params_proposed_bo = self.bo_proposer.propose_params(self.data_collected)

            if self.rank_based_on_bo and len(params_proposed_llm) > 0:
                #print(params_proposed_llm)
                params_proposed_llm_array = np.array([utils.params_dict_to_nparray(x) for x in params_proposed_llm ])
                ranked_llm_acq_fn_values = self.bo_proposer.obtain_acq_function_values(params_proposed_llm_array)
                top_k_indices = np.argsort(ranked_llm_acq_fn_values)[-1:]
                #top_k_values = params_proposed_llm_array[top_k_indices]
                params_proposed_llm = [utils.nparray_to_params_dict(params_proposed_llm_array[x], self.params_list) for x in top_k_indices]
                #params_proposed_llm = [utils.nparray_to_params_dict(params_proposed_llm_array[np.argmax(ranked_llm_acq_fn_values)], self.params_list)]

            for params_query_llm in params_proposed_llm:
                print('IN LLM LOOP')
                data_new_llm = self.task.evaluate(params_query_llm, log_info=False)
                self.update_data(self.data_collected_llm, data_new_llm)
                self.update_data(self.data_collected, data_new_llm)

            for params_query_bo in params_proposed_bo:
                print('IN BO LOOP')
                data_new_bo = self.task.evaluate(params_query_bo, log_info=False)
                self.update_data(self.data_collected_bo, data_new_bo)
                self.update_data(self.data_collected, data_new_bo)

            
            self.target_best = max(self.data_collected["targets"])
            self.target_best_index = self.data_collected["targets"].index(self.target_best)
            self.target_best_llm = max(self.data_collected_llm["targets"])
            self.target_best_index_llm = self.data_collected_llm["targets"].index(self.target_best_llm)
            self.target_best_bo = max(self.data_collected_bo["targets"])
            self.target_best_index_bo = self.data_collected_bo["targets"].index(self.target_best_bo)

            #print("Best opt. point",  self.data_collected["params"][self.target_best_index])
            #print(f"At iteration: {itr + 1}, the best target value is: {self.target_best:.3f}, bo best target: {self.target_best_bo:.3f}")
            #print("Best LLM opt. point" ,self.data_collected_llm["params"][self.target_best_index_llm])
            print("Best metrics",  self.data_collected["metrics"][self.target_best_index])
            print("Best BO metrics",  self.data_collected_bo["metrics"][self.target_best_index_bo])
            print("Best LLM metrics",  self.data_collected_llm["metrics"][self.target_best_index_llm])
            print(f"At iteration: {itr + 1}, the best target value is: {self.target_best:.3f}, llm best target: {self.target_best_llm:.3f}, bo best target: {self.target_best_bo:.3f}")

            fom_array.append(self.target_best)
            

            # Log current iteration information and save checkpoints; TODO:
            self.save_checkpoints(itr)

        os.makedirs("results", exist_ok=True)
        model_label = getattr(args, "model", None) or "nomodel"
        with open(f"results/{args.history}_{args.refined}_comp_{model_label}.txt" , 'w') as f:
            print("Best metrics",  self.data_collected["metrics"][self.target_best_index], file=f)
            print("Best BO metrics",  self.data_collected_bo["metrics"][self.target_best_index_bo], file=f)
            print("Best LLM metrics",  self.data_collected_llm["metrics"][self.target_best_index_llm], file=f)
            print(f"At iteration: {itr + 1}, the best target value is: {self.target_best:.3f}, llm best target: {self.target_best_llm:.3f}, bo best target: {self.target_best_bo:.3f}", file=f)

        np.savetxt(f"results/foms_{args.history}_{args.refined}_comp_{model_label}.txt", np.array(fom_array), fmt="%.4f")


    def critic(self, file_name):
        with open(f'history_summary/{file_name}.txt', 'r') as file:
            content = file.read()

        prefix = """Consider you are an analog designer critic. Given a circuit type, netlist and description of design rules for a given circuit, your job is to judge the correctness of the design rules from the knowledge you possess about circuit design. Below is given one such circuit.

        *** Circuit Description with design rules ***
        
        """
        suffix = """Given the netlist and the design rules, please modify the design rules mention accordingly if you find factual inconsistencies and generate a response similar to the input, but with changed design rules if you think there are factual discrepancies in the design rules in the input. Please output only the corrected design rules for Optimizing parameters considering tradeoffs:."""

        prompt = prefix
        prompt += content
        prompt += suffix

        response = self.backend.request(prompt)

        with open(f'critic_{file_name}.txt', 'w') as file:
            file.write(response[0])

    @staticmethod
    def update_data(data_collected, data_new):
        for k, v in data_new.items():
            data_collected[k].append(v)


    def save_checkpoints(self, itr):
        self.checkpoints["data_collected"] = self.data_collected
        self.checkpoints["itr"] = itr + 1

        with open(self.path_checkpoints + '/checkpoints.pkl', 'wb') as f:
            pickle.dump(self.checkpoints, f)

    def load_checkpoints(self, itr):
        with open(self.path_checkpoints + '/checkpoints.pkl', 'rb') as f:
            checkpoints = pickle.load(f)

        return checkpoints


if __name__ == "__main__":

    # np.random.seed(14)
    # torch.random.manual_seed(14)
    np.random.seed(114)
    torch.random.manual_seed(114)
    openai_api_seed = 514

    parser = argparse.ArgumentParser(description="Simple argparse example")
    # String argument
    parser.add_argument('--model', type=str, help='Your name: DeepSeek-R1-Distill-Llama-70B')
    # Boolean flag (store_true means it becomes True if specified)
    parser.add_argument('--history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--refined', type=int, choices=[0, 1], default=0)
    parser.add_argument('--target_id', type=str, default=None,
                        help=(
                            'Anchor circuit selector used ONLY for related KG loading. '
                            'Accepts either a numeric netlists id (e.g., 77) or one of {amp2, FC, comp, ldo}. '
                            'For the symbolic ids, the anchor embedding is computed from '
                            '`LLMBO/<id>_ati_new/comb_graph_gnn.npz` using `--gnn_checkpoint`.'
                        ))
    parser.add_argument('--related_mode', type=str, default='topk', choices=['topk', 'bottomk', 'random_family'],
                        help='How to pick related circuits for KG context.')
    parser.add_argument('--related_k', type=int, default=3,
                        help='Number of related circuits whose fun_graph.json to include.')
    parser.add_argument('--embeddings_json', type=str, default='umap_results_full/gnn_embeddings.json',
                        help='Path to *raw GNN* embeddings+metadata JSON for similarity search (same D as the trained encoder).')
    parser.add_argument('--reference_metadata_json', type=str, default='umap_results_full/umap_embeddings.json',
                        help='Metadata JSON used to decide which netlists circuits to embed when generating embeddings_json.')
    parser.add_argument('--gnn_checkpoint', type=str, default='checkpoints_full_training/best_model.pt',
                        help='Checkpoint used to encode the LLMBO anchor and (if needed) generate raw reference embeddings.')
    parser.add_argument('--gnn_device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device used for encoding graphs into embeddings.')
    parser.add_argument(
        '--gnn_embedding_mode',
        type=str,
        default='skg',
        choices=['skg', 'sg'],
        help=(
            "Which graph to encode into embeddings: 'skg' uses full SG+KG, 'sg' strips knowledge nodes first. "
            "If 'sg', set --embeddings_json to an SG embeddings file (or let it default to umap_results_full/gnn_embeddings_sg.json)."
        ),
    )
    # List argument (space-separated values)
    parser.add_argument('--related_ckts', nargs='+', help='List of items')
    args = parser.parse_args()

    _maybe_prepare_llmbo_anchor_embedding(args)

    # Optimization must only target these curated circuits.
    # NOTE: user clarification indicates netlists/<family> are KG sources only.
    allowed_opt_subdirs = [
        "amp2_ati_new",
        "FC_ati_new",
        "comp_ati_new",
        "ldo_ati_new",
    ]

    # LLMBO + GPBO;
    print(_resolve_llmbo_relative_path("tasks/ldo/ldo.json"))
    task_path = _resolve_llmbo_relative_path("tasks/ldo/ldo.json")
    # print(_resolve_llmbo_relative_path("tasks/comp/comp.json"))
    # task_path = _resolve_llmbo_relative_path("tasks/comp/comp.json")
    # print(_resolve_llmbo_relative_path("tasks/comp/comp_test.json"))
    # task_path = _resolve_llmbo_relative_path("tasks/comp/comp_test.json")
    # print(_resolve_llmbo_relative_path("tasks/FC/FC.json"))
    # task_path = _resolve_llmbo_relative_path("tasks/FC/FC.json")
    # print(_resolve_llmbo_relative_path("tasks/amp2/amp2.json"))
    # task_path = _resolve_llmbo_relative_path("tasks/amp2/amp2.json")
    try:
        import json as _json
        with open(task_path, "r") as _f:
            _task_setting = _json.load(_f)
        _enforce_optimization_circuit_roots(_task_setting, allowed_opt_subdirs)
    except Exception as e:
        raise SystemExit(f"[LLMBO] Invalid optimization circuit setup: {e}")

    llmbo = LLMBO(
        task_path,
        #"tasks/comp/comp.json",
        #"tasks/ldo/ldo.json",
        #"tasks/dcdc/dcdc.json",
        openai_api_seed=openai_api_seed,
        gpt_version="3.5",
        #n_init_data=5, # for amp
        n_init_data=3,
        #n_init_data=1,
        n_proposal_llm=1, #,4, # for amp
        #n_proposal_llm=3,
        n_proposal_bo=1,
        # n_itr=30,
        n_itr=20,
        rank_based_on_bo = False#True #True#True# True
    )


    # Only run with GPBO;
    # llmbo = LLMBO(
    #     "tasks/amp2/amp2.json",
    #     openai_api_seed=openai_api_seed,
    #     gpt_version="3.5",
    #     n_init_data=5,
    #     n_proposal_llm=0,
    #     n_proposal_bo=2,
    #     n_itr=10
    # )

    llmbo.optimize(args)

    # llmbo_critic = LLMBO(
    #     #"tasks/amp2/amp2.json",
    #     #"tasks/FC/FC.json",
    #     "tasks/comp/comp.json",
    #     #"tasks/ldo/ldo.json",
    #     openai_api_seed=openai_api_seed,
    #     gpt_version="4",
    #     n_init_data=4,
    #     #n_init_data=1,
    #     n_proposal_llm=1,
    #     n_proposal_bo=1,
    #     n_itr=30
    #     #n_itr=20
    # )

    # llmbo_critic.critic(file_name= 'Low_Dropout_Regulator')

    # python llmbo.py --history 1 --refined 1 --model DeepSeek-R1-Distill-Llama-70B --related_ckts ["Two_Stage_Differential_Amplifier"]