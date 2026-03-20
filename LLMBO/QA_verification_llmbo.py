import argparse
import copy
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from backend.llm import gpt
from core import bo
from core import sampler
from core import task
from core.QA_verification_proposer import QAValidationLLMProposer


def _resolve_llmbo_relative_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    llmbo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(llmbo_root, path)


def _default_task_path(circuit: str) -> str:
    circuit = str(circuit)
    task_rel = {
        "amp2": "tasks/amp2/amp2.json",
        "FC": "tasks/FC/FC.json",
        "comp": "tasks/comp/comp.json",
        "ldo": "tasks/ldo/ldo.json",
    }.get(circuit)
    if task_rel is None:
        raise ValueError(f"Unknown circuit '{circuit}'. Expected one of amp2, FC, comp, ldo.")
    return _resolve_llmbo_relative_path(task_rel)


def _default_qa_json_path(circuit: str) -> str:
    return os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "QA_Task",
        "raw_QA",
        f"{circuit}_raw_QA.json",
    )


def _load_qa_sentences(
    qa_json_path: str,
    *,
    include_ids: Optional[List[str]] = None,
    max_sentences: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with open(qa_json_path, "r") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(f"Expected a list in QA JSON: {qa_json_path}")

    if include_ids:
        wanted = set(map(str, include_ids))
        items = [x for x in items if str(x.get("id")) in wanted]

    # Ensure stable order: by id then question.
    items.sort(key=lambda x: (str(x.get("id", "")), str(x.get("question", ""))))

    if max_sentences is not None:
        items = items[: int(max_sentences)]

    return items


def _compile_paragraph(qa_items: List[Dict[str, Any]]) -> str:
    lines = []
    for x in qa_items:
        qid = str(x.get("id", ""))
        sent = str(x.get("sentence", "")).strip()
        if not sent:
            continue
        if qid:
            lines.append(f"[{qid}] {sent}")
        else:
            lines.append(sent)
    return "\n".join(lines)


class QAVerificationLLMBO:
    def __init__(
        self,
        path_task_setting: str,
        *,
        init_data: Optional[Dict[str, Any]] = None,
        n_init_data: int = 3,
        init_method: str = "zeroshot",
        n_sample: int = 3,
        sample_method: str = "topk",
        shuffle_sample: bool = False,
        n_proposal_llm: int = 1,
        n_proposal_bo: int = 1,
        n_itr: int = 10,
        gpt_version: str = "3.5",
        openai_api_seed: int = 514,
        llm_temperature: float = 0.0,
        path_checkpoints: str = "./checkpoints",
        rank_based_on_bo: bool = False,
    ):
        assert init_method in ["zeroshot", "random", "fixed"]
        assert sample_method in ["topk", "random", "all", "mixed"]
        if gpt_version not in {"3.5", "4"}:
            raise ValueError(f"gpt_version expect to be '3.5' or '4', got '{gpt_version}'.")

        self.gpt_version = gpt_version
        self.openai_api_seed = openai_api_seed
        self.llm_temperature = float(llm_temperature)

        if os.environ.get("OPENAI_API_KEY"):
            self.backend = gpt.GPT(
                model=gpt_version,
                seed=openai_api_seed,
                temperature=self.llm_temperature,
                debug_mode=False,
            )
        else:
            self.backend = None
            if init_method == "zeroshot":
                print("[QAVerification] OPENAI_API_KEY not set; falling back init_method='random'.")
                init_method = "random"

        self.data_dict_keys = ["params", "metrics", "targets", "aux_info", "params_numpy"]
        self.task = task.Task(path_task_setting, self.data_dict_keys)

        self.name_task = self.task.name_task
        self.params_list = self.task.params_list
        self.ranges = self.task.ranges

        self.n_init_data = int(n_init_data)
        self.init_method = init_method
        self.n_sample = int(n_sample)
        self.sample_method = sample_method
        self.shuffle_sample = bool(shuffle_sample)

        self.n_proposal_llm = min(5, int(n_proposal_llm)) if rank_based_on_bo else int(n_proposal_llm)
        self.n_proposal_bo = int(n_proposal_bo)
        self.rank_based_on_bo = bool(rank_based_on_bo)
        self.n_itr = int(n_itr)

        self.path_checkpoints = path_checkpoints + f"_{self.name_task}_qa_verify_gpt{gpt_version}"
        os.makedirs(self.path_checkpoints, exist_ok=True)

        self.checkpoints: Dict[str, Any] = {
            "gpt_version": self.gpt_version,
            "openai_api_seed": openai_api_seed,
            "llm_temperature": self.llm_temperature,
        }

        import utils

        self.data_collected = utils.create_empty_data_dict(self.data_dict_keys)
        self.data_collected_llm = utils.create_empty_data_dict(self.data_dict_keys)
        self.data_collected_bo = utils.create_empty_data_dict(self.data_dict_keys)

        if init_data is not None:
            loaded = init_data.get("data_collected") if isinstance(init_data, dict) else None
            if not isinstance(loaded, dict) or "targets" not in loaded:
                raise ValueError("init_data must be a dict containing key 'data_collected' with targets.")
            self.data_collected = copy.deepcopy(loaded)
        else:
            self.data_collected, self.data_collected_llm, self.data_collected_bo = self.task.initialize(
                self.n_init_data, self.init_method, log_info=True
            )

        self.llm_proposer = None
        if self.backend is not None and self.n_proposal_llm > 0:
            self.llm_proposer = QAValidationLLMProposer(
                ckt_name_description=self.task.ckt_name_description,
                params_list=self.params_list,
                task_context=self.task.task_context,
                backend=self.backend,
                n_proposal=self.n_proposal_llm,
                ranges=self.ranges,
                example_keys=["params", "metrics", "targets", "aux_info"],
            )

        self.sampler = sampler.Sampler(n_sample=self.n_sample, method=self.sample_method, shuffle=self.shuffle_sample)
        self.bo_proposer = bo.GPBO(
            data_init=self.data_collected,
            params_list=self.params_list,
            n_proposal=self.n_proposal_bo,
            ranges=self.ranges,
        )

        self.target_best = max(self.data_collected["targets"])
        print(f"[QAVerification] Initial best FOM: {self.target_best:.4f}")

    @staticmethod
    def update_data(data_collected: Dict[str, list], data_new: Dict[str, Any]) -> None:
        for k, v in data_new.items():
            data_collected[k].append(v)

    def save_checkpoints(self, itr: int) -> None:
        self.checkpoints["data_collected"] = self.data_collected
        self.checkpoints["itr"] = itr + 1
        with open(os.path.join(self.path_checkpoints, "checkpoints.pkl"), "wb") as f:
            pickle.dump(self.checkpoints, f)

    def optimize(self, args) -> float:
        import utils

        print("[QAVerification] Beginning Design Cycle")
        for itr in range(self.n_itr):
            print(f"[QAVerification] Iteration {itr + 1}/{self.n_itr}")

            data_pro = self.sampler.sample(self.data_collected)

            params_proposed_llm = []
            if self.llm_proposer is not None:
                params_proposed_llm = self.llm_proposer.propose_params(data_pro, args)

            params_proposed_bo = self.bo_proposer.propose_params(self.data_collected)

            if self.rank_based_on_bo and len(params_proposed_llm) > 0:
                params_proposed_llm_array = np.array([utils.params_dict_to_nparray(x) for x in params_proposed_llm])
                ranked = self.bo_proposer.obtain_acq_function_values(params_proposed_llm_array)
                top_k_indices = np.argsort(ranked)[-1:]
                params_proposed_llm = [
                    utils.nparray_to_params_dict(params_proposed_llm_array[x], self.params_list) for x in top_k_indices
                ]

            for params_query_llm in params_proposed_llm:
                data_new_llm = self.task.evaluate(params_query_llm, log_info=False)
                self.update_data(self.data_collected_llm, data_new_llm)
                self.update_data(self.data_collected, data_new_llm)

            for params_query_bo in params_proposed_bo:
                data_new_bo = self.task.evaluate(params_query_bo, log_info=False)
                self.update_data(self.data_collected_bo, data_new_bo)
                self.update_data(self.data_collected, data_new_bo)

            self.target_best = max(self.data_collected["targets"])
            best_idx = self.data_collected["targets"].index(self.target_best)
            print(f"[QAVerification] Best metrics: {self.data_collected['metrics'][best_idx]}")
            print(f"[QAVerification] Best FOM so far: {self.target_best:.4f}")

            self.save_checkpoints(itr)

        print(f"[QAVerification] FINAL_BEST_FOM={self.target_best:.6f}")
        return float(self.target_best)


def main() -> None:
    np.random.seed(114)
    torch.random.manual_seed(114)

    ap = argparse.ArgumentParser(description="QA verification runner (paragraph-injection).")
    ap.add_argument("--circuit", type=str, required=True, choices=["amp2", "FC", "comp", "ldo"])
    ap.add_argument("--task_path", type=str, default=None, help="Path to task JSON. Default inferred from --circuit.")
    ap.add_argument("--qa_json", type=str, default=None, help="Path to raw QA JSON. Default inferred from --circuit.")
    ap.add_argument("--qa_ids", nargs="+", default=None, help="Optional list of QA ids to include.")
    ap.add_argument("--qa_max_sentences", type=int, default=None, help="Optional cap on number of sentences.")
    ap.add_argument(
        "--paragraph_file",
        type=str,
        default=None,
        help="If set, reads paragraph text from this file and uses it instead of compiling from QA JSON.",
    )

    # LLMBO-like knobs
    ap.add_argument("--history", type=int, choices=[0, 1], default=1)
    ap.add_argument("--model", type=str, default="gpt", help="Unused; kept for CLI compatibility.")
    ap.add_argument("--n_init_data", type=int, default=3)
    ap.add_argument("--n_itr", type=int, default=10)
    ap.add_argument("--n_proposal_llm", type=int, default=1)
    ap.add_argument("--n_proposal_bo", type=int, default=1)
    ap.add_argument("--gpt_version", type=str, default="3.5", choices=["3.5", "4"])
    ap.add_argument("--openai_api_seed", type=int, default=514)
    ap.add_argument(
        "--llm_temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM proposer backend (lower is more deterministic).",
    )
    ap.add_argument("--rank_based_on_bo", type=int, choices=[0, 1], default=0)

    ap.add_argument(
        "--init_data_pkl",
        type=str,
        default=None,
        help="Optional pickle containing {'data_collected': ...} to reuse init simulations.",
    )
    ap.add_argument(
        "--save_init_data_pkl",
        type=str,
        default=None,
        help="If set, saves initialization data to this pickle path and exits.",
    )

    # Accept (and ignore) common llmbo.py args so you can reuse command-lines.
    ap.add_argument("--related_mode", type=str, default="bottomk")
    ap.add_argument("--related_k", type=int, default=1)
    ap.add_argument("--target_id", type=str, default=None)
    ap.add_argument("--gnn_embedding_mode", type=str, default="skg")
    ap.add_argument("--embeddings_json", type=str, default=None)
    ap.add_argument("--reference_metadata_json", type=str, default=None)
    ap.add_argument("--gnn_checkpoint", type=str, default=None)
    ap.add_argument("--gnn_device", type=str, default="cpu")

    args = ap.parse_args()

    task_path = args.task_path or _default_task_path(args.circuit)
    qa_json = args.qa_json or _default_qa_json_path(args.circuit)

    if args.paragraph_file:
        with open(args.paragraph_file, "r") as f:
            paragraph = f.read()
    else:
        qa_items = _load_qa_sentences(qa_json, include_ids=args.qa_ids, max_sentences=args.qa_max_sentences)
        paragraph = _compile_paragraph(qa_items)

    # Attach the paragraph to args for the proposer.
    args.qa_paragraph = paragraph

    init_data = None
    if args.init_data_pkl:
        with open(args.init_data_pkl, "rb") as f:
            init_data = pickle.load(f)

    runner = QAVerificationLLMBO(
        task_path,
        init_data=init_data,
        n_init_data=args.n_init_data,
        n_itr=args.n_itr,
        n_proposal_llm=args.n_proposal_llm,
        n_proposal_bo=args.n_proposal_bo,
        gpt_version=args.gpt_version,
        openai_api_seed=args.openai_api_seed,
        llm_temperature=args.llm_temperature,
        rank_based_on_bo=bool(args.rank_based_on_bo),
    )

    if args.save_init_data_pkl:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_init_data_pkl)), exist_ok=True)
        with open(args.save_init_data_pkl, "wb") as f:
            pickle.dump({"data_collected": runner.data_collected}, f)
        print(f"[QAVerification] Saved init data to {args.save_init_data_pkl}")
        return

    runner.optimize(args)


if __name__ == "__main__":
    main()
