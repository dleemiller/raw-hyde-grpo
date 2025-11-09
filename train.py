#!/usr/bin/env python
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional

import yaml
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SparseEncoder, util
from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Config dataclasses
# -----------------------------
CORPUS_NAME = "BeIR/nfcorpus"
CORPUS_QRELS = "BeIR/nfcorpus-qrels"

@dataclass
class IRConfig:
    sparse_model_name: str = "naver/splade-cocondenser-ensembledistil"
    nfcorpus_split: str = "train"   # which qrels split to use: "train" | "dev" | "test"
    max_queries: Optional[int] = None  # optional cap on number of queries (for debugging)


@dataclass
class TrainConfig:
    model_name: str
    ir: IRConfig
    grpo: Dict[str, Any]  # raw GRPOConfig kwargs


# -----------------------------
# NFcorpus loading helpers
# -----------------------------

def load_nfcorpus() -> tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """
    Load NFcorpus corpus, queries and qrels into convenient dicts:

    - corpus: cid -> text (title + text)
    - queries: qid -> query text
    - relevant_docs: qid -> set of relevant corpus IDs
    """
    # Corpus & queries
    corpus_ds = load_dataset(CORPUS_NAME, "corpus", split="corpus")
    queries_ds = load_dataset(CORPUS_NAME, "queries", split="queries")

    # For this dataset, concatenate title + text
    corpus_ds = corpus_ds.map(
        lambda x: {"text": (x["title"] + " " + x["text"]).strip()},
        remove_columns=["title"],
    )

    corpus = dict(zip(corpus_ds["_id"], corpus_ds["text"]))
    queries = dict(zip(queries_ds["_id"], queries_ds["text"]))

    # Qrels (relevant docs)
    qrels_ds = load_dataset(CORPUS_QRELS, split="train")
    relevant_docs: Dict[str, Set[str]] = {}

    for qid, cid in zip(qrels_ds["query-id"], qrels_ds["corpus-id"]):
        qid = str(qid)
        cid = str(cid)
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(cid)
    return corpus, queries, relevant_docs


# -----------------------------
# Reward factory
# -----------------------------
def build_nfcorpus_multi_metric_reward(
    model: SparseEncoder,
    corpus: Dict[str, str],
    relevant_docs: Dict[str, Set[str]],
    ndcg_ks: List[int] = (10, 100),
    recall_ks: List[int] = (10, 100),
    mrr_ks: List[int] = (10,),
    # Optional explicit weights per metric variant; if None we just average them all.
    metric_weights: Optional[Dict[str, float]] = None,
    batch_size: int = 64,
):
    """
    Build a TRL-compatible reward function that combines multiple IR metrics:
    - NDCG@k for k in ndcg_ks
    - Recall@k for k in recall_ks
    - MRR@k for k in mrr_ks

    Reward is a weighted average of all these metric values.
    """

    # ---- Pre-encode corpus once ----
    corpus_ids: List[str] = list(corpus.keys())
    corpus_texts: List[str] = [corpus[cid] for cid in corpus_ids]

    print(f"[IR] Encoding corpus with SparseEncoder ({len(corpus_ids)} docs)...")
    corpus_emb = model.encode(
        corpus_texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )  # [num_docs, dim]

    # We may need multiple different K values; take max for the topk call
    all_ks = list(set(list(ndcg_ks) + list(recall_ks) + list(mrr_ks)))
    max_k = max(all_ks)

    # ---- Precompute ideal DCG for each qid and each NDCG cutoff ----
    def _idcg(num_rel: int, k: int) -> float:
        return sum(1.0 / math.log2(rank + 2) for rank in range(min(num_rel, k)))

    idcg_per_qid = {
        qid: {k: _idcg(len(doc_ids), k) for k in ndcg_ks}
        for qid, doc_ids in relevant_docs.items()
        if len(doc_ids) > 0
    }

    # ---- Metric weighting ----
    # We define metric names like: "ndcg@10", "ndcg@100", "recall@10", ...
    metric_names = []
    for k in ndcg_ks:
        metric_names.append(f"ndcg@{k}")
    for k in recall_ks:
        metric_names.append(f"recall@{k}")
    for k in mrr_ks:
        metric_names.append(f"mrr@{k}")

    if metric_weights is None:
        # uniform weights
        w = 1.0 / len(metric_names)
        metric_weights = {name: w for name in metric_names}
    else:
        # normalize provided weights to sum to 1
        s = sum(metric_weights.values())
        if s <= 0:
            raise ValueError("metric_weights must sum to > 0")
        metric_weights = {k: v / s for k, v in metric_weights.items()}

    # ---- Reward function used by GRPOTrainer ----
    def reward_fn(
        completions: List[str],
        query_id: List[str],
        prompts=None,
        completions_ids=None,
        trainer_state=None,
        **kwargs,
    ) -> List[float]:
        """
        Multi-metric NFcorpus reward:

        For each completion / query_id pair:
          - compute NDCG@k, Recall@k, MRR@k
          - combine them with metric_weights into a single scalar reward
        """
        if len(completions) == 0:
            return []

        # Encode completions as sparse queries
        query_emb = model.encode(
            completions,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )  # [num_completions, dim]

        scores = util.dot_score(query_emb, corpus_emb)  # [num_completions, num_docs]

        rewards: List[float] = []

        for row_idx, qid in enumerate(query_id):
            rel_docs = relevant_docs.get(str(qid))
            if not rel_docs:
                rewards.append(0.0)
                continue

            row_scores = scores[row_idx]  # [num_docs]

            # Top-K over all needed ks
            topk = torch.topk(row_scores, k=max_k)
            top_indices = topk.indices.tolist()
            top_cids = [corpus_ids[i] for i in top_indices]

            # Binary relevance vector for ranks 1..max_k
            is_rel = [1 if cid in rel_docs else 0 for cid in top_cids]
            num_rel = len(rel_docs)

            metric_values: Dict[str, float] = {}

            # ---- NDCG@k ----
            for k in ndcg_ks:
                dcg = 0.0
                for rank in range(k):
                    if is_rel[rank]:
                        dcg += 1.0 / math.log2(rank + 2)  # rank is 0-based
                idcg = idcg_per_qid.get(str(qid), {}).get(k, 0.0)
                if idcg <= 0.0:
                    ndcg = 0.0
                else:
                    ndcg = dcg / idcg
                metric_values[f"ndcg@{k}"] = float(ndcg)

            # ---- Recall@k ----
            for k in recall_ks:
                hits = sum(is_rel[:k])
                if num_rel > 0:
                    recall = hits / num_rel
                else:
                    recall = 0.0
                metric_values[f"recall@{k}"] = float(recall)

            # ---- MRR@k ----
            for k in mrr_ks:
                rr = 0.0
                for rank in range(k):
                    if is_rel[rank]:
                        rr = 1.0 / (rank + 1)
                        break
                metric_values[f"mrr@{k}"] = float(rr)

            # ---- Combine metrics into a single scalar reward ----
            reward = 0.0
            for name, value in metric_values.items():
                weight = metric_weights.get(name, 0.0)
                reward += weight * value

            print(" | ".join([f"{k}: {v:.3f}" for k,v in metric_values.items()]))
            rewards.append(float(reward))

        return rewards

    return reward_fn


# -----------------------------
# Dataset building
# -----------------------------

def build_train_dataset(
    queries: Dict[str, str],
    relevant_docs: Dict[str, Set[str]],
    nfcorpus_split: str = "train",
    max_queries: Optional[int] = None,
) -> Dataset:
    """
    Build a GRPO training dataset with:

        - 'prompt': instruction + original query text
        - 'query_id': NFcorpus qid (string)

    We only include qids that appear in the qrels for the specified split.
    """
    # Load the qrels split and restrict to those qids
    qrels_ds = load_dataset(CORPUS_QRELS, split=nfcorpus_split)
    qids_with_qrels = sorted({str(qid) for qid in qrels_ds["query-id"]})

    if max_queries is not None:
        qids_with_qrels = qids_with_qrels[:max_queries]

    prompts: List[str] = []
    query_ids: List[str] = []

    for qid in qids_with_qrels:
        if qid not in queries:
            continue
        qtext = queries[qid]
        # prompt = (
        #     "You are rewriting search queries to improve sparse retrieval quality "
        #     "over the NFcorpus collection.\n\n"
        #     f"Original query: {qtext}\n\n"
        #     "Rewrite this query to retrieve more relevant documents, while staying "
        #     "faithful to the original intent. Return only the rewritten query."
        # )
        prompts.append(qtext)
        query_ids.append(qid)

    print(f"[DATA] Built training dataset with {len(prompts)} prompts.")

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "query_id": query_ids,
        }
    )


# -----------------------------
# Config loading
# -----------------------------

def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    ir_cfg = IRConfig(**raw["ir"])
    train_cfg = TrainConfig(
        model_name=raw["model_name"],
        ir=ir_cfg,
        grpo=raw["grpo"],
    )
    return train_cfg


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config.yaml)",
    )
    args = parser.parse_args()

    # 1) Load config
    cfg = load_config(args.config)

    # 2) Load NFcorpus resources
    corpus, queries, relevant_docs = load_nfcorpus()

    # 3) Load SparseEncoder externally
    print(f"[IR] Loading SparseEncoder: {cfg.ir.sparse_model_name}")
    sparse_model = SparseEncoder(cfg.ir.sparse_model_name)

    # 4) Build GRPO train dataset
    train_dataset = build_train_dataset(
        queries=queries,
        relevant_docs=relevant_docs,
        nfcorpus_split=cfg.ir.nfcorpus_split,
        max_queries=cfg.ir.max_queries,
    )
    print("Total NFcorpus queries:", len(queries))              # should be 3237
    print("Train dataset prompts:", len(train_dataset))        # this is what drives steps/epoch

    # 5) Build reward function
    ir_reward = build_nfcorpus_multi_metric_reward(
        model=sparse_model,
        corpus=corpus,
        relevant_docs=relevant_docs,
        ndcg_ks=[10, 100, 1000],
        recall_ks=[100, 1000],
        mrr_ks=[100],
        metric_weights={
            "ndcg@10": 0.25,
            "ndcg@100": 0.25,
            "ndcg@1000": 0.15,
            "recall@100": 0.2,
            "recall@1000": 0.1,
            "mrr@100": 0.05
            # others will default to weight 0 if not listed
        },
        batch_size=64,
    )

    # 6) Build GRPOConfig from YAML
    grpo_args = GRPOConfig(**cfg.grpo)

    # 7) Create GRPOTrainer
    trainer = GRPOTrainer(
        model=cfg.model_name,
        args=grpo_args,
        reward_funcs=ir_reward,
        train_dataset=train_dataset,
    )

    # 8) Train
    trainer.train()


if __name__ == "__main__":
    main()

