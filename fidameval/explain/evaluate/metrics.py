from typing import Dict

import numpy as np
import torch
from sklearn.metrics import ndcg_score


def calc_all_metrics(interactions, deps) -> Dict[str, float]:
    score_dict = {
        "pos": deps_pos(interactions, deps),
        "ndcg": row_ndcg(interactions, deps),
        "row_rank": row_ranks(interactions, deps),
    }

    return score_dict


def deps_pos(interactions, deps):
    """Returns the ratio of *positive* dependency interactions"""
    pos_ids = np.argwhere(interactions > 0).tolist()
    pos_ids = set(zip(*pos_ids))

    return len(set(deps) & pos_ids) / len(deps)


def row_ndcg(interactions, deps):
    """Returns the average row-wise NDCG scores."""
    assert interactions.ndim == 2
    deps_dict = dict(deps)

    ndcgs = []

    for i in range(interactions.shape[0]):
        if i in deps_dict:
            gold = np.zeros((1, interactions.shape[1]))
            gold[0, deps_dict[i]] = 1

            ndcgs.append(ndcg_score(gold, interactions[i].view(1, -1).numpy()))

    return np.mean(ndcgs)


def row_ranks(interactions, deps):
    topk = torch.topk(-interactions, k=interactions.shape[-1]).indices
    row_ranks = []

    for i, j in dict(deps).items():
        row_rank = topk[i].tolist().index(j)
        row_ranks.append(row_rank / (interactions.shape[-1] - 1))

    return np.mean(row_ranks)
