import numpy as np
import torch
from sklearn.metrics import ndcg_score


def deps_pos(interactions, deps):
    """Returns the ratio of *positive* dependency interactions"""
    pos_ids = np.argwhere(interactions > 0).tolist()
    pos_ids = set(zip(*pos_ids))

    return len(set(deps) & pos_ids) / len(deps)


def mean_rank(interactions, deps):
    """Returns the rank of each dependency interaction based on interaction scores

    A perfect ordering would have ranks [0, 1, 2, ...]
    The ranks are normalised between 0 and 1 (1 being best)
    """
    sorted_ids = np.argsort(-interactions.flatten().numpy())

    n = interactions.shape[0]
    sorted_ids = [tuple(sorted((i % n, i // n))) for i in sorted_ids if i % n <= i // n]

    ranks = [sorted_ids.index(dep) for dep in deps]

    total = len(sorted_ids)

    mean_ranks = np.mean(ranks)
    num_deps = len(deps)
    # Scale between 0 and 1
    norm_rank = (
        (mean_ranks - ((num_deps - 1) / 2))
        * ((total - 1) / (total - num_deps))
        / (total - 1)
    )

    return 1 - norm_rank


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
