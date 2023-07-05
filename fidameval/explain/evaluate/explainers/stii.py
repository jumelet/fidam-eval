import multiprocessing
from itertools import chain, combinations, product
from math import comb, factorial

import torch
from tqdm import *


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def sii(model, input_ids, baseline_ids, **kwargs):
    return stii(model, input_ids, baseline_ids, taylor=False, **kwargs)


def compute_shapley_value(i, j, output_matrix, N, k, n, normalising_terms, taylor):
    val = 0

    for T in powerset(N - {i, j}):
        idx_T = sum(2 ** x for x in T)
        idx_i = 2 ** i
        idx_j = 2 ** j

        # We compute the loop over W in S (eq.3) directly, much more efficient
        # When generalising STII to any k this needs to be modified
        delta = (
            output_matrix[idx_T]
            - output_matrix[idx_T + idx_i]
            - output_matrix[idx_T + idx_j]
            + output_matrix[idx_T + idx_i + idx_j]
        )

        val += delta * normalising_terms[len(T)]

    if taylor:
        val *= k / n

    return val


def stii(
    model,
    input_ids,
    baseline_ids,
    taylor=True,
    batch_size=1024,
    verbose=False,
    background_distribution=None,
    use_positional_prior=False,
    num_marginal_samples=1,
    processes=1,
):
    N = set(range(len(input_ids)))
    k = 2
    n = len(N)
    num_coalitions = 2 ** n

    pair_ids = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sti_matrix = torch.zeros(len(input_ids), len(input_ids))
    iterator = tqdm_notebook if verbose else lambda x, *args, **kwargs: x

    output_matrix = torch.zeros(num_marginal_samples, num_coalitions)

    # We precompute the model outputs for all possible coalitions
    for sample_idx in range(num_marginal_samples):
        input_matrix = torch.zeros(num_coalitions, n, dtype=torch.long)

        # fuse input and baseline ids
        for S in iterator(powerset(range(n))):
            complement = list(N - set(S))
            S = list(S)
            # Coalitions are indexed as a binary number based on present features
            idx = sum(2 ** x for x in S)

            if background_distribution is None:
                input_matrix[idx, S] = input_ids[S]
                input_matrix[idx, complement] = baseline_ids[complement]
            else:
                input_matrix[idx] = background_distribution(
                    input_ids, S, use_positional_prior=use_positional_prior
                )[0]

        with torch.no_grad():
            for idx in iterator(range(0, num_coalitions, batch_size)):
                batch_input = input_matrix[idx : idx + batch_size]
                batch_embs = model.create_inputs_embeds(batch_input)
                batch_output = model(inputs_embeds=batch_embs)
                output_matrix[sample_idx, idx : idx + batch_size] = batch_output

    # Average over sample dimension
    output_matrix = output_matrix.mean(0)

    if taylor:
        normalising_terms = {t: 1 / comb(n - 1, t) for t in range(n - 1)}
    else:
        normalising_terms = {
            t: ((factorial(n - t - k) * factorial(t)) / factorial(n - k + 1))
            for t in range(n - 1)
        }

    if processes > 1:
        # Shapley values can be computed in parallel processes
        with multiprocessing.Pool(processes) as pool:
            params = [
                (i, j, output_matrix, N, k, n, normalising_terms, taylor)
                for i, j in pair_ids
            ]
            jobs = [pool.apply_async(compute_shapley_value, p) for p in params]
            for (i, j), job in iterator(zip(pair_ids, jobs), total=len(pair_ids)):
                val = job.get()
                sti_matrix[i, j] = val
                sti_matrix[j, i] = val
    else:
        for i, j in iterator(pair_ids):
            val = compute_shapley_value(
                i, j, output_matrix, N, k, n, normalising_terms, taylor
            )
            sti_matrix[i, j] = val
            sti_matrix[j, i] = val

    return sti_matrix
