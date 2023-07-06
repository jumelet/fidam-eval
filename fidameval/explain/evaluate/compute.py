from torch import Tensor

from fidameval.explain.baselines import Baseline


def compute_interactions(
    interaction_fn,
    model,
    input_ids: Tensor,
    baseline: Baseline,
    use_embs: bool,
    **kwargs
):
    if baseline.is_marginal:
        assert not use_embs, "marginal not compatible with emb based methods"
        baseline_ids = None
        background_distribution = baseline.baseline_fn
    else:
        baseline_ids = baseline.baseline_fn(input_ids)
        background_distribution = None

    if use_embs:
        embs = model.create_inputs_embeds(input_ids)
        baseline_embs = model.create_inputs_embeds(baseline_ids)
        interactions = interaction_fn(model, embs, baseline_embs, **kwargs)
    else:
        interactions = interaction_fn(
            model,
            input_ids,
            baseline_ids,
            background_distribution=background_distribution,
            **kwargs,
        )

    return interactions.squeeze()
