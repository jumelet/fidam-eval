from explainers.ablate import group_ablation
from explainers.archipelago import archipelago
from explainers.hessian import hessian, hessian_input
from explainers.ih import integrated_hessians
from explainers.stii import sii, stii
from sklearn.metrics import ndcg_score


def compute_interactions(
    explainer,
    model,
    item,
    baseline_type,
    use_embs,
    cond_prob_fn=None,
    interventional_prob_fn=None,
    **kwargs
):
    if baseline_type == "zero":
        baseline = torch.zeros_like(item) + 4
    elif baseline_type == "negative":
        baseline = random.choice(negative_distribution(item))
    elif baseline_type == "positive":
        baseline = random.choice(positive_distribution(item))
    elif baseline_type == "independent":
        baseline = torch.tensor(
            [random.choice(list(set(range(4)) - {x})) for x in item]
        )
    elif baseline_type == "positional":
        baseline = torch.tensor(
            [
                random.choices(
                    range(language.num_symbols), positional_distribution_dict[idx]
                )[0]
                for idx in range(len(item))
            ]
        )
    elif baseline_type == "invert":
        baseline = invert(item)
    elif baseline_type == "conditional":
        assert cond_prob_fn is not None
        assert not use_embs, "conditional not compatible with emb based methods"
        kwargs["background_distribution"] = cond_prob_fn
        baseline = None
    elif baseline_type == "interventional_independent":
        assert interventional_prob_fn is not None
        assert not use_embs, "marginal not compatible with emb based methods"
        kwargs["background_distribution"] = interventional_prob_fn
        kwargs["use_positional_prior"] = False
        baseline = None
    elif baseline_type == "interventional_positional":
        assert interventional_prob_fn is not None
        assert not use_embs, "interventional not compatible with emb based methods"
        kwargs["background_distribution"] = interventional_prob_fn
        kwargs["use_positional_prior"] = True
        baseline = None
    elif baseline_type == "joint_interventional":
        kwargs["background_distribution"] = joint_interventional
        baseline = None
    else:
        raise ValueError(baseline_type)

    if use_embs:
        embs = model.create_inputs_embeds(item)
        baseline_embs = model.create_inputs_embeds(baseline)
        interactions = explainer(model, embs, baseline_embs, **kwargs)
    else:
        interactions = explainer(model, item, baseline, **kwargs)

    return interactions.squeeze()
