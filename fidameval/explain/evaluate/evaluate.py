from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from fidameval.explain.baselines import Baseline
from fidameval.explain.fidams import *

from .compute import compute_interactions
from .metrics import calc_all_metrics


def evaluate_all_configuration(
    language,
    model,
    dep_fn,
    baseline_class=Baseline,
    num_marginal_samples=5,
    expectation_samples=5,
    evaluation_corpus_size=None,
):
    interaction_fns = [
        (group_ablation, "Group Ablation", False, {}),
        (archipelago, "Archipelago", True, {}),
        (sii, "SII", False, {"verbose": False, "processes": 1}),
        (stii, "STII", False, {"verbose": False, "processes": 1}),
        (hessian, "Hessian", True, {}),
        (hessian_input, "Hessian * Input", True, {}),
        (integrated_hessians, "IH", True, {}),
    ]

    baseline_types = [
        "zero",
        "invert",
        "positive",
        "negative",
        "interventional",
        "interventional_positional",
        "joint_interventional",
        "observational",
    ]

    evaluation_corpus = [sen for sen, label in language.corpus if label == 1]
    evaluation_corpus = evaluation_corpus[:evaluation_corpus_size]

    raw_df = evaluate_interaction_fns(
        model,
        language,
        interaction_fns,
        dep_fn,
        baseline_types,
        evaluation_corpus,
        baseline_class=baseline_class,
        expectation_samples=expectation_samples,
        num_marginal_samples=num_marginal_samples,
    )

    for metric in ["row_rank"]:
        df = gen_latex_table(raw_df, metric, baseline_types, digits=3)

    return df


def evaluate_interaction_fns(
    model,
    language,
    interaction_fns: List[Tuple[Any, str, bool, Dict[str, Any]]],
    deps_fn,
    baseline_types: List[str],
    evaluation_corpus: List[Tensor],
    baseline_class: Type[Baseline] = Baseline,
    expectation_samples: int = 1,
    num_marginal_samples: int = 1,
) -> List[Any]:
    raw_dataframe = []

    for interaction_fn, name, use_embs, kwargs in interaction_fns:
        scores = defaultdict(list)

        for item in tqdm(evaluation_corpus):
            deps = deps_fn(item)

            for baseline_type in baseline_types:
                baseline = baseline_class(
                    language, model, current_baseline=baseline_type
                )

                if (
                    baseline_type != "zero"
                    and interaction_fn in [hessian, hessian_input]
                ) or (
                    baseline.is_marginal
                    and interaction_fn in [integrated_hessians, archipelago]
                ):
                    continue

                all_interactions = []

                if baseline_type not in ["zero", "invert"] and name not in [
                    "Archipelago",
                    "IH",
                ]:
                    kwargs["num_marginal_samples"] = num_marginal_samples
                if baseline_type not in [
                    "positive",
                    "negative",
                    "independent",
                    "positional",
                ]:
                    expectation_samples = min(expectation_samples, 1)
                if name == "Archipelago":
                    kwargs["top_k"] = len(deps)

                for k in range(expectation_samples):
                    interactions = compute_interactions(
                        interaction_fn,
                        model,
                        item,
                        baseline,
                        use_embs,
                        **kwargs,
                    )
                    all_interactions.append(interactions)

                avg_interactions = torch.mean(torch.stack(all_interactions), dim=0)
                score_dict = calc_all_metrics(avg_interactions, deps)

                scores[baseline_type].append(score_dict)

        raw_dataframe.append([name, scores])

    return raw_dataframe


def gen_latex_table(
    raw_df, metric, baseline_types, do_print=True, digits=3
) -> pd.DataFrame:
    raw_mean_df = [
        [name]
        + [
            np.mean([scores[metric] for scores in all_scores[baseline_type]])
            for baseline_type in baseline_types
        ]
        for name, all_scores in raw_df
    ]
    df = pd.DataFrame(raw_mean_df, columns=["method"] + baseline_types)
    latex_str = df.to_latex(index=False, float_format=lambda x: f"%.{digits}f" % x)

    max_vals = [f"{x:.{digits}f}" for x in df.max()[1:].values]
    for val in set(max_vals):
        latex_str = latex_str.replace(val, f"\\textbf{{{val}}}")
    latex_str = latex_str.replace(" * ", "$\\times$")
    latex_str = latex_str.replace("NaN", "--")

    if do_print:
        print(latex_str)

    return df
