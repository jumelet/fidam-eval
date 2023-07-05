def palin_cond_prob(
    input_ids: torch.Tensor,
    removed_ids: Tuple[int],
    num_marginal_samples=1,
    use_positional_prior=False,
):
    baselines = []
    sen_len = len(input_ids)

    for _ in range(num_marginal_samples):
        baseline = torch.zeros_like(input_ids)
        for idx in range(sen_len):
            if idx in removed_ids:
                if idx < len(input_ids) // 2:
                    sym_range = range(language.config.n_items)
                else:
                    sym_range = range(
                        language.config.n_items, language.config.n_items * 2
                    )
                candidate_ids = list(set(sym_range) - {input_ids[idx].item()})

                if sen_len - idx - 1 in removed_ids:
                    if idx >= (len(input_ids) // 2):
                        continue
                    symbol = random.choice(candidate_ids)
                    baseline[idx] = symbol
                    baseline[sen_len - idx - 1] = symbol + language.config.n_items
                else:
                    baseline[idx] = random.choice(sym_range)
            else:
                baseline[idx] = input_ids[idx]
        baselines.append(baseline)

        return baselines


def palin_interventional(
    input_ids, removed_ids, num_marginal_samples=1, use_positional_prior=False
):
    baselines = []
    remaining_ids = tuple(i for i in range(len(input_ids)) if i not in removed_ids)

    for _ in range(num_marginal_samples):
        baseline = []
        for idx in range(len(input_ids)):
            if idx in remaining_ids:
                baseline.append(input_ids[idx].item())
            else:
                candidate_ids = list(
                    set(range(language.num_symbols)) - {input_ids[idx].item()}
                )
                if use_positional_prior:
                    weights = positional_distribution_dict[idx][candidate_ids]
                    symbol = random.choices(candidate_ids, weights=weights)[0]
                else:
                    symbol = random.choice(candidate_ids)
                baseline.append(symbol)
        baselines.append(torch.tensor(baseline))

    return baselines
