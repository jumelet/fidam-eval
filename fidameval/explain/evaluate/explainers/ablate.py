import torch


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def group_ablation(
    model,
    input_ids,
    baseline_ids,
    coalitions=None,
    use_embs=False,
    exclude_diagonal=False,
    keep=False,
    discard=False,
    background_distribution=None,
    num_marginal_samples=1,
    use_positional_prior=False,
):
    if coalitions is None:
        coalitions = [[idx] for idx in range(len(input_ids))]

    ablation_matrix = torch.zeros(len(coalitions), len(coalitions))

    for i, coal_i in enumerate(coalitions):
        for j, coal_j in enumerate(
            coalitions[i + exclude_diagonal :], start=i + exclude_diagonal
        ):
            coal_ij = list(set(flatten(coal_i + coal_j)))

            if discard:
                mixed_input = torch.tensor(
                    [x for k, x in enumerate(input_ids) if k not in coal_ij]
                )
                if len(mixed_input) == 0:
                    break
            elif background_distribution is not None:
                mixed_input = background_distribution(
                    input_ids,
                    coal_ij,
                    num_marginal_samples=num_marginal_samples,
                    use_positional_prior=use_positional_prior,
                )
                mixed_input = torch.stack(mixed_input)
            elif keep:
                mixed_input = torch.clone(baseline_ids)
                mixed_input[:, coal_ij] = input_ids[:, coal_ij]
            else:
                mixed_input = torch.clone(input_ids)
                mixed_input[:, coal_ij] = baseline_ids[:, coal_ij]

            with torch.no_grad():
                if not use_embs:
                    mixed_input = model.create_inputs_embeds(mixed_input)
                output = model(inputs_embeds=mixed_input).mean()
                if background_distribution is not None and num_marginal_samples > 1:
                    output = output.mean()
                ablation_matrix[i, j] = output
                ablation_matrix[j, i] = output

    return ablation_matrix
