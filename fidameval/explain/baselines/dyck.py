import random
from typing import *

import torch
from baseline import Baseline
from torch import Tensor


class DyckBaseline(Baseline):
    all_dyck_strings: Dict[int, Tensor] = {}

    def interventional(
        self,
        input_ids,
        removed_ids,
        num_marginal_samples=1,
        use_positional_prior=False,
    ):
        baselines = []
        remaining_ids = tuple(i for i in range(len(input_ids)) if i not in removed_ids)

        for _ in range(num_marginal_samples):
            baseline = []
            for idx in range(len(input_ids)):
                if idx in remaining_ids:
                    baseline.append(input_ids[idx].item())
                else:
                    candidate_ids = list(set(range(4)) - {input_ids[idx].item()})
                    if use_positional_prior:
                        weights = self.distribution.positional[idx][candidate_ids]
                        symbol = random.choices(candidate_ids, weights=weights)[0]
                    else:
                        symbol = random.choice(candidate_ids)
                    baseline.append(symbol)
            baselines.append(torch.tensor(baseline))

        return baselines

    def observational(
        self,
        input_ids: torch.Tensor,
        removed_ids: Tuple[int],
        num_marginal_samples=1,
        use_positional_prior=False,
    ):
        num_pairs = len(input_ids) // 2

        # Dynamically create full set of all Dyck strings
        if num_pairs not in self.all_dyck_strings:
            self.all_dyck_strings[num_pairs] = self.gen_dyck(num_pairs)

        remaining_ids = tuple(i for i in range(len(input_ids)) if i not in removed_ids)

        skip_input_mask = self.all_dyck_strings[num_pairs] == input_ids
        mask = skip_input_mask[:, remaining_ids].all(1) & ~skip_input_mask[
            :, removed_ids
        ].any(1)
        subset: Tensor = self.all_dyck_strings[num_pairs][mask]

        if len(subset) > 0:
            baselines = random.choices(subset, k=num_marginal_samples)
        else:
            baselines = self.interventional(
                input_ids,
                removed_ids,
                num_marginal_samples=num_marginal_samples,
                use_positional_prior=use_positional_prior,
            )

        return baselines

    def gen_dyck(self, n):
        memoise = {0: [""], 1: ["aA", "bB"]}

        strings = self.gen_dyck_rec(n, memoise)
        tokenized = [[self.language.token2idx[x] for x in item] for item in strings]

        return torch.tensor(tokenized)

    def gen_dyck_rec(self, n, memoise):
        if n in memoise:
            return memoise[n]

        memoise[n] = []

        for i in range(1, n + 1):
            between = self.gen_dyck_rec(i - 1, memoise)
            after = self.gen_dyck_rec(n - i, memoise)
            for b in between:
                for a in after:
                    memoise[n].append("a" + b + "A" + a)
                    memoise[n].append("b" + b + "B" + a)

        return memoise[n]

    @staticmethod
    def invert(input_ids):
        return torch.tensor([{0: 1, 1: 0, 2: 3, 3: 2}[x.item()] for x in input_ids])
