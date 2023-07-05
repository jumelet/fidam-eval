import random
from typing import *

import torch
from distribution import Distribution
from torch import Tensor


class Baseline:
    def __init__(self, language):
        self.language = language
        self.distribution = self.init_distribution()

    def init_distribution(self):
        return Distribution(self.language)

    def observational(
        self,
        input_ids: Tensor,
        removed_ids: Tuple[int],
        num_marginal_samples: int,
        use_positional_prior: bool,
    ):
        raise NotImplementedError

    def interventional(
        self,
        input_ids: Tensor,
        removed_ids: Tuple[int],
        num_marginal_samples: int,
        use_positional_prior: bool,
    ):
        raise NotImplementedError

    def joint_interventional(self, input_ids, removed_ids, num_marginal_samples=1):
        baselines = []
        remaining_ids = tuple(i for i in range(len(input_ids)) if i not in removed_ids)

        distribution = self.positive_distribution(input_ids, indices=removed_ids)

        for sample in random.choices(distribution, k=num_marginal_samples):
            baseline = []
            for idx in range(len(input_ids)):
                if idx in remaining_ids:
                    baseline.append(input_ids[idx].item())
                else:
                    baseline.append(sample[idx].item())
            baselines.append(torch.tensor(baseline))

        return baselines

    @staticmethod
    def invert(input_ids: Tensor):
        raise NotImplementedError
