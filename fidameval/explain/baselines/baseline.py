import random
from typing import *

import torch
from torch import Tensor

from .distribution import Distribution


class Baseline:
    def __init__(self, language, model, current_baseline: Optional[str] = None):
        self.language = language
        self.model = model
        self.distribution: Distribution = self.init_distribution()
        self.current_baseline: Optional[str] = current_baseline

    def init_distribution(self) -> Distribution:
        return Distribution(self.language)

    @property
    def is_marginal(self):
        marginal_baselines = [
            "observational",
            "interventional",
            "joint_interventional",
            "interventional_positional",
        ]

        return self.current_baseline in marginal_baselines

    @property
    def baseline_fn(self):
        assert self.current_baseline is not None, "current_baseline has not been set"
        assert hasattr(self, self.current_baseline), "current_baseline is unknown"

        return getattr(self, self.current_baseline)

    def zero(self, input_ids: Tensor) -> Tensor:
        return torch.zeros_like(input_ids) + self.model.zero_idx

    def negative(self, input_ids: Tensor) -> Tensor:
        return random.choice(self.distribution.negative_item_distribution(input_ids))

    def positive(self, input_ids: Tensor) -> Tensor:
        return random.choice(self.distribution.positive_item_distribution(input_ids))

    def independent(self, input_ids: Tensor) -> Tensor:
        baseline = [
            random.choice(list(set(range(self.language.num_symbols)) - {x}))
            for x in input_ids
        ]

        return torch.tensor(baseline)

    def positional(self, input_ids: Tensor) -> Tensor:
        baseline = [
            random.choices(
                range(self.language.num_symbols), self.distribution.positional[idx]
            )[0]
            for idx in range(len(input_ids))
        ]

        return torch.tensor(baseline)

    def interventional(
        self,
        input_ids: Tensor,
        removed_ids: Tuple[int],
        num_marginal_samples: int,
        use_positional_prior: bool = False,
    ):
        raise NotImplementedError

    def interventional_positional(
        self,
        input_ids: Tensor,
        removed_ids: Tuple[int],
        num_marginal_samples: int,
    ):
        return self.interventional(
            input_ids, removed_ids, num_marginal_samples, use_positional_prior=True
        )

    def joint_interventional(self, input_ids, removed_ids, num_marginal_samples=1):
        baselines = []
        remaining_ids = tuple(i for i in range(len(input_ids)) if i not in removed_ids)

        distribution = self.distribution.positive_item_distribution(
            input_ids, indices=removed_ids
        )

        for sample in random.choices(distribution, k=num_marginal_samples):
            baseline = []
            for idx in range(len(input_ids)):
                if idx in remaining_ids:
                    baseline.append(input_ids[idx].item())
                else:
                    baseline.append(sample[idx].item())
            baselines.append(torch.tensor(baseline))

        return baselines

    def observational(
        self,
        input_ids: Tensor,
        removed_ids: Tuple[int],
        num_marginal_samples: int,
        use_positional_prior: bool,
    ):
        raise NotImplementedError

    @staticmethod
    def invert(input_ids: Tensor):
        raise NotImplementedError
