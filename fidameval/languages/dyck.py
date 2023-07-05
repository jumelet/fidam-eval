import random
from typing import *

import numpy as np
import torch
from torch import Tensor

from .language import Corpus, Language, LanguageConfig


class DyckConfig(LanguageConfig):
    min_length: int
    max_length: int
    max_depth: int
    n_items: int

    start_token: bool = False
    p: float = 0.5
    q: float = 0.25


class Dyck(Language[DyckConfig]):
    @property
    def num_symbols(self):
        return self.config.n_items * 2

    def create_vocab(self) -> Dict[str, int]:
        return {
            (
                chr(x + 97)
                if x < self.config.n_items
                else chr(x - self.config.n_items + 65)
            ): x
            for x in range(self.config.n_items * 2 + self.config.start_token)
        }

    def create_corpus(self) -> Corpus:
        corpus = []
        num_loops = 0

        while (
            len(corpus) < self.config.corpus_size
            and num_loops < self.config.corpus_size * 10
        ):
            dyck = self.gen_dyck(self.config.max_length, self.config.max_depth)
            if len(dyck) > self.config.min_length and dyck not in corpus:
                if self.config.start_token:
                    dyck.insert(0, self.config.n_items * 2)
                corpus.append(dyck)

            num_loops += 1

        corpus = [torch.tensor(item) for item in corpus]

        if len(corpus) != self.config.corpus_size:
            print(len(corpus))

        return corpus

    def gen_dyck(self, length, depth) -> List[int]:
        prob = random.random()

        if length <= 0 or depth <= 0:
            return []

        if prob < self.config.p:
            s = self.gen_dyck(length - 2, depth - 1)
            x = random.choice(range(self.config.n_items))
            return [x, *s, x + self.config.n_items]
        elif self.config.p < prob < (self.config.p + self.config.q):
            s1 = self.gen_dyck(length, depth)
            s2 = self.gen_dyck(length, depth)
            if len(s1 + s2) < length:
                return s1 + s2

        return []

    def _create_corrupt_item(self, item: Tensor) -> Tensor:
        corrupt_item = item.clone()
        sen_len = corrupt_item.shape[0]

        start_idx = int(self.config.start_token)
        candidate_ids = set(range(start_idx, sen_len))

        if isinstance(self.config.corrupt_k, int):
            k = min(
                sen_len, self.config.corrupt_k
            )  # trim k for potentially shorter strings
        elif isinstance(self.config.corrupt_k, Iterable):
            k_candidates = [k for k in self.config.corrupt_k if k <= sen_len]
            k = random.choice(k_candidates) if k_candidates else sen_len
        else:
            k = sen_len

        for _ in range(k):
            idx = random.choice(list(candidate_ids))
            # possible replacement values (original item is removed)
            candidate_values = list(
                set(range(self.config.n_items * 2)) - {corrupt_item[idx].item()}
            )
            corrupt_item[idx] = random.choice(candidate_values)
            candidate_ids.remove(idx)

            if len(candidate_ids) == 0:
                break

        # make sure we did not accidentally create a string that is well formed too
        if self.is_dyck(corrupt_item.tolist()):
            return self._create_corrupt_item(item)

        return corrupt_item

    def is_dyck(self, sen: List[int]):
        stack = []
        for x in sen[int(self.config.start_token) :]:
            if x < self.config.n_items:
                stack.append(x)
            elif len(stack) == 0:
                return False
            else:
                y = stack.pop()
                if y != (x - self.config.n_items):
                    return False

        if len(stack) != 0:
            return False

        return True

    def item_depths(self, item: Iterable[int]):
        cur_depth = 0
        depths = []

        for x in item:
            depths.append(cur_depth)
            if x < self.config.n_items:
                cur_depth += 1
            else:
                cur_depth -= 1

        return depths

    def item_dep_lens(self, item: Iterable[int]):
        dep_lens = []
        stack = []

        for x in item:
            stack = [y + 1 for y in stack]

            if x < self.config.n_items:
                dep_lens.append(0)
                stack.append(0)
            else:
                dep_lens.append(stack.pop())

        return dep_lens

    def gen_baselines(
        self, sen_len: int, n_samples: int, well_formed: bool = False
    ) -> np.ndarray:
        if well_formed:
            if self.config.is_binary:
                sen_len_items = [
                    x[0].numpy()
                    for x in self.corpus
                    if x[0].shape[0] == sen_len and x[1] == 1
                ]
            else:
                sen_len_items = [
                    x.numpy() for x in self.corpus if x.shape[0] == sen_len
                ]
            baselines = random.choices(sen_len_items, k=n_samples)
            baselines = np.array(baselines)
        else:
            baselines = np.array(
                [
                    random.choices(range(self.config.n_items * 2), k=sen_len)
                    for _ in range(n_samples)
                ]
            )

        return baselines
