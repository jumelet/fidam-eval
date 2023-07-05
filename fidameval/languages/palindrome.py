import itertools
import random
from typing import *

import numpy as np
import torch
from torch import Tensor

from .language import Corpus, Language, LanguageConfig


class PalindromeConfig(LanguageConfig):
    sen_len: Union[int, List[int]]
    n_items: int
    use_separator: bool = False
    map_homomorphic: bool = False
    palindrome_ids: Optional[List[int]] = None
    mode: str = "mirror"


class Palindrome(Language):
    def __repr__(self):
        summary = [
            self.config.mode,
            "-".join(map(str, self.config.sen_len)),
            str(self.config.n_items),
            str(int(self.config.is_binary)),
            str(int(self.config.use_separator)),
            str(int(self.config.map_homomorphic)),
        ]

        if self.config.is_binary:
            corrupt_k = (
                str(self.config.corrupt_k)
                if self.config.corrupt_k is not None
                else "all"
            )
            summary.append(corrupt_k)

        return "_".join(summary)

    @property
    def num_symbols(self):
        return self.config.n_items * (1 + self.config.map_homomorphic) + self.config.use_separator

    def create_corpus(self) -> Corpus:
        all_corpora = [
            self.create_sen_len_corpus(length) for length in self.config.sen_len
        ]

        if (
            self.config.corpus_size is None
            or sum(map(len, all_corpora)) < self.config.corpus_size
        ):
            corpus = list(itertools.chain.from_iterable(all_corpora))
        else:
            corpus = []
            for i, sen_len_corpus in enumerate(all_corpora):
                max_corpus_length = (self.config.corpus_size - len(corpus)) // (
                    len(self.config.sen_len) - i
                )
                if len(sen_len_corpus) < max_corpus_length:
                    corpus.extend(sen_len_corpus)
                else:
                    corpus.extend(random.sample(sen_len_corpus, max_corpus_length))

        return corpus

    def create_sen_len_corpus(self, sen_len: int) -> Corpus:
        corpus_size = self.config.corpus_size
        if corpus_size is None or self.config.n_items ** sen_len < corpus_size:
            onsets = itertools.product(range(self.config.n_items), repeat=sen_len)
        else:
            # Sample palindromes without any duplicates
            onsets = set()
            for _ in range(corpus_size):
                sample = tuple(
                    np.random.choice(range(self.config.n_items), size=sen_len)
                )
                while sample in onsets:
                    sample = tuple(
                        np.random.choice(range(self.config.n_items), size=sen_len)
                    )
                onsets.add(sample)

        corpus = [torch.tensor(item + self.gen_second_half(item)) for item in onsets]

        if self.config.palindrome_ids is not None:
            corpus = list(map(self.set_palindrome_ids, corpus))

        return corpus

    def gen_second_half(self, item):
        second_half_direction = {
            "mirror": -1,
            "copy": 1,
        }[self.config.mode]

        second_half = item[::second_half_direction]

        if self.config.map_homomorphic:
            second_half = tuple(
                x + self.config.n_items + self.config.use_separator for x in second_half
            )

        if self.config.use_separator:
            second_half = (self.config.n_items,) + second_half

        return second_half

    def set_palindrome_ids(self, item: torch.Tensor) -> torch.Tensor:
        """ Only retains palindromic dependencies between specific indices. """
        first_half_idx = len(item) // 2
        for idx in range(first_half_idx):
            neg_idx = idx - first_half_idx
            if (
                idx not in self.config.palindrome_ids
                and neg_idx not in self.config.palindrome_ids
            ):
                candidate_values = set(range(self.config.n_items)) - {item[idx].item()}
                new_value = random.choice(tuple(candidate_values))
                item[idx] = new_value

        return item

    def _create_corrupt_item(self, item: Tensor) -> Tensor:
        sen_len = len(item) // 2

        # possible ids that can be corrupted (first half only)
        if self.config.palindrome_ids is not None:
            candidate_ids = [
                x if x >= 0 else sen_len + x
                for x in self.config.palindrome_ids
                if x < sen_len
            ]
        else:
            candidate_ids = set(range(sen_len))

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
                set(range(self.config.n_items)) - {item[idx].item()}
            )
            item[idx] = random.choice(candidate_values)
            candidate_ids.remove(idx)

            if len(candidate_ids) == 0:
                break

        return item

    def gen_baselines(self, sen_len: int, n_samples: int) -> np.ndarray:
        baselines = np.array(
            [
                random.choices(range(self.config.n_items), k=sen_len)
                for _ in range(n_samples)
            ]
        )

        return baselines

    def create_vocab(self) -> Dict[str, int]:
        return {
            (
                chr(x + 97)
                if x < self.config.n_items
                else chr(x - self.config.n_items + 65)
            ): x
            for x in range(self.config.n_items * 2)
        }
