import warnings
from collections import defaultdict
from typing import *

import numpy as np
import torch
from torch import Tensor


class Distribution:
    def __init__(self, language, max_sen_len: Optional[int] = None):
        self.language = language

        self.negative_items: Dict[int, Tensor] = self.create_len_distribution(
            0, max_sen_len=max_sen_len
        )
        self.positive_items: Dict[int, Tensor] = self.create_len_distribution(
            1, max_sen_len=max_sen_len
        )
        self.positional: Dict[int, np.array] = self.create_positional(language.corpus)

    def positive_item_distribution(self, item, indices=None):
        mask = self.positive_items[len(item)] != item

        if indices is not None:
            mask = mask[:, indices]
        mask = mask.all(-1)

        return self.positive_items[len(item)][mask]

    def negative_item_distribution(self, item, indices=None):
        mask = self.negative_items[len(item)] != item

        if indices is not None:
            mask = mask[:, indices]
        mask = mask.all(-1)

        return self.negative_items[len(item)][mask]

    def create_len_distribution(
        self, label: int, max_sen_len: Optional[int] = None
    ) -> Dict[int, Tensor]:
        """Creates dict that maps each sentence length to all corpus
        items of that length.
        """
        if self.language.config.is_binary:
            label_items = [
                item for item, item_label in self.language.corpus if item_label == label
            ]
        else:
            if label == 0:
                warnings.warn(
                    "Corpus contains no negative items for negative corpus distribution"
                )
            label_items = self.language.corpus
        len_distribution: Dict[int, List[Tensor]] = defaultdict(list)
        for item in label_items:
            if max_sen_len is None or len(item) <= max_sen_len:
                len_distribution[len(item)].append(item)

        len_tensor_distribution: Dict[int, Tensor] = {}
        for length, items in len_distribution.items():
            len_tensor_distribution[length] = torch.stack(items, 0)

        return len_tensor_distribution

    def create_positional(
        self,
        corpus: List[Tuple[Tensor, int]],
    ) -> Dict[int, np.array]:
        if self.language.config.is_binary:
            base_corpus = [x for x, label in corpus if label == 1]
        else:
            base_corpus = corpus

        max_sen_len = max(map(len, base_corpus))

        distribution_dict = {
            idx: np.zeros(self.language.num_symbols) for idx in range(max_sen_len)
        }

        for item in base_corpus:
            for idx, symbol in enumerate(item.tolist()):
                distribution_dict[idx][symbol] += 1

        # Normalise
        for idx, counts in distribution_dict.items():
            distribution_dict[idx] /= sum(counts)

        return distribution_dict
