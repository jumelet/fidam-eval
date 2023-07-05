from typing import *

import torch
from torch import Tensor

from fidameval.utils import DEVICE, Config


class TokenizerConfig(Config):
    cls_token = "[CLS]"
    mask_token = "<mask>"
    pad_token = "<pad>"
    unk_token = "<unk>"
    unk_threshold: Optional[int] = None
    add_cls: bool = False
    masked_lm: bool = False
    sep_token: str = " "


class Vocab(dict):
    def __init__(self, unk_idx: int, pad_idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unk_idx = unk_idx
        self.pad_idx = pad_idx

    def __missing__(self, key):
        return self.unk_idx


class Tokenizer:
    """ White space tokenizer """

    def __init__(self, config: TokenizerConfig):
        self.config = config

        self.idx2token: List[str] = [self.config.unk_token, self.config.pad_token]
        self.token2idx: Vocab = Vocab(0, 1)

    @property
    def cls_idx(self) -> Optional[int]:
        return self.token2idx.get(self.config.cls_token, None)

    @property
    def mask_idx(self) -> Optional[int]:
        return self.token2idx.get(self.config.mask_token, None)

    @property
    def unk_idx(self) -> int:
        return self.token2idx.unk_idx

    @property
    def pad_idx(self) -> int:
        return self.token2idx.pad_idx

    def create_vocab(self, str_corpus: List[str], pos_dict=None, is_binary=False):
        """Creates the vocab dictionary.

        Parameters
        ----------
        str_corpus : List[str]


        """
        if self.config.masked_lm:
            self.idx2token.append(self.config.mask_token)

        if self.config.add_cls:
            self.idx2token.append(self.config.cls_token)

        if self.config.unk_threshold is not None:
            all_tokens = (w for s in str_corpus for w in s.split(self.config.sep_token))
            distribution = Counter(all_tokens)
            self.idx2token.extend(
                [
                    token
                    for token, counts in distribution.items()
                    if counts > self.config.unk_threshold
                ]
            )
        else:
            unique_tokens = set(
                w for s in str_corpus for w in s.split(self.config.sep_token)
            )
            self.idx2token.extend(list(unique_tokens))

        if pos_dict is not None:
            unique_pos = set(
                pos_tag for pos_tags in pos_dict.values() for pos_tag in pos_tags
            )
            self.idx2token.extend(list(unique_pos))

        self.token2idx.update({x: idx for idx, x in enumerate(self.idx2token)})

    def tokenize(
        self,
        item: Union[str, List[str]],
    ) -> Tensor:
        """Tokenization method.

        Parameters
        ----------
        item : str | List[str]
            Input string, or input string that has already been split.

        Returns
        -------
        token_ids : Tensor
            Tokenized tensor of token ids.
        """
        if isinstance(item, str):
            item = item.split(self.config.sep_token)

        token_ids = []
        if self.config.add_cls and item[0] != self.config.cls_token:
            token_ids.append(self.cls_idx)

        token_ids.extend([self.token2idx[w] for w in item])

        return torch.tensor(token_ids, device=DEVICE)

    def translate(self, item: Tensor, omit_cls: bool = False) -> str:
        sen_list = [self.idx2token[idx] for idx in item.tolist()]
        if omit_cls:
            sen_list = sen_list[1:]

        return self.config.sep_token.join(sen_list)

    def make_masked_post_hoc(self) -> None:
        """Transform Tokenizer object to masked tokenizer after it has
        already been instantiated.
        """
        self.config.masked_lm = True
        if self.config.mask_token not in self.token2idx:
            self.token2idx[self.config.mask_token] = len(self.token2idx)
            self.idx2token.append(self.config.mask_token)

    def add_cls_post_hoc(self) -> None:
        self.config.add_cls = True

        if self.config.cls_token not in self.token2idx:
            self.token2idx[self.config.cls_token] = len(self.token2idx)
            self.idx2token.append(self.config.cls_token)
