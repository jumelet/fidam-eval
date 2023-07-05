import itertools
import random
from typing import *

import nltk
import torch
from nltk import PCFG as nltk_PCFG
from nltk import Nonterminal, Production
from nltk.parse import ChartParser
from nltk.parse import IncrementalLeftCornerChartParser as Parser
from torch import Tensor
from tqdm import tqdm

from .language import Corpus, Language, LanguageConfig


class PCFGConfig(LanguageConfig):
    grammar_file: str
    max_length: int
    max_depth: int
    min_length: int = 0
    corrupt_grammar: Optional[Union[str, Callable[[Corpus], Corpus]]] = None
    start: Optional[str] = None
    generation_factor: int = 10
    verbose: bool = True
    generate_strings: bool = True


class PCFG(Language[PCFGConfig]):
    def __repr__(self):
        return str(self.grammar)

    def create_corpus(self) -> List[str]:
        corpus = self._generate_corpus(self.grammar)

        return corpus

    def create_grammar(self) -> nltk_PCFG:
        with open(self.config.grammar_file) as f:
            raw_grammar = f.read()
        grammar = nltk_PCFG.fromstring(raw_grammar)

        if self.config.start is not None:
            grammar._start = Nonterminal(self.config.start)

        grammar._lhs_prob_index = {}
        for lhs in grammar._lhs_index.keys():
            lhs_probs = [prod.prob() for prod in grammar.productions(lhs=lhs)]
            grammar._lhs_prob_index[lhs] = lhs_probs

        return grammar

    def _generate_corpus(self, grammar: nltk_PCFG) -> List[str]:
        """
        We divide the generation in an inner and outer loop:
        The outer loop sets up a new generation procedure, the inner loop
        determines how many items we sample from a top-down approach,
        This outer/inner division appears to yield the least redundant generation.
        """
        str_corpus = set()

        total = self.config.corpus_size * self.config.generation_factor

        for _ in tqdm(range(total)):
            tree = generate_tree(grammar, depth=self.config.max_depth)
            item = tree.leaves()
            item_len = len(item)

            if self.config.min_length < item_len < self.config.max_length:
                str_item = self.tokenizer.config.sep_token.join(item)
                if not self.config.allow_duplicates and str_item in str_corpus:
                    continue

                str_corpus.add(str_item)
                self.tree_corpus[str_item] = tree

            if len(str_corpus) >= self.config.corpus_size:
                return list(str_corpus)

        return list(str_corpus)

    def append_corrupt_corpus(self, corpus: List[Tensor]) -> List[Tuple[Tensor, int]]:
        assert self.config.corrupt_grammar is not None

        if isinstance(self.config.corrupt_grammar, str):
            with open(self.config.corrupt_grammar) as f:
                raw_grammar = f.read()
            corrupt_grammar = nltk_PCFG.fromstring(raw_grammar)

            if self.config.start is not None:
                corrupt_grammar._start = Nonterminal(self.config.start)

            corrupt_corpus = self._generate_corpus(corrupt_grammar)
        else:
            corrupt_corpus = self.config.corrupt_grammar(self)

        # merge corrupt corpus with original one + add labels
        new_corpus = [
            (item, label)
            for items, label in [(corrupt_corpus, 0), (corpus, 1)]
            for item in items
        ]

        return new_corpus

    def gen_parse(self, sen: List[str]):
        srp = ChartParser(self.grammar)

        for parse in srp.parse(sen):
            print(parse)

        return next(srp.parse(sen))

    def _create_corrupt_item(self, item: Tensor) -> Tensor:
        raise NotImplementedError

    def gen_baselines(self, *args, **kwargs):
        raise NotImplementedError


def swap_characters(pcfg: PCFG):
    """ Randomly swaps two symbols in the string """
    chart_parser = Parser(pcfg.grammar)
    add_cls = pcfg.tokenizer.config.add_cls

    corrupt_corpus = []

    for item in pcfg.corpus:
        parses = [None]
        new_item = None

        while len(parses) > 0:
            i, j = sorted(random.sample(range(int(add_cls), len(item)), 2))
            a, x, b, y, c = (
                item[:i],
                item[[i]],
                item[i + 1 : j],
                item[[j]],
                item[j + 1 :],
            )
            new_item = torch.cat((a, y, b, x, c))
            new_sen = pcfg.tokenizer.translate(new_item).split()
            if add_cls:
                # Omit CLS token for parse check
                new_sen = new_sen[1:]
            parses = list(chart_parser.parse(new_sen))

        corrupt_corpus.append(new_item)

    return corrupt_corpus


def generate_pcfg(grammar, start=None, depth=None, n=None):
    if not start:
        start = grammar.start()
    if depth is None:
        depth = 1_000

    iterator = _generate_all_pcfg(grammar, [start], depth)

    if n:
        iterator = itertools.islice(iterator, n)

    return iterator


def _generate_all_pcfg(grammar, items, depth):
    if items:
        for frag1 in _generate_one_pcfg(grammar, items[0], depth):
            for frag2 in _generate_all_pcfg(grammar, items[1:], depth):
                yield frag1 + frag2
    else:
        yield []


def _generate_one_pcfg(grammar, lhs, depth):
    if depth > 0:
        if isinstance(lhs, Nonterminal):
            productions = grammar.productions(lhs=lhs)
            probs = grammar._lhs_prob_index[lhs]

            for prod in random.choices(productions, probs, k=1):
                yield from _generate_all_pcfg(grammar, prod.rhs(), depth - 1)
        else:
            yield [lhs]
    else:
        yield []


def generate_tree(grammar, start=None, depth=None, max_tries=10) -> nltk.Tree:
    if not start:
        start = grammar.start()
    if depth is None:
        depth = 100

    for _ in range(max_tries):
        try:
            tree_str = concatenate_subtrees(grammar, [start], depth)
            return nltk.Tree.fromstring(tree_str)
        except RecursionError:
            pass

    raise ValueError("No tree could be generated with current depth")


def concatenate_subtrees(grammar, items, depth):
    if items:
        children = []
        for item in items:
            children.append(generate_subtree(grammar, item, depth))

        return " ".join(children)
    else:
        return []


def generate_subtree(grammar, lhs, depth):
    if depth > 0:
        if isinstance(lhs, Nonterminal):
            productions = grammar.productions(lhs=lhs)
            probs = grammar._lhs_prob_index[lhs]

            for prod in random.choices(productions, probs, k=1):
                children = concatenate_subtrees(grammar, prod.rhs(), depth - 1)
                return f"({lhs.symbol()} {children})"
        else:
            return lhs
    else:
        raise RecursionError


def cfg_str(prod):
    return Production.__str__(prod)


def rev_prod(prod):
    return Production(prod.lhs(), prod.rhs()[::-1])


def swap_subtrees(language: PCFG):
    """ Randomly swaps two subtrees in the parse tree """
    chart_parser = Parser(language.grammar)
    cfg_productions = set(cfg_str(prod) for prod in language.grammar.productions())

    corrupt_corpus = []
    sep_token = language.tokenizer.config.sep_token

    for tree in language.tree_corpus.values():
        corrupted_tree = swap_subtree(tree.copy(), chart_parser, cfg_productions)
        corrupt_corpus.append(sep_token.join(corrupted_tree.leaves()))

    return corrupt_corpus


def swap_subtree(tree: nltk.Tree, parser: ChartParser, cfg_productions) -> nltk.Tree:
    binary_prods = [
        prod
        for prod in tree.productions()
        if len(prod.rhs()) == 2 and cfg_str(rev_prod(prod)) not in cfg_productions
    ]

    corrupted_prod = random.choice(binary_prods)
    for subtree in tree.subtrees():
        if subtree.productions()[0] == corrupted_prod:
            subtree.reverse()
            break

    try:
        # Recursively try again if corrupted sentence yields a parse tree
        next(parser.parse(tree.leaves()))
        return swap_subtree(tree, parser, cfg_productions)
    except StopIteration:
        return tree
