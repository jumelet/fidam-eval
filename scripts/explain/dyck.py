import torch

from fidameval.explain.baselines import DyckBaseline
from fidameval.explain.evaluate import evaluate_all_configuration, gen_dyck_deps
from fidameval.languages import Dyck, DyckConfig
from fidameval.train import LanguageClassifier

if __name__ == "__main__":
    model: LanguageClassifier = torch.load("good_dyck_lstm.pt")
    model.is_binary = True
    model.binary_decoder = model.decoder
    model.zero_idx = 4

    dyck_config = DyckConfig(
        is_binary=True,
        min_length=4,
        max_length=20,
        max_depth=4,
        n_items=2,
        corpus_size=10_000,
    )

    language = Dyck(dyck_config)

    evaluate_all_configuration(
        language,
        model,
        gen_dyck_deps,
        baseline_class=DyckBaseline,
        evaluation_corpus_size=25,
    )
