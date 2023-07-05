from fidameval.languages import Palindrome, PalindromeConfig
from fidameval.train import (
    Experiment,
    ExperimentConfig,
    LanguageClassifier,
    ModelConfig,
)

if __name__ == "__main__":
    encoder = "lstm"

    palin_config = PalindromeConfig(
        is_binary=True,
        corpus_size=5_000,
        corrupt_k=(3,),
        sen_len=(2, 3, 5, 6),
        n_items=10,
        use_separator=False,
        map_homomorphic=True,
    )

    language = Palindrome(palin_config)

    model_config = ModelConfig(
        nhid=10,
        num_layers=1,
        vocab_size=language.num_symbols,
        is_binary=True,
        encoder=encoder,
        num_heads=1,
        one_hot_embedding=False,
        emb_dim=10,
        learned_pos_embedding=True,
    )
    model = LanguageClassifier(model_config)

    experiment_config = ExperimentConfig(
        lr=1e-2,
        batch_size=48,
        epochs=1000,
        verbose=True,
        continue_after_optimum=100,
    )

    experiment = Experiment(
        model,
        experiment_config,
    )

    experiment.train(language)
