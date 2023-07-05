from fidameval.languages import Dyck, DyckConfig
from fidameval.train import (
    Experiment,
    ExperimentConfig,
    LanguageClassifier,
    ModelConfig,
)

if __name__ == "__main__":
    encoder = "lstm"

    dyck_config = DyckConfig(
        is_binary=True,
        min_length=4,
        max_length=20,
        max_depth=4,
        n_items=2,
        corpus_size=20_000,
        corrupt_k=(1, 2, 3),
    )

    language = Dyck(dyck_config)

    model_config = ModelConfig(
        nhid=10,
        num_layers=1,
        vocab_size=language.num_symbols,
        is_binary=True,
        encoder="lstm",
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

    performances = experiment.train(language)

    print(performances)
