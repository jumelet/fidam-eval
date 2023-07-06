import torch
from torch import Tensor

from .path_explain import EmbeddingExplainerTorch


def model_forward(model):
    def forward(batch_embedding: torch.Tensor) -> torch.Tensor:
        return model(inputs_embeds=batch_embedding)

    return forward


def integrated_hessians(
    model: torch.nn.Module,
    input_embs: Tensor,
    baseline_embs: Tensor,
    batch_size: int = 50,
    num_samples: int = 900,
    use_expectation: bool = False,
) -> Tensor:
    """
    Parameters
    ----------
        model : nn.Module
            Torch model.
        input_embs : (batch_size, sen_len, emb_dim)
            Tensor containing the input embeddings.
        baseline_embs : (batch_size, sen_len, emb_dim)
            Tensor containing the baseline embeddings.
        batch_size : int
            Batch size used by IH internally to compute the interaction
            matrix.
        num_samples : int
            Number of interpolation samples, or number of samples
            taken from baseline expectation if `use_expaction` is set
            to True.
        use_expectation : bool
            Set to True to sample baseline embeddings from a
            distribution. If set to True the samples are assumed to be
            part of the baseline_embs tensor, and will be sampled from
            there.

    Returns
    -------
        interactions : (batch_size, sen_len, sen_len)
    """

    explainer = EmbeddingExplainerTorch(model_forward(model))

    interactions = explainer.interactions(
        input_embs.detach(),
        baseline_embs.detach(),
        batch_size=batch_size,
        num_samples=num_samples,
        use_expectation=use_expectation,
    )

    return torch.tensor(interactions)
