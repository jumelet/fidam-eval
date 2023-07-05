import sys

sys.path.append("../../archipelago/src")
import itertools

import torch
from application_utils.text_utils import *
from explainer import Archipelago


class ModelWrapperTorch:
    def __init__(self, model):
        self.model = model

    def get_predictions(self, batch_embs):
        batch_embs = torch.tensor(batch_embs)
        if batch_embs.ndim == 2:
            batch_embs = batch_embs.unsqueeze(0)

        return self.model(inputs_embeds=batch_embs).unsqueeze(1)

    def __call__(self, batch_embs):
        batch_predictions = self.get_predictions(batch_embs)

        return batch_predictions.detach().numpy()


def archipelago(model, input_embs, baseline_embs, top_k=1, output_indices=0):
    model_wrapper = ModelWrapperTorch(model)

    np_input_embs = input_embs.detach().numpy()[0]
    np_baseline_embs = baseline_embs.detach().numpy()[0]

    xf = TextXformer(np_input_embs, np_baseline_embs)

    apgo = Archipelago(
        model_wrapper, data_xformer=xf, output_indices=output_indices, batch_size=50
    )

    explanation = apgo.explain(top_k=top_k)

    num_features = np_input_embs.shape[0]
    explanation_matrix = torch.zeros((num_features, num_features))

    for ids, value in explanation.items():
        if len(ids) > 1:
            for i, j in itertools.combinations(ids, 2):
                explanation_matrix[i, j] = value.item()
        else:
            explanation_matrix[ids[0], ids[0]] = value.item()

    return explanation_matrix
