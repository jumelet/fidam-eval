"""
A module for explaining the output of gradient-based
models using path attributions.
"""
__version__ = "1.0"

from .explainers.embedding_explainer_tf import EmbeddingExplainerTF
from .explainers.embedding_explainer_torch import EmbeddingExplainerTorch
from .explainers.path_explainer_tf import PathExplainerTF
from .explainers.path_explainer_torch import PathExplainerTorch
from .plot.scatter import scatter_plot
from .plot.summary import summary_plot
from .plot.text import bar_interaction_plot, matrix_interaction_plot, text_plot
from .utils import set_up_environment, softplus_activation
