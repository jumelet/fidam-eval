import math
import random
from copy import deepcopy
from typing import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import *

from fidameval.languages import Dyck, Language, Palindrome
from fidameval.utils import Config

from .model import LanguageClassifier


class ExperimentConfig(Config):
    lr: float = 1e-2
    batch_size: int = 48
    epochs: int = 10
    verbose: bool = False
    early_stopping: Optional[int] = None
    continue_after_optimum: int = 0
    eval_every: int = 100
    warmup_duration: int = 0


class ExitException(Exception):
    pass


class Experiment:
    def __init__(self, model: LanguageClassifier, config: ExperimentConfig) -> None:
        self.model = model
        self.config = config
        self.best_model = None

    def save(self, filename):
        torch.save(self, filename)
        print("Saved experiment to", filename)

    def train(self, languages: Union[Language, List[Language]]):
        if not isinstance(languages, list):
            languages = [languages]

        performances: Dict[int, Dict[str, Any]] = {}

        for lang_idx, language in enumerate(languages):
            performances[lang_idx] = self._train_language(language)

            self.model = self.best_model

        return performances

    def _train_language(self, language: Language) -> Dict[str, Any]:
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_duration,
        )
        if language.config.real_output:
            loss_function = nn.MSELoss()
        elif language.config.is_binary:
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()

        performance_scores = {
            "train": [],
            "dev": [],
            "test": None,
        }

        best_dev_acc = -math.inf
        batches_seen = 0
        batches_seen_at_best_dev_acc = 0
        batch_optimum = 0

        if self.config.verbose:
            iterator = tqdm_notebook(range(self.config.epochs))
        else:
            iterator = range(self.config.epochs)

        try:
            for _epoch in iterator:
                random.shuffle(language.train_corpus)

                for batch in language.batchify(
                    corpus=language.train_corpus,
                    batch_size=self.config.batch_size,
                ):
                    self._update_model(batch, optimizer, loss_function)

                    batches_seen += 1

                    if batches_seen % self.config.eval_every == 0:
                        best_dev_acc, batches_seen_at_best_dev_acc = self._eval(
                            batches_seen,
                            language,
                            performance_scores,
                            best_dev_acc,
                            batches_seen_at_best_dev_acc,
                        )

                        time_since_improvement = (
                            batches_seen - batches_seen_at_best_dev_acc
                        )
                        if (
                            self.config.early_stopping is not None
                            and time_since_improvement > self.config.early_stopping
                        ):
                            # Stop if no increases have been registered for past X epochs
                            if self.config.verbose:
                                print("Stopping early...")
                            raise ExitException
                        elif best_dev_acc == 1.0 and batch_optimum == 0:
                            if self.config.verbose:
                                print(f"Optimum reached at iteration {batches_seen}")
                            batch_optimum = batches_seen

                        if best_dev_acc == 1.0 and (
                            (batch_optimum + self.config.continue_after_optimum)
                            == batches_seen
                        ):
                            raise ExitException

                        scheduler.step()

        except (KeyboardInterrupt, ExitException):
            pass

        if len(language.test_corpus) > 0:
            performance_scores["test"] = self.eval_corpus(
                language, language.test_corpus, model=self.best_model
            )

        return performance_scores

    def _update_model(self, batch, optimizer, loss_function) -> None:
        input_ids, targets, lengths, mask_ids = batch

        self.model.zero_grad()

        predictions = self.model(
            input_ids=input_ids, input_lengths=lengths, mask_ids=mask_ids
        )

        loss = loss_function(predictions, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

        optimizer.step()

    def _eval(
        self,
        batches_seen,
        language,
        performance_scores,
        best_dev_acc,
        batches_seen_at_best_dev_acc,
    ):
        train_acc = self.eval_corpus(language, language.train_corpus)
        performance_scores["train"].append(train_acc)

        dev_acc = self.eval_corpus(language, language.dev_corpus)
        performance_scores["dev"].append(dev_acc)

        if dev_acc > best_dev_acc:
            self.best_model = deepcopy(self.model)
            best_dev_acc = dev_acc
            batches_seen_at_best_dev_acc = batches_seen
            if self.config.verbose:
                print(
                    f"New best at iteration {batches_seen}, dev acc: {dev_acc:.4f}, "
                    f"train acc: {train_acc:.4f}"
                )

        return best_dev_acc, batches_seen_at_best_dev_acc

    def eval_corpus(self, language, corpus, model=None):
        model = model or self.model

        model.eval()

        correct = 0

        batch_size = int(1e9 / model.num_parameters)

        for input_ids, targets, lengths, mask_ids in language.batchify(
            corpus=corpus, batch_size=batch_size
        ):
            with torch.no_grad():
                raw_predictions = model(
                    input_ids=input_ids, input_lengths=lengths, mask_ids=mask_ids
                )

            if language.config.real_output:
                loss_function = nn.MSELoss()
                correct += -loss_function(raw_predictions, targets)
            elif self.model.is_binary:
                predictions = (raw_predictions > 0).to(int)
                correct += int(sum(predictions == targets))
            elif mask_ids is not None:
                ce_loss = F.cross_entropy(raw_predictions, targets)
                perplexity = -ce_loss.exp()

                correct += perplexity * len(mask_ids)
            else:
                split_predictions = torch.split(raw_predictions, tuple(lengths))
                targets = torch.split(targets, tuple(lengths))

                for raw_prediction, target, length in zip(
                    split_predictions, targets, lengths
                ):
                    if isinstance(language, Palindrome):
                        prediction = raw_prediction.argmax(-1)
                        sen_half_idx = (
                            length.item() + int(language.config.use_separator)
                        ) // 2

                        correct += int(
                            (prediction[sen_half_idx:] == target[sen_half_idx:])
                            .to(float)
                            .mean()
                        )
                    elif isinstance(language, Dyck):
                        # Only consider output of closing brackets
                        prediction = raw_prediction[
                            :, language.config.n_items :
                        ].argmax(-1)
                        # Adjust indices for the indices used in the target tensor
                        prediction += language.config.n_items
                        closing_brackets = target >= language.config.n_items
                        prediction = prediction[closing_brackets]
                        target = target[closing_brackets]

                        correct += int((prediction == target).to(float).mean())
                    else:
                        ce_loss = F.cross_entropy(raw_prediction, target)
                        perplexity = -ce_loss.exp()

                        correct += perplexity

        model.train()

        performance = correct / len(corpus)

        if isinstance(performance, torch.Tensor):
            performance = performance.item()

        return performance

    def eval_corpora(
        self,
        languages: List[Language],
        model: Optional[LanguageClassifier] = None,
        lang_names: Optional[List[str]] = None,
        indomain_langs: Optional[List[int]] = None,
        xlabel: str = "Language",
        plot: bool = True,
    ):
        model = model or self.model

        accuracies = {}

        for lang_idx, language in enumerate(languages):
            accuracy = self.eval_corpus(language, language.corpus, model=model)

            lang_name = lang_names[lang_idx] if lang_names else repr(language)
            accuracies[lang_name] = accuracy

            if isinstance(accuracy, torch.Tensor):
                accuracy = accuracy.item()
            if self.config.verbose:
                print(f"{lang_name}\t{accuracy:.4f}")

        if plot:
            plot_eval_corpora(accuracies, indomain_langs or [], xlabel)

        return accuracies


def plot_results(train_accs, dev_accs, test_acc, real_output=False):
    plt.plot(train_accs)
    plt.plot(dev_accs)
    if test_acc is not None:
        plt.axhline(test_acc, ls="--", lw=2)
    if not real_output:
        plt.ylim(0, 1)
    plt.title("Performance")
    plt.show()


def plot_eval_corpora(
    accuracies: Dict[str, float], indomain_langs: List[int], xlabel: str
):
    id_color = "#40B0A6"
    ood_color = "#E1BE6A"

    colors = [
        id_color if idx in indomain_langs else ood_color
        for idx in range(len(accuracies))
    ]

    plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    plt.ylabel("Accuracy")
    plt.xlabel(xlabel)
    plt.ylim(-0.02, 1.01)

    legend_elements = [
        plt.Line2D([0], [0], color=id_color, lw=10, label="Trained on (ID)"),
        plt.Line2D([0], [0], lw=10, color=ood_color, label="OOD"),
    ]
    legend = plt.legend(
        handles=legend_elements, bbox_to_anchor=(1.03, 1), loc="upper left"
    )
    frame = legend.get_frame()
    frame.set_facecolor("w")
    frame.set_edgecolor("black")

    plt.show()


def train(model, language, config):
    experiment = Experiment(
        model,
        config,
    )

    return experiment, experiment.train(language)
