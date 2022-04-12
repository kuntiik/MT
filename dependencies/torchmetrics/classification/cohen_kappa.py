# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification.cohen_kappa import _cohen_kappa_compute, _cohen_kappa_update
from torchmetrics.metric import Metric


class CohenKappa(Metric):
    r"""
    Calculates `Cohen's kappa score`_ that measures
    inter-annotator agreement. It is defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    Works with binary, multiclass, and multilabel data.  Accepts probabilities from a model output or
    integer class values in prediction.  Works with multi-dimensional preds and target.

    Forward accepts
        - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes

        - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.

        weights: Weighting type to calculate the score. Choose from
            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import CohenKappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> cohenkappa = CohenKappa(num_classes=2)
        >>> cohenkappa(preds, target)
        tensor(0.5000)

    """
    is_differentiable = False
    higher_is_better = True
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        weights: Optional[str] = None,
        threshold: float = 0.5,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.num_classes = num_classes
        self.weights = weights
        self.threshold = threshold

        allowed_weights = ("linear", "quadratic", "none", None)
        if self.weights not in allowed_weights:
            raise ValueError(f"Argument weights needs to one of the following: {allowed_weights}")

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _cohen_kappa_update(preds, target, self.num_classes, self.threshold)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computes cohen kappa score."""
        return _cohen_kappa_compute(self.confmat, self.weights)
