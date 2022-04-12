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
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.wer import _wer_compute, _wer_update
from torchmetrics.metric import Metric


class WordErrorRate(Metric):
    r"""
    Word error rate (WordErrorRate_) is a common metric of the performance of an automatic speech recognition system.
    This value indicates the percentage of words that were incorrectly predicted.
    The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.
    Word error rate can then be computed as:

    .. math::
        WER = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where:
        - S is the number of substitutions,
        - D is the number of deletions,
        - I is the number of insertions,
        - C is the number of correct words,
        - N is the number of words in the reference (N=S+D+C).

    Compute WER score of transcribed segments against references.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Word error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> metric = WordErrorRate()
        >>> metric(preds, target)
        tensor(0.5000)
    """
    is_differentiable = False
    higher_is_better = False
    error: Tensor
    total: Tensor

    def __init__(
        self,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:  # type: ignore
        """Store references/predictions for computing Word Error Rate scores.

        Args:
            preds: Transcription(s) to score as a string or list of strings
            target: Reference(s) for each speech input as a string or list of strings
        """
        errors, total = _wer_update(preds, target)
        self.errors += errors
        self.total += total

    def compute(self) -> Tensor:
        """Calculate the word error rate.

        Returns:
            Word error rate score
        """
        return _wer_compute(self.errors, self.total)
