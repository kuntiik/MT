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
from torch import Tensor, tensor

from torchmetrics.functional.regression.mse import _mean_squared_error_compute, _mean_squared_error_update
from torchmetrics.metric import Metric


class MeanSquaredError(Metric):
    r"""
    Computes `mean squared error`_ (MSE):

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        squared:
            If True returns MSE value, if False returns RMSE value.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MeanSquaredError
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> mean_squared_error = MeanSquaredError()
        >>> mean_squared_error(preds, target)
        tensor(0.8750)

    """
    is_differentiable = True
    higher_is_better = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = True,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, n_obs = _mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)
