__all__ = ["Metric", "CaptureStdout"]

import sys
from abc import ABC, abstractmethod
from typing import Dict
from io import StringIO


class Metric(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def accumulate(self, preds):
        """Accumulate stats for a single batch"""

    @abstractmethod
    def finalize(self) -> Dict[str, float]:
        """Called at the end of the validation loop"""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class CaptureStdout(list):
    """Capture the stdout (like prints)
    From: https://stackoverflow.com/a/16571630/6772672
    """

    def __init__(self, propagate_stdout: bool = False):
        self.propagate_stdout = propagate_stdout

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout

        if self.propagate_stdout:
            print("\n".join(self))
