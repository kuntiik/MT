__all__ = ["BaseParser", "Parser"]

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

from src.core import ClassMap
from src.core.class_map import IDMap
from src.core.record import RecordType, BaseRecord
from src.data.random_splitter import DataSplitter, RandomSplitter
from src.data.record_collection import RecordCollection
from src.utils import logger
from src.utils.exceptions import AbortParseRecord
from src.utils.ice_utils import pbar


class BaseParser(ABC):
    @abstractmethod
    def parse(
            self, data_splitter: DataSplitter, autofix: bool = True, show_pbar: bool = True
    ) -> List[List[RecordType]]:
        pass


class Parser(BaseParser, ABC):
    """Base class for all parsers, implements the main parsing logic.

    The actual fields to be parsed are defined by the mixins used when
    defining a custom parser. The only required fields for all parsers
    are `image_id` and `image_width_height`.

    # Arguments
        idmap: Maps from filenames to unique ids, pass an `IDMap()` if you need this information.

    # Examples

    Create a parser for image filepaths.
    ```python
    class FilepathParser(Parser, FilepathParserMixin):
        # implement required abstract methods
    ```
    """

    def __init__(
            self,
            template_record,
            class_map: Optional[ClassMap] = None,
            idmap: Optional[IDMap] = None,
    ):
        # self.class_map = class_map or ClassMap()
        # if class_map is None:
        #     self.class_map.unlock()
        self.template_record = template_record
        self.idmap = idmap or IDMap()

    @abstractmethod
    def __iter__(self) -> Any:
        pass

    @abstractmethod
    def parse_fields(self, o, record: BaseRecord, is_new: bool) -> None:
        pass

    def create_record(self) -> BaseRecord:
        return deepcopy(self.template_record)

    def prepare(self, o):
        pass

    def parse_dicted(self, show_pbar: bool = True) -> Dict[int, RecordType]:
        records = RecordCollection(self.create_record)

        for sample in pbar(self, show_pbar):
            try:
                self.prepare(sample)
                record_id = self.record_id(sample)
                record = records.get_by_record_id(record_id)
                self.parse_fields(sample, record=record, is_new=record.is_new)

            except AbortParseRecord as e:
                logger.warning(
                    "Record with record_id: {} was skipped because: {}",
                    record_id,
                    str(e),
                )

        return records

    def _check_path(self, path: Union[str, Path] = None):
        if path is None:
            return False
        if path is not None:
            return Path(path).exists()

    def parse(
            self,
            data_splitter: DataSplitter = None,
            autofix: bool = True,
            show_pbar: bool = True,
    ) -> List[List[BaseRecord]]:
        """Loops through all samples points parsing the required fields.

        # Arguments
            data_splitter: How to split the parsed samples, defaults to a [0.8, 0.2] random split.
            show_pbar: Whether or not to show a progress bar while parsing the samples.
            cache_filepath: Path to save records in pickle format. Defaults to None, e.g.
                            if the user does not specify a path, no saving nor loading happens.

        # Returns
            A list of records for each split defined by `data_splitter`.
        """
        data_splitter = data_splitter or RandomSplitter([0.8, 0.2])
        records = self.parse_dicted(show_pbar=show_pbar)

        if autofix:
            logger.opt(colors=True).info("<blue><bold>Autofixing records</></>")
            records = records.autofix()

        records = records.make_splits(data_splitter)

        return records

