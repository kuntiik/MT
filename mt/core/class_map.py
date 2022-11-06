__all__ = ["ClassMap", "BACKGROUND", "IDMap"]

from copy import copy
from typing import Optional, Sequence, Hashable, List
from collections import OrderedDict

BACKGROUND = "background"


class ClassMap:
    """Utility class for mapping between class name and id."""

    def __init__(
            self,
            classes: Optional[Sequence[str]] = None,
            background: Optional[str] = BACKGROUND,
    ):
        self._lock = True

        self._id2class = copy(list(classes)) if classes else []
        # insert background if required
        self._background = background
        if self._background is not None:
            try:
                self._id2class.remove(self._background)
            except ValueError:
                pass
            # background is always index zero
            self._id2class.insert(0, self._background)

        self._class2id = {name: i for i, name in enumerate(self._id2class)}

    @property
    def num_classes(self):
        return len(self)

    def get_classes(self) -> Sequence[str]:
        return self._id2class

    def get_by_id(self, id: int) -> str:
        return self._id2class[id]

    def get_by_name(self, name: str) -> int:
        try:
            return self._class2id[name]
        except KeyError as e:
            if not self._lock:
                return self.add_name(name)
            else:
                raise e

    def add_name(self, name: str) -> int:
        # Raise error if trying to add duplicate value
        if name in self._id2class:
            raise ValueError(
                f"'{name}' already exists in the ClassMap. You can only add new labels that are unique"
            )

        self._id2class.append(name)
        id = len(self._class2id)
        self._class2id[name] = id
        return id

    def lock(self):
        self._lock = True
        return self

    def unlock(self):
        self._lock = False
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, ClassMap):
            return self.__dict__ == other.__dict__
        return False

    def __len__(self):
        return len(self._id2class)

    def __repr__(self):
        return f"<ClassMap: {self._class2id.__repr__()}>"


class IDMap:
    """
    Works like a dictionary that automatically assign values for new keys.
    """

    def __init__(self, initial_names: Optional[Sequence[Hashable]] = None):
        names = initial_names or []
        self.id2name = OrderedDict([(id, name) for id, name in enumerate(names)])
        self.name2id = OrderedDict([(name, id) for id, name in enumerate(names)])

    def get_id(self, id: int) -> Hashable:
        return self.id2name[id]

    def get_name(self, name: Hashable) -> int:
        try:
            id = self.name2id[name]
        except KeyError:
            id = len(self.name2id)
            self.name2id[name] = id
            self.id2name[id] = name

        return id

    def filter_ids(self, ids: List[int]) -> "IDMap":
        idmap = IDMap()
        for id in ids:
            name = self.get_id(id)
            idmap.id2name[id] = name
            idmap.name2id[name] = id

        return idmap

    def get_ids(self) -> List[int]:
        return list(self.id2name.keys())

    def get_names(self) -> List[Hashable]:
        return list(self.name2id.keys())

    def __getitem__(self, record_id):
        return self.get_name(record_id)
