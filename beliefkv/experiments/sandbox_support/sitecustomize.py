"""Compatibility aliases needed by older SWE-bench Python repositories."""

import collections
import collections.abc


for _name in (
    "Callable",
    "Container",
    "Hashable",
    "ItemsView",
    "Iterable",
    "Iterator",
    "KeysView",
    "Mapping",
    "MappingView",
    "MutableMapping",
    "MutableSequence",
    "MutableSet",
    "Sequence",
    "Set",
    "Sized",
    "ValuesView",
):
    if _name not in collections.__dict__:
        setattr(collections, _name, getattr(collections.abc, _name))
