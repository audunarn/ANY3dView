"""Bounded topology windows shared by desktop frontends."""
from itertools import islice
from typing import Iterable

TREE_ENTITY_ROW_LIMIT = 2000

def bounded_entity_ids(
    collection: Iterable[object],
    query: str = "",
    *,
    limit: int = TREE_ENTITY_ROW_LIMIT,
) -> list[int]:
    """Return only the entity IDs that the virtual model tree needs.

    Normal refreshes consume at most ``limit`` values.  A numeric search uses
    mapping membership directly, so jumping to an entity in a 50k-owner model
    does not first allocate or scan a 50k-item list.  The helper is deliberately
    Tk-free so the scalability contract remains enforceable in headless CI.
    """

    if limit < 0:
        raise ValueError("tree row limit cannot be negative")
    normalized = str(query).strip().casefold()
    digits = "".join(character for character in normalized if character.isdigit())
    if digits:
        wanted = int(digits)
        try:
            present = wanted in collection  # type: ignore[operator]
        except (TypeError, AttributeError):
            present = any(int(identifier) == wanted for identifier in collection)
        return [wanted] if present else []
    return [int(value) for value in islice(iter(collection), limit)]


def bounded_unowned_entity_ids(
    collection: Iterable[object],
    owned_ids: set[int] | frozenset[int],
    query: str = "",
    *,
    limit: int = TREE_ENTITY_ROW_LIMIT,
) -> list[int]:
    """Return a bounded window of independently authored entity IDs.

    Generated topology is normally collapsed below its feature. Filtering it
    while iterating prevents the tree from materializing every remaining ID,
    and ensures generated IDs at the beginning of a large mapping do not hide
    later user-authored entities from the virtualized window.
    """

    if limit < 0:
        raise ValueError("tree row limit cannot be negative")
    normalized = str(query).strip().casefold()
    digits = "".join(character for character in normalized if character.isdigit())
    if digits:
        wanted = int(digits)
        if wanted in owned_ids:
            return []
        try:
            present = wanted in collection  # type: ignore[operator]
        except (TypeError, AttributeError):
            present = any(int(identifier) == wanted for identifier in collection)
        return [wanted] if present else []
    return [
        int(value)
        for value in islice(
            (identifier for identifier in collection if int(identifier) not in owned_ids),
            limit,
        )
    ]
