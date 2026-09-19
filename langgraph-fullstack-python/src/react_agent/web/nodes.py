"""Typing helpers for FastHTML nodes.

`python-fasthtml` ships no type stubs, so mypy under `--strict` treats its
node types as `Any`. Aliasing that in one place keeps the rest of the web layer
readable without sprinkling `Any` everywhere.

The `children` helper wraps its items in a `Fragment`. That matters: FastHTML
does **not** escape a plain Python list passed as a single child (it renders the
list's `repr` instead, which both breaks the markup and would let user text
through unescaped), while a `Fragment` renders its children inline and escapes
every string child.
"""

from typing import Any, Iterable

from fasthtml.common import Fragment

FT = Any
"""A FastHTML node (or a nested list/tuple of nodes)."""


def children(*items: FT) -> FT:
    """Drop empty children and render the rest inline.

    Args:
        *items: Candidate children; None and empty strings are removed.

    Returns:
        A `Fragment` holding the children that carry content.
    """
    return Fragment(*[item for item in items if item is not None and item != ""])


def flatten(items: Iterable[Iterable[FT]]) -> list[FT]:
    """Flatten one level of nested child lists.

    Args:
        items: Iterable of child lists.

    Returns:
        A single list of children.
    """
    result: list[FT] = []
    for group in items:
        result.extend(group)
    return result
