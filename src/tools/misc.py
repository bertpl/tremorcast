from typing import Any, Iterable, List


# =================================================================================================
#  Sorting
# =================================================================================================
def sort_any(values: Iterable[Any]) -> List[Any]:
    """Sort iterable containing any values; even unsortable ones (in which case we str() it)."""
    try:
        return sorted(values)
    except TypeError as e:
        return sorted(values, key=lambda x: str(x))
