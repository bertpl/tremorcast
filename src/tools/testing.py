def is_sorted(x: list) -> bool:
    # true if list is sorted
    return all(a <= b for a, b in zip(x, x[1:]))


def all_values_unique(x: list) -> bool:
    # true if all values in list are unique; elements need to be hashable.
    return len(set(x)) == len(x)
