def mlvl_get(d, key, default=None):
    """"mlvl" stands for multi-level.

    Args:
        d: a dict, possibly containing more dicts within.
        key: a string, multiple levels of indexing into d separated by "."
        default: if not found at any level, returns default.
    """
    key_split = key.split('.', maxsplit=1)
    if len(key_split) == 1:
        return d.get(key, default)
    else:
        curr_key, rest = key_split
        if curr_key in d:
            return mlvl_get(d[curr_key], rest, default)
        else:
            return default


def mlvl_set(d, key, val):
    """Similar to mlvl_get, except "setting".

    If intermediate key doesn't exist, create it with value dict(). Throws
    error if exists already.
    """
    key_split = key.split('.', maxsplit=1)
    if len(key_split) == 1:
        if key in d:
            raise Exception('Key already exists')
        else:
            d[key] = val
    else:
        curr_key, rest = key_split
        if curr_key not in d:
            d[curr_key] = dict()

        mlvl_set(d[curr_key], rest, val)


def mlvl_getattr(c, key, default=None):
    """Similar to mlvl_get, except c is a class and key gets attributes."""
    key_split = key.split('.', maxsplit=1)
    if len(key_split) == 1:
        return getattr(c, key)
    else:
        curr_key, rest = key_split
        if hasattr(c, curr_key):
            return mlvl_getattr(getattr(c, curr_key), rest, default)
        else:
            return default