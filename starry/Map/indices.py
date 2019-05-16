import numpy as np


__all__ = ["get_ylm_inds", "get_ul_inds"]


def get_ylm_inds(ydeg, ls, ms):
    """
    
    """

    # Turn `ls` and `ms` into slices
    if isinstance(ls, (int, np.int)):
        ls = slice(ls, ls + 1)
    if isinstance(ms, (int, np.int)):
        ms = slice(ms, ms + 1)

    if isinstance(ls, slice) and isinstance(ms, slice):

        # List of indices user is accessing
        inds = []

        # Fill in the `None`s
        if ls.start is None:
            ls = slice(0, ls.stop, ls.step)
        if ls.stop is None:
            ls = slice(ls.start, ydeg + 1, ls.step)
        if ls.step is None:
            ls = slice(ls.start, ls.stop, 1)
        if ms.step is None:
            ms = slice(ms.start, ms.stop, 1)

        if ((ls.start < 0) or (ls.start > ydeg)):
            raise ValueError("Invalid value for `l`.")
        
        # Loop through all the Ylms
        for l in range(ls.start, ls.stop, ls.step):
            ms_ = slice(ms.start, ms.stop, ms.step)
            if (ms_.start is None) or (ms_.start < -l):
                ms_ = slice(-l, ms_.stop, ms_.step)
            if (ms_.stop is None) or (ms_.stop > l):
                ms_ = slice(ms_.start, l + 1, ms_.step)
            for m in range(ms_.start, ms_.stop, ms_.step):
                n = l * l + l + m
                if ((n < 0) or (n >= (ydeg + 1) ** 2) or (m > l) or (m < -l)):
                    raise ValueError("Invalid value for `l` and/or `m`.")
                inds.append(n)

        return inds
    
    else:

        # Not a slice, not an int... What is it?
        raise ValueError("Invalid value for `l` and/or `m`.")


def get_ul_inds(udeg, ls):
    """
    
    """

    # Turn `ls` into a slice
    if isinstance(ls, (int, np.int)):
        ls = slice(ls, ls + 1)

    if isinstance(ls, slice):

        # List of indices user is accessing
        inds = []

        # Fill in the `None`s
        if ls.start is None:
            ls = slice(0, ls.stop, ls.step)
        if ls.stop is None:
            ls = slice(ls.start, udeg + 1, ls.step)
        if ls.step is None:
            ls = slice(ls.start, ls.stop, 1)

        if ((ls.start < 0) or (ls.start > udeg)):
            raise ValueError("Invalid value for `l`.")
        
        # Loop through all the `ls`
        for l in range(ls.start, ls.stop, ls.step):
            inds.append(l)

        return inds
    
    else:

        # Not a slice, not an int... What is it?
        raise ValueError("Invalid value for `l`.")