"""Generate the tables of square roots and factorials in `tables.h`."""
import numpy as np
from scipy.misc import factorial, factorial2


def hfp(x):
    if x > 1:
        if (x % 2) == 0:
            return factorial(x // 2, exact=True)
        else:
            return hfp(x - 2) * 0.5 * x
    elif x == 1:
        return 0.5 * np.sqrt(np.pi)
    elif x == 0:
        return 1
    else:
        raise ValueError()


def hfn(x):
    if x > 1:
        if (x % 2) == 0:
            return np.inf
        else:
            return hfn(x - 2) / (1 - 0.5 * x)
    elif x == 1:
        return np.sqrt(np.pi)
    elif x == 0:
        return 1
    else:
        raise ValueError()


def generate(func, name, maxval, vals_per_row=3):
    n = 0
    table = ""
    for i in range(maxval + 1):
        val = func(i)
        table += "%25.17e" % val
        table += ", "
        n += 1
        if n == vals_per_row:
            n = 0
            table += "\n"
    table = "double %s[] = \n{%s};" % (name, table[1:-2])
    table = table.replace("               inf", "          INFINITY")
    return table


table = [None for i in range(6)]
table[0] = generate(np.sqrt, 'sqrt_int', 300)
table[1] = generate(lambda x: 1 / np.sqrt(x), 'invsqrt_int', 300)
table[2] = generate(lambda x: factorial(x, exact=True), 'factorial', 170)
table[3] = generate(lambda x: factorial2(x, exact=True),
                    'double_factorial', 300)
table[4] = generate(hfp, 'half_factorial_pos', 340)
table[5] = generate(hfn, 'half_factorial_neg', 340)

print("\n\n".join(table))
