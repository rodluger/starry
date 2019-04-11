import numpy as np


def DfDveq(f, A, B, C, alpha, veq):
    return f / veq


def DfDalpha(f, A, B, C, alpha, veq):
    if alpha == 0:
        return np.array([
                0,
                np.sqrt(3) * B * (-A ** 2 - B ** 2 - C ** 2) / 15,
                0,
                np.sqrt(3) * A * (-A ** 2 - B ** 2 - C ** 2) / 15,
                0,
                0,
                0,
                0,
                0,
                np.sqrt(70) * B * (3 * A ** 2 - B ** 2) / 70,
                2 * np.sqrt(105) * C * (-A ** 2 + B ** 2) / 105,
                np.sqrt(42) * B * (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
                0,
                np.sqrt(42) * A * (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
                4 * np.sqrt(105) * A * B * C / 105,
                np.sqrt(70) * A * (A ** 2 - 3 * B ** 2) / 70
            ]) * veq
    else:
        DfDalpha = f / alpha
        DfDalpha[1] = -B * (A ** 2 + B ** 2 + C ** 2) / (5 * np.sqrt(3)) * veq
        DfDalpha[3] = -A * (A ** 2 + B ** 2 + C ** 2) / (5 * np.sqrt(3)) * veq
    return DfDalpha


def DfDA(f, A, B, C, alpha, veq):
    if alpha == 0:
        DfDA = np.zeros_like(f)
        DfDA[3] = veq / np.sqrt(3)
        return DfDA
    else:
        return np.array([
            0,
            -2 * A * B / (5 * np.sqrt(3)),
            0,
            (-3 * A ** 2 - B ** 2 - C ** 2 + 5 / alpha) / (5 * np.sqrt(3)),
            0,
            0,
            0,
            0,
            0,
            3 * np.sqrt(2 / 35) * A * B,
            -4 * A * C / np.sqrt(105),
            np.sqrt(2 / 21) / 5 * A * B,
            0,
            (3 * A ** 2 + B ** 2 - 4 * C ** 2) / (5 * np.sqrt(42)),
            4 * B * C / np.sqrt(105),
            3 * (A - B) * (A + B) / np.sqrt(70)
        ]) * veq * alpha


def DfDB(f, A, B, C, alpha, veq):
    if alpha == 0:
        DfDB = np.zeros_like(f)
        DfDB[1] = veq / np.sqrt(3)
        return DfDB
    else:
        return np.array([
            0,
            (-A ** 2 - 3 * B ** 2 - C ** 2 + 5 / alpha) / (5 * np.sqrt(3)),
            0,
            -2 * A * B / (5 * np.sqrt(3)),
            0,
            0,
            0,
            0,
            0,
            3 * (A - B) * (A + B) / np.sqrt(70),
            4 * B * C / np.sqrt(105),
            (A ** 2 + 3 * B ** 2 - 4 * C ** 2) / (5 * np.sqrt(42)),
            0,
            np.sqrt(2 / 21) / 5 * A * B,
            4 * A * C / np.sqrt(105),
            -3 * np.sqrt(2 / 35) * A * B
        ]) * veq * alpha


def DfDC(f, A, B, C, alpha, veq):
    if alpha == 0:
        return np.zeros_like(f)
    else:
        return np.array([
            0,
            -2 * B * C / (5 * np.sqrt(3)),
            0,
            -2 * A * C / (5 * np.sqrt(3)),
            0,
            0,
            0,
            0,
            0,
            0,
            2 * (-A ** 2 + B ** 2) / np.sqrt(105),
            -4 / 5 * np.sqrt(2 / 21) * B * C,
            0,
            -4 / 5 * np.sqrt(2 / 21) * A * C,
            4 * A * B / np.sqrt(105),
            0
        ]) * veq * alpha