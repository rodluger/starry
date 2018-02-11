"""Rotation of spherical harmonics, adapted from Alverez Collado+ (1989)."""
import numpy as np
from scipy.linalg import block_diag


__all__ = ["R"]


class MATRIX(object):
    """Matrix with indexing equal to that in Alverez Collado et al. (1989)."""

    def __init__(self, lmax):
        """Initialize."""
        self.lmax = lmax
        self._matrix = np.zeros((2 * lmax + 1, 2 * lmax + 1, lmax + 1),
                                dtype=float)

    def __setitem__(self, index, value):
        """Set an item in the matrix."""
        a, b, c = index
        self._matrix[self.lmax + a, self.lmax + b, c] = value

    def __getitem__(self, index):
        """Get an item from the matrix."""
        a, b, c = index
        return self._matrix[self.lmax + a, self.lmax + b, c]

    @property
    def matrix(self):
        """Return the full block diagonal matrix."""
        R = []
        for order in range(self.lmax + 1):
            offset = self.lmax - order
            if offset > 0:
                R.append(self._matrix[offset:-offset, offset:-offset, order])
            else:
                R.append(self._matrix[:, :, order])
        return block_diag(*R)


def root(x):
    """Return the square root of x."""
    return np.sqrt(x)


def rooti(x):
    """Return the inverse of the square root of x."""
    return 1 / np.sqrt(x)


def rotar(lmax, c1, s1, c2, s2, c3, s3):
    """
    Return the rotation matrix.

    THIS SUBROUTINE YIELDS THE ROTATION MATRICES R(L;M',M) THAT ARE
    NECESSARY TO PERFORM A COORDINATE TRANSFORMATION USED TO ALIGN
    2. SETS OF REAL SPHERICAL HARMONICS CENTERED AT DIFFERENT POINTS
    (A AND B).

    THIS TRANSFORMATION CONVERTS EACH ORIGINAL REAL SPHERICAL HARMONIC
    IN A LINEAR COMBINATION OF THE REAL SPHERICAL HARMONICS WITH THE
    SAME L AND DIFFERENT M.

    THE MAXIMUM VALUE FOR THE ORBITAL QUANTUM NUMBER L IS 12, TO EXTEND
    THIS PROGRAM TO GREATER VALUES OF L IT IS NECESSARY TO EXTEND THE
    COMMON SQROOT (NEEDED IN THE SUBROUTINE DLMN) WITH THE VALUES OF
    THE SQUARE ROOTS OF THE FIRST 2*lmax+1 INTEGERS AND THEIR
    RECIPROCALS.
    """
    # Initialize the matrices
    assert lmax > 0, "Parameter lmax must be greater than zero."
    RL = MATRIX(lmax)
    DL = MATRIX(lmax)

    # COMPUTATION OF THE INITIAL MATRICES D0, R0, D1 AND R1
    DL[0, 0, 0] = 1.
    RL[0, 0, 0] = 1.
    DL[1, 1, 1] = (1. + c2) / 2.
    DL[1, 0, 1] = -s2 / np.sqrt(2)
    DL[1, -1, 1] = (1. - c2) / 2.
    DL[0, 1, 1] = -DL[1, 0, 1]
    DL[0, 0, 1] = DL[1, 1, 1] - DL[1, -1, 1]
    DL[0, -1, 1] = DL[1, 0, 1]
    DL[-1, 1, 1] = DL[1, -1, 1]
    DL[-1, 0, 1] = DL[0, 1, 1]
    DL[-1, -1, 1] = DL[1, 1, 1]
    COSAG = c1 * c3 - s1 * s3
    COSAMG = c1 * c3 + s1 * s3
    SINAG = s1 * c3 + c1 * s3
    SINAMG = s1 * c3 - c1 * s3
    RL[0, 0, 1] = DL[0, 0, 1]
    RL[1, 0, 1] = np.sqrt(2) * DL[0, 1, 1] * c1
    RL[-1, 0, 1] = np.sqrt(2) * DL[0, 1, 1] * s1
    RL[0, 1, 1] = np.sqrt(2) * DL[1, 0, 1] * c3
    RL[0, -1, 1] = -np.sqrt(2) * DL[1, 0, 1] * s3
    RL[1, 1, 1] = DL[1, 1, 1] * COSAG - DL[1, -1, 1] * COSAMG
    RL[1, -1, 1] = -DL[1, 1, 1] * SINAG - DL[1, -1, 1] * SINAMG
    RL[-1, 1, 1] = DL[1, 1, 1] * SINAG - DL[1, -1, 1] * SINAMG
    RL[-1, -1, 1] = DL[1, 1, 1] * COSAG + DL[1, -1, 1] * COSAMG

    # THE REMAINING MATRICES ARE CALCULATED USING SYMMETRY AND
    # RECURRENCE RELATIONS BY MEANS OF THE SUBROUTINE DLMN.
    if np.abs(s2) < 1.e-15:
        TGBET2 = 0.
    else:
        TGBET2 = (1. - c2) / s2

    for L in range(2, lmax + 1):
        L1 = L
        dlmn(L1, s1, c1, c2, TGBET2, s3, c3, DL, RL)

    return RL


def dlmn(L, s1, c1, c2, TGBET2, s3, c3, DL, RL):
    """
    Compute the matrices DL.

    THIS SUBROUTINE GENERATES THE MATRICES DL[L;M',M) FOR A FIXED VALUE
    OF THE ORBITAL QUANTUM NUMBER L, AND IT NEEDS THE DL[L-2;M',M) AND
    DL[L-1;M',M) MATRICES. THIS SUBROUTINE USES SYMMETRY AND RECURRENCE
    RELATIONS. THE MATRICES DL[L;M',M) ARE THE ROTATION MATRICES FOR
    COMPLEX SPHERICAL HARMONICS.
    """
    IINF = 1 - L
    ISUP = -IINF

    # COMPUTATION OF THE DL[L;M',M) MATRIX, MP IS M' AND M IS M.
    # FIRST ROW BY RECURRENCE (SEE EQUATIONS 19 AND 20)
    DL[L, L, L] = DL[ISUP, ISUP, L - 1] * (1. + c2) / 2.
    DL[L, -L, L] = DL[ISUP, -ISUP, L - 1] * (1. - c2) / 2.
    for M in range(ISUP, IINF - 1, -1):
        DL[L, M, L] = -TGBET2 * root(L + M + 1) * \
                        rooti(L - M) * DL[L, M + 1, L]

    # THE ROWS OF THE UPPER QUARTER TRIANGLE OF THE DL[L;M',M) MATRIX
    # (SEE EQUATION 21)
    AL = L
    AL1 = AL - 1.
    TAL1 = AL + AL1
    ALI = 1. / AL1
    COSAUX = c2 * AL * AL1
    for MP in range(L - 1, -1, -1):
        AMP = MP
        LAUX = L + MP
        LBUX = L - MP

        AUX = rooti(LAUX) * rooti(LBUX) * ALI
        CUX = root(LAUX - 1) * root(LBUX - 1) * AL

        for M in range(ISUP, IINF - 1, -1):
            AM = M
            LAUZ = L + M
            LBUZ = L - M
            AUZ = rooti(LAUZ) * rooti(LBUZ)
            FACT = AUX * AUZ
            TERM = TAL1 * (COSAUX - AM * AMP) * DL[MP, M, L - 1]
            if (LBUZ != 1) and (LBUX != 1):
                CUZ = root(LAUZ - 1) * root(LBUZ - 1)
                TERM = TERM - DL[MP, M, L - 2] * CUX * CUZ
            DL[MP, M, L] = FACT * TERM
        IINF = IINF + 1
        ISUP = ISUP - 1

    # THE REMAINING ELEMENTS OF THE DL[L;M',M) MATRIX ARE CALCULATED
    # USING THE CORRESPONDING SYMMETRY RELATIONS:
    # REFLEXION ---> ((-1)**(M-M')) DL[L;M,M') = DL[L;M',M), M'<=M
    # INVERSION ---> ((-1)**(M-M')) DL[L;-M',-M) = DL[L;M',M)

    # REFLEXION
    SIGN = 1.
    IINF = -L
    ISUP = L - 1
    for M in range(L, 0, -1):
        for MP in range(IINF, ISUP + 1):
            DL[MP, M, L] = SIGN * DL[M, MP, L]
            SIGN = -SIGN
        IINF = IINF + 1
        ISUP = ISUP - 1

    # INVERSION
    IINF = -L
    ISUP = IINF

    for M in range(L - 1, -L - 1, -1):
        SIGN = -1.
        for MP in range(ISUP, IINF - 1, -1):
            DL[MP, M, L] = SIGN * DL[-MP, -M, L]
            SIGN = -SIGN
        ISUP = ISUP + 1

    # COMPUTATION OF THE ROTATION MATRICES RL[L;M',M) FOR REAL SPHERICAL
    # HARMONICS USING THE MATRICES DL[L;M',M) FOR COMPLEX SPHERICAL
    # HARMONICS (SEE EQUATIONS 10 TO 18)
    RL[0, 0, L] = DL[0, 0, L]
    COSMAL = c1
    SINMAL = s1
    SIGN = - 1.
    for MP in range(1, L + 1):
        COSMGA = c3
        SINMGA = s3
        AUX = np.sqrt(2) * DL[0, MP, L]
        RL[MP, 0, L] = AUX * COSMAL
        RL[-MP, 0, L] = AUX * SINMAL
        for M in range(1, L + 1):
            AUX = np.sqrt(2) * DL[M, 0, L]
            RL[0, M, L] = AUX * COSMGA
            RL[0, -M, L] = -AUX * SINMGA
            D1 = DL[-MP, -M, L]
            D2 = SIGN * DL[MP, -M, L]
            COSAG = COSMAL * COSMGA - SINMAL * SINMGA
            COSAGM = COSMAL * COSMGA + SINMAL * SINMGA
            SINAG = SINMAL * COSMGA + COSMAL * SINMGA
            SINAGM = SINMAL * COSMGA - COSMAL * SINMGA
            RL[MP, M, L] = D1 * COSAG + D2 * COSAGM
            RL[MP, -M, L] = -D1 * SINAG + D2 * SINAGM
            RL[-MP, M, L] = D1 * SINAG + D2 * SINAGM
            RL[-MP, -M, L] = D1 * COSAG - D2 * COSAGM
            AUX = COSMGA * c3 - SINMGA * s3
            SINMGA = SINMGA * c3 + COSMGA * s3
            COSMGA = AUX

        SIGN = - SIGN
        AUX = COSMAL * c1 - SINMAL * s1
        SINMAL = SINMAL * c1 + COSMAL * s1
        COSMAL = AUX


def R(lmax, u, theta, tol=1e-15):
    """Return the full rotation matrix for a given spherical harmonic order."""
    # Construct the axis-angle rotation matrix R_A
    ux, uy, uz = u
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    RA = np.zeros((3, 3))
    RA[0, 0] = costheta + ux ** 2 * (1 - costheta)
    RA[0, 1] = ux * uy * (1 - costheta) - uz * sintheta
    RA[0, 2] = ux * uz * (1 - costheta) + uy * sintheta
    RA[1, 0] = uy * ux * (1 - costheta) + uz * sintheta
    RA[1, 1] = costheta + uy ** 2 * (1 - costheta)
    RA[1, 2] = uy * uz * (1 - costheta) - ux * sintheta
    RA[2, 0] = uz * ux * (1 - costheta) - uy * sintheta
    RA[2, 1] = uz * uy * (1 - costheta) + ux * sintheta
    RA[2, 2] = costheta + uz ** 2 * (1 - costheta)

    # Determine the Euler angles
    if (RA[2, 2] < -1 + tol) and (RA[2, 2] > -1 - tol):
        cosbeta = -1
        sinbeta = 0
        cosgamma = RA[1, 1]
        singamma = RA[0, 1]
        cosalpha = 1
        sinalpha = 0
    elif (RA[2, 2] < 1 + tol) and (RA[2, 2] > 1 - tol):
        cosbeta = 1
        sinbeta = 0
        cosgamma = RA[1, 1]
        singamma = -RA[0, 1]
        cosalpha = 1
        sinalpha = 0
    else:
        cosbeta = RA[2, 2]
        sinbeta = np.sqrt(1 - cosbeta ** 2)
        cosgamma = -RA[2, 0] / np.sqrt(RA[2, 0] ** 2 + RA[2, 1] ** 2)
        singamma = RA[2, 1] / np.sqrt(RA[2, 0] ** 2 + RA[2, 1] ** 2)
        cosalpha = RA[0, 2] / np.sqrt(RA[0, 2] ** 2 + RA[1, 2] ** 2)
        sinalpha = RA[1, 2] / np.sqrt(RA[0, 2] ** 2 + RA[1, 2] ** 2)

    '''
    # We can verify that the rotation matrix for the Euler
    # angles,
    #
    # R_E = R_alpha . R_beta . R_gamma
    #
    # is identical:
    RE = np.zeros((3, 3))
    RE[0, 0] = cosalpha * cosbeta * cosgamma - sinalpha * singamma
    RE[0, 1] = -cosgamma * sinalpha - cosalpha * cosbeta * singamma
    RE[0, 2] = cosalpha * sinbeta
    RE[1, 0] = cosbeta * cosgamma * sinalpha + cosalpha * singamma
    RE[1, 1] = cosalpha * cosgamma - cosbeta * sinalpha * singamma
    RE[1, 2] = sinalpha * sinbeta
    RE[2, 0] = -cosgamma * sinbeta
    RE[2, 1] = sinbeta * singamma
    RE[2, 2] = cosbeta
    '''

    mat = rotar(lmax, cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma)
    return mat.matrix
