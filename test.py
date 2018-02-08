from starry import R, A, S
import numpy as np
import matplotlib.pyplot as pl

lmax = 3
y = np.ones((lmax + 1) ** 2, dtype=float)

# The polynomial term we're integrating
print(np.dot(A(lmax), y))

print(S(lmax, 0.001, 0.5))
quit()


r = 0.5
b = np.linspace(0, 2, 500)

F = [np.dot(np.dot(S(lmax, bi, r).T, A(lmax)), y) for bi in b]

pl.plot(b, F)
pl.show()
