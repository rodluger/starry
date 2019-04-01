# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from starry import Map
from starry.ops import DefaultYlmOp
import pytest


def starry_op(lmax, y, theta, xo, yo, zo, ro):
    args = [tt.as_tensor_variable(y),
            tt.as_tensor_variable(theta),
            tt.as_tensor_variable(xo),
            tt.as_tensor_variable(yo),
            tt.as_tensor_variable(zo),
            tt.as_tensor_variable(ro)]
    op = DefaultYlmOp(lmax)
    return op(*args)

@pytest.mark.xfail
def test_specific():
    lmax = 2
    y = tt.dvector(name="y")
    theta = tt.dvector(name="theta")
    xo = tt.dvector(name="xo")
    yo = tt.dvector(name="yo")
    zo = tt.dvector(name="zo")
    ro = tt.dvector(name="ro")
    lc = starry_op(lmax, y, theta, xo, yo, zo, ro)

    args = {
        y: np.array([1.0, 0.1, 0.05, 0.01, 0.3, -0.1, 0.02, 0.3, -0.1]),
        theta: np.zeros(100),
        xo: np.linspace(-1.5, 1.5, 100),
        yo: np.zeros(100),
        zo: np.ones(100),
        ro: 0.1 * np.ones(100)
    }

    var, val = zip(*args.items())

    func = theano.function(var, lc)
    grad = theano.function(var, theano.grad(tt.sum(lc), var))

    lc_val = func(*val)
    grad_val = grad(*val)

    map = Map(lmax=lmax)
    map[:, :] = args[y]
    starry_flux, starry_grad = map.flux(theta=args[theta], xo=args[xo], 
        yo=args[yo], zo=args[zo], ro=args[ro], gradient=True)
    
    assert np.allclose(lc_val, starry_flux)

    for i, v in enumerate(var):
        if v.name in ["theta", "xo", "yo"]:
            assert np.allclose(starry_grad[v.name], grad_val[i]), v.name
        elif v.name == "zo":
            pass
        else:
            assert np.allclose(np.sum(starry_grad[v.name], axis=-1), 
                grad_val[i]), v.name


if __name__ == "__main__":
    test_specific()