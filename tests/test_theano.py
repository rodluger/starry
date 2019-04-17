# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from starry import DopplerMap
from starry.ops import DopplerMapOp
import pytest


def mini_op(y, u, inc, obl, veq, alpha, theta, xo, yo, zo, ro):
    args = [tt.as_tensor_variable(y),
            tt.as_tensor_variable(u),
            tt.as_tensor_variable(inc),
            tt.as_tensor_variable(obl),
            tt.as_tensor_variable(veq),
            tt.as_tensor_variable(alpha),
            tt.as_tensor_variable(theta),
            tt.as_tensor_variable(xo),
            tt.as_tensor_variable(yo),
            tt.as_tensor_variable(zo),
            tt.as_tensor_variable(ro)]
    op = DopplerMapOp()
    return op(*args)


def test_specific():
    # All our parameters
    y = tt.dvector(name="y")
    u = tt.dvector(name="u")
    inc = tt.dscalar(name="inc")
    obl = tt.dscalar(name="obl")
    veq = tt.dscalar(name="veq")
    alpha = tt.dscalar(name="alpha")
    theta = tt.dvector(name="theta")
    xo = tt.dvector(name="xo")
    yo = tt.dvector(name="yo")
    zo = tt.dvector(name="zo")
    ro = tt.dvector(name="ro")
    lc = mini_op(y, u, inc, obl, veq, alpha, theta, xo, yo, zo, ro)

    # Test arguments
    args = {
        y: [],
        u: [],
        inc: 90.,
        obl: 0.,
        veq: 1.,
        alpha: 0.,
        theta: np.zeros(10),
        xo: np.linspace(0.1, 1.5, 10),
        yo: np.zeros(10),
        zo: np.ones(10),
        ro: 0.1 * np.ones(10)
    }

    # Compute using theano
    var, val = zip(*args.items())
    func = theano.function(var, lc)
    lc_val = func(*val)
    grad = theano.function(var, theano.grad(tt.sum(lc), var))
    grad_val = grad(*val)

    # Compute the light curve directly using `DopplerMap`
    map = DopplerMap()
    map.inc = args[inc]
    map.obl = args[obl]
    map.veq = args[veq]
    map.alpha = args[alpha]
    starry_flux, starry_grad = map.rv(theta=args[theta], xo=args[xo], 
        yo=args[yo], zo=args[zo], ro=args[ro], gradient=True)
    
    # Check that the values match
    assert np.allclose(lc_val, starry_flux)

    # Check that the gradients match
    for i, v in enumerate(var):
        if v.name in ["theta", "xo", "yo", "ro"]:
            assert np.allclose(starry_grad[v.name], grad_val[i]), v.name
        elif v.name == "zo":
            pass
        else:
            assert np.allclose(np.sum(starry_grad[v.name], axis=-1), 
                grad_val[i]), v.name
