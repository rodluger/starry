# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt

from starry.theano_op import starry_op
from starry.kepler import Primary, Secondary, System


def test_defaults():
    t = tt.dvector()
    lc = starry_op({}, [dict()], t)
    func = theano.function([t], lc)
    grad = theano.function([t], theano.grad(tt.sum(lc), [t]))

    t_val = np.linspace(0, 1.0, 10000)
    lc_val = func(t_val)
    grad_val = grad(t_val)

    primary = Primary()
    second = Secondary()
    system = System(primary, second)
    system.compute(t_val, gradient=True)

    assert np.allclose(lc_val, system.lightcurve)
    assert np.allclose(system.gradient["time"], grad_val)


def test_specific():
    u = tt.dvector(name="A.u")
    y = tt.dvector(name="b.y")

    L = tt.dscalar(name="b.L")
    r = tt.dscalar(name="b.r")
    a = tt.dscalar(name="b.a")
    porb = tt.dscalar(name="b.porb")
    inc = tt.dscalar(name="b.inc")
    prot = tt.dscalar(name="b.prot")
    lambda0 = tt.dscalar(name="b.lambda0")
    ecc = tt.dscalar(name="b.ecc")
    w = tt.dscalar(name="b.w")

    t = tt.dvector(name="time")

    lc = starry_op(
        dict(lmax=2, u=u),
        [dict(lmax=1, L=L, y=y, r=r, a=a, porb=porb,
              inc=inc, prot=prot, lambda0=lambda0,
              ecc=ecc, w=w)], t)

    args = {
        u: np.array([0.25, 0.3]),
        y: np.array([0.1, 0.05, 0.01]),
        L: 0.01,
        r: 0.08,
        a: 27.1234,
        porb: 1.345,
        inc: 89.9,
        prot: 3.27,
        lambda0: 10.2,
        ecc: 0.1234,
        w: 57.0,
        t: np.linspace(0, 4.0, 10000)
    }

    var, val = zip(*args.items())

    func = theano.function(var, lc)
    grad = theano.function(var, theano.grad(tt.sum(lc), var))

    lc_val = func(*val)
    grad_val = grad(*val)

    primary = Primary(lmax=2)
    primary[:] = args[u]

    second = Secondary(lmax=1)
    second[1:, :] = args[y]
    second.L = args[L]
    second.r = args[r]
    second.a = args[a]
    second.porb = args[porb]
    second.prot = args[prot]
    second.inc = args[inc]
    second.ecc = args[ecc]
    second.w = args[w]
    second.lambda0 = args[lambda0]

    system = System(primary, second)
    system.compute(args[t], gradient=True)

    assert np.allclose(lc_val, system.lightcurve)

    starry_grad = system.gradient
    for i, v in enumerate(var):
        if v.name == "time":
            assert np.allclose(starry_grad[v.name], grad_val[i]), "time"
            continue
        assert np.allclose(np.sum(starry_grad[v.name], axis=-1),
                           grad_val[i]), v.name
