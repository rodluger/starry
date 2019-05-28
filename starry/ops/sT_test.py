# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt
from .sT import sTOp


class TestsTOp(utt.InferShapeTester):

    def setUp(self):
        super(TestsTOp, self).setUp()
        self.op_class = sTOp
        self.op = sTOp(2)

    def get_args(self):
        np.random.seed(1234)
        b_val = np.random.uniform(0.0, 1.1, 100)
        r_val = 0.1
        bsT_val = np.ones((100, 9))

        b = tt.vector()
        b.tag.test_value = b_val
        r = tt.scalar()
        r.tag.test_value = r_val
        bsT = tt.matrix()
        bsT.tag.test_value = bsT_val

        op = self.op(b, r, bsT)
        sT = op[0]
        bb = op[1]
        br = op[2]
        f = theano.function([b, r, bsT], sT)

        return f, [b, r, bsT], [b_val, r_val, bsT_val]

    def test_basic(self):
        f, args, arg_vals = self.get_args()
        assert f(*arg_vals).shape == (len(arg_vals[0]), self.op.N)

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args,
                                self.op(*args),
                                arg_vals,
                                self.op_class)
    
    #def test_grad(self):
    #    _, _, in_args = self.get_args()
    #    func = lambda *args: self.op(*args)[0]  # NOQA
    #    utt.verify_grad(func, in_args)