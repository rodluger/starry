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

        b = tt.vector()
        b.tag.test_value = b_val
        r = tt.scalar()
        r.tag.test_value = r_val
        sT = self.op(b, r)
        f = theano.function([b, r], sT)

        return f, [b, r], [b_val, r_val]

    def test_basic(self):
        f, args, arg_vals = self.get_args()
        assert f(*arg_vals).shape == (len(arg_vals[0]), self.op.N)

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args,
                                [self.op(*args)],
                                arg_vals,
                                self.op_class)