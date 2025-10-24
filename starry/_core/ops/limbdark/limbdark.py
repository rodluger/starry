# -*- coding: utf-8 -*-

__all__ = ["LimbDarkOp"]

from ....compat import Apply, pytensor, pt
from .base_op import LimbDarkBaseOp


class LimbDarkOp(LimbDarkBaseOp):

    __props__ = ()
    func_file = "./limbdark.cc"
    func_name = "APPLY_SPECIFIC(limbdark)"

    def make_node(self, c, b, r, los):
        in_args = []
        dtype = pytensor.config.floatX
        for a in [c, b, r, los]:
            try:
                a = pt.as_tensor_variable(a)
            except pt.AsTensorError:
                pass
            else:
                dtype = pytensor.scalar.upcast(dtype, a.dtype)
            in_args.append(a)

        out_args = [
            in_args[1].type(),
            pt.TensorType(
                dtype=dtype, broadcastable=[False] * (in_args[1].ndim + 1)
            )(),
            in_args[1].type(),
            in_args[2].type(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return (
            shapes[1],
            list(shapes[0]) + list(shapes[1]),
            shapes[1],
            shapes[2],
        )

    def grad(self, inputs, gradients):
        c, b, r, los = inputs
        f, dfdcl, dfdb, dfdr = self(*inputs)
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, pytensor.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        bc = pt.sum(
            pt.reshape(bf, (1, bf.size))
            * pt.reshape(dfdcl, (c.size, bf.size)),
            axis=-1,
        )
        bb = bf * dfdb
        br = bf * dfdr
        return bc, bb, br, pt.zeros_like(los)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
