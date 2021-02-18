# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as tt
from starry._core.ops.limbdark.get_cl import GetClOp
from starry._core.ops.limbdark.get_cl_rev import GetClRevOp


class InferShapeTester:
    def setup_method(self):
        # Take into account any mode that may be defined in a child class
        # and it can be None
        mode = getattr(self, "mode", None)
        if mode is None:
            mode = theano.compile.get_default_mode()
        # This mode seems to be the minimal one including the shape_i
        # optimizations, if we don't want to enumerate them explicitly.
        self.mode = mode.including("canonicalize")

    def _compile_and_check(
        self,
        inputs,
        outputs,
        numeric_inputs,
        cls,
        excluding=None,
        warn=True,
        check_topo=True,
    ):
        mode = self.mode
        if excluding:
            mode = mode.excluding(*excluding)
        if warn:
            for var, inp in zip(inputs, numeric_inputs):
                if isinstance(inp, (int, float, list, tuple)):
                    inp = var.type.filter(inp)
                if not hasattr(inp, "shape"):
                    continue
                # remove broadcasted dims as it is sure they can't be
                # changed to prevent the same dim problem.
                if hasattr(var.type, "broadcastable"):
                    shp = [
                        inp.shape[i]
                        for i in range(inp.ndim)
                        if not var.type.broadcastable[i]
                    ]
                else:
                    shp = inp.shape
                if len(set(shp)) != len(shp):
                    break

        outputs_function = theano.function(inputs, outputs, mode=mode)
        shapes_function = theano.function(
            inputs, [o.shape for o in outputs], mode=mode
        )
        # theano.printing.debugprint(shapes_function)
        # Check that the Op is removed from the compiled function.
        if check_topo:
            topo_shape = shapes_function.maker.fgraph.toposort()
            assert not any(isinstance(t.op, cls) for t in topo_shape)
        topo_out = outputs_function.maker.fgraph.toposort()
        assert any(isinstance(t.op, cls) for t in topo_out)
        # Check that the shape produced agrees with the actual shape.
        numeric_outputs = outputs_function(*numeric_inputs)
        numeric_shapes = shapes_function(*numeric_inputs)
        for out, shape in zip(numeric_outputs, numeric_shapes):
            assert np.all(out.shape == shape), (out.shape, shape)


class TestGetCl(InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = GetClOp
        self.op = GetClOp()

    def test_basic(self):
        x = tt.dvector()
        f = theano.function([x], self.op(x))

        inp = np.array([-1, 0.3, 0.2, 0.5])
        out = f(inp)

        assert np.allclose(np.array([-0.85, 2.5, -0.425, 0.1]), out)

    def test_infer_shape(self):
        x = tt.dvector()
        self._compile_and_check(
            [x], [self.op(x)], [np.asarray(np.random.rand(5))], self.op_class
        )

    def test_grad(self):
        tt.verify_grad(self.op, [np.array([-1, 0.3, 0.2, 0.5])], rng=np.random)


class TestGetClRev(InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = GetClRevOp
        self.op = GetClRevOp()

    def test_basic(self):
        x = tt.dvector()
        f = theano.function([x], self.op(x))

        inp = np.array([-1, 0.3, 0.2, 0.5])
        out = f(inp)

        assert np.allclose(np.array([0, 1.3, 2.05, 3.53]), out)

    def test_infer_shape(self):
        x = tt.dvector()
        self._compile_and_check(
            [x], [self.op(x)], [np.asarray(np.random.rand(5))], self.op_class
        )
