from starry._c_ops import Ops
import pickle
import numpy as np


def test_pickle():
    ops = Ops(5, 0, 0)
    data = pickle.dumps(ops, -1)
    new_ops = pickle.loads(data)
    assert np.allclose(ops.rTA1, new_ops.rTA1)
