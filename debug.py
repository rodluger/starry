import theano
import starry
import theano.tensor as tt


def get_deriv(b, theta, bo, ro, call_forward_pass=True):

    # Instantiate a zero degree map
    map = starry.Map(ydeg=0, reflected=True)

    # Inputs
    _b = tt.dvector()
    _theta = tt.dvector()
    _bo = tt.dvector()
    _ro = tt.dscalar()

    # The function
    sT = theano.function(
        [_b, _theta, _bo, _ro], map.ops.sT(_b, _theta, _bo, _ro)[0, 0]
    )

    # Its derivative wrt b
    dsTdb = theano.function(
        [_b, _theta, _bo, _ro],
        tt.grad(map.ops.sT(_b, _theta, _bo, _ro)[0, 0], _b),
    )

    # Explicitly call the forward pass first?
    if call_forward_pass:
        sT(b, theta, bo, ro)

    # Return the deriv
    return dsTdb(b, theta, bo, ro)


# This works fine
print(get_deriv([0.5], [0.0], [0.4], 0.3, True))

# This fails
print(get_deriv([0.5], [0.0], [0.4], 0.3, False))
