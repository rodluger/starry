try:
    from . import lightcurve
    from . import theano_op
    from .theano_op import TheanoOp
    from .lightcurve import LightCurve
except ImportError:
    lightcurve = None
    theano_op = None
    TheanoOp = None
    LightCurve = None