try:
    from . import lightcurve
    from . import theano_op
    from .theano_op import StarryOp
    from .lightcurve import LightCurve
except ImportError:
    lightcurve = None
    theano_op = None
    StarryOp = None
    LightCurve = None