# Pretty figures
get_ipython().magic('config InlineBackend.figure_format = "retina"')


import matplotlib
import matplotlib.pyplot as plt
import warnings


# Disable annoying font warnings
matplotlib.font_manager._log.setLevel(50)


# Disable theano deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="theano")


# Style
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["font.size"] = 14
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.fallback_to_cm"] = True


# TODO: In case we need to patch some symbols.
if False:
    try:
        old_get_unicode_index
    except NameError:
        import matplotlib.mathtext as mathtext

        old_get_unicode_index = mathtext.get_unicode_index
        mathtext.get_unicode_index = (
            lambda symbol, math=True: ord("x")
            if symbol == "\\times"
            else old_get_unicode_index(symbol, math)
        )

# Several hacks to `corner` to make it prettier
import corner
import numpy as np


try:
    old_corner
except NameError:
    old_corner = corner.corner


def new_corner(*args, **kwargs):
    # Get the usual corner plot
    figure = old_corner(*args, **kwargs)

    # Get the axes
    ndim = int(np.sqrt(len(figure.axes)))
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Smaller tick labels
    for ax in axes[1:, 0]:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(formatter)
    for ax in axes[-1, :]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.xaxis.set_major_formatter(formatter)

    # Pad the axes to always include the truths
    truths = kwargs.get("truths", None)
    if truths is not None:
        for row in range(1, ndim):
            for col in range(row):
                lo, hi = np.array(axes[row, col].get_xlim())
                if truths[col] < lo:
                    lo = truths[col] - 0.1 * (hi - truths[col])
                    axes[row, col].set_xlim(lo, hi)
                    axes[col, col].set_xlim(lo, hi)
                elif truths[col] > hi:
                    hi = truths[col] - 0.1 * (hi - truths[col])
                    axes[row, col].set_xlim(lo, hi)
                    axes[col, col].set_xlim(lo, hi)

                lo, hi = np.array(axes[row, col].get_ylim())
                if truths[row] < lo:
                    lo = truths[row] - 0.1 * (hi - truths[row])
                    axes[row, col].set_ylim(lo, hi)
                    axes[row, row].set_xlim(lo, hi)
                elif truths[row] > hi:
                    hi = truths[row] - 0.1 * (hi - truths[row])
                    axes[row, col].set_ylim(lo, hi)
                    axes[row, row].set_xlim(lo, hi)

    return figure


corner.corner = new_corner
