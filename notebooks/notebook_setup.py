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
