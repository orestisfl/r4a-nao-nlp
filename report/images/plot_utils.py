import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("pgf")

# TODO: reconfirm \columnwidth{}
def figure(figsize=(6.291 - 0.024,), pos=111, *args, **kwargs):
    if len(figsize) == 1:
        width = figsize[0]
        figsize = (width, width / 1.618)

    fig = plt.figure(figsize=figsize, *args, **kwargs)
    ax = fig.add_subplot(pos)
    return fig, ax


# The following functions can be used by scripts to get the sizes of the various elements
# of the figures.


def subplots_adjust(fig, *args, adjust=None, **kwargs):
    adjust = adjust or {}
    adjust.update(kwargs)
    adjust.setdefault("left", 0.07)
    adjust.setdefault("bottom", 0.1)
    adjust.setdefault("right", 0.99)
    adjust.setdefault("top", 0.99)
    fig.subplots_adjust(*args, **adjust)


def label_size():
    """Size of axis labels
    """
    return 10


def font_size():
    """Size of all texts shown in plots
    """
    return 10


def ticks_size():
    """Size of axes' ticks
    """
    return 8


def axis_lw():
    """Line width of the axes
    """
    return 0.6


def plot_lw():
    """Line width of the plotted curves
    """
    return 1.5


def figure_setup():
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {
        "text.usetex": True,
        "pgf.texsystem": "xelatex",
        "pgf.rcfonts": False,
        "pgf.preamble": [
            # put LaTeX preamble declarations here
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            # macros defined here will be available in plots, e.g.:
            r"\newcommand{\text}[1]{#1}",
            # You can use dummy implementations, since you LaTeX document
            # will render these properly, anyway.
        ],
        "figure.dpi": 200,
        "font.size": font_size(),
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": label_size(),
        "axes.titlesize": font_size(),
        "axes.linewidth": axis_lw(),
        "lines.linewidth": plot_lw(),
        "legend.fontsize": font_size(),
        "xtick.labelsize": ticks_size(),
        "ytick.labelsize": ticks_size(),
        "font.family": "serif",
    }
    plt.rcParams.update(params)
