import os
import matplotlib.pyplot as plt
import numpy as np

import plot_utils as pu


def main():
    pu.figure_setup()

    x = np.linspace(-20, 20, 300)

    fig, ax = pu.figure(pos=121)
    pu.subplots_adjust(fig)
    ax.plot(x, sigmoid(x))
    ax.axhline(0.5, color=".5")
    ax.set_ylabel(r"$\sigma(x)$")
    ax.set_xlabel(r"$x$")
    ax.set_yticks([0, 0.5, 1])
    # ax.set_ylim(-0.25, 1.25)

    # Bigger x in .plot() for smoother curve
    x = np.linspace(0, 25, 300)
    den = sum(np.exp(np.linspace(0, 25, 15)))

    ax = fig.add_subplot(122)
    ax.plot(x, np.exp(x) / den)
    ax.set_ylabel(r"$\text{softmax}(\vec{z_i})$")
    ax.set_xlabel(r"$z_i$")
    ax.set_ylim(0, 1)

    plt.savefig(
        os.path.join("images", os.path.splitext(os.path.basename(__file__))[0] + ".pgf")
    )
    plt.close()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def x_linspace(x, num=300):
    start = min(x)
    start -= max(0.1, abs(start) * 0.03)
    end = max(x)
    end += max(0.1, abs(end) * 0.03)
    return np.linspace(start, end, num)


if __name__ == "__main__":
    main()
