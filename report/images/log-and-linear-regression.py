import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

import plot_utils as pu

N_SAMPLES_1 = 100


def main():
    pu.figure_setup()

    # Toy dataset: it's just a straight line with some Gaussian noise:
    np.random.seed(0)
    x = np.random.normal(size=N_SAMPLES_1)
    y = (x > 0).astype(np.float)
    x[x > 0] *= 4
    x += 0.3 * np.random.normal(size=N_SAMPLES_1)
    x = x[:, np.newaxis]

    fig, ax = pu.figure(pos=121)
    pu.subplots_adjust(fig)
    plot(ax, x, y)

    x = np.array(
        [float(idx) for idx in range(10)]
        + [
            10.5,
            10.7,
            10.85,
            10.95,
            11.5,
            12.1,
            12.5,
            12.6,
            12.65,
            13.33,
            33.3,
            35.5,
            36.6,
        ]
    )
    y = (x > 11).astype(np.float)
    x = x[:, np.newaxis]

    ax = fig.add_subplot(122)
    plot(ax, x, y)

    plt.savefig(
        os.path.join("images", os.path.splitext(os.path.basename(__file__))[0] + ".pgf")
    )
    plt.close()


def plot(ax, x, y):
    # Fit the classifiers
    clf = linear_model.LogisticRegression(C=1e5, solver="lbfgs")
    clf.fit(x, y)

    ols = linear_model.LinearRegression()
    ols.fit(x, y)

    print("LogisticRegression:", sum(np.equal(clf.predict(x) > 0.5, y)) / len(y))
    print("LinearRegression:", sum(np.equal(ols.predict(x) > 0.5, y)) / len(y))

    ax.plot(x, y, "o", markersize=pu.plot_lw() * 2, zorder=20, label="Original data")

    x_ = x_linspace(x)

    loss = model(x_ * clf.coef_ + clf.intercept_).ravel()

    ax.plot(x_, loss, color="red", label="Logistic Regression Model")
    ax.plot(x_, ols.coef_ * x_ + ols.intercept_, label="Linear Regression Model")
    ax.axhline(0.5, color=".5")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    # ax.set_xticks()
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.25, 1.25)
    plt.xlim(min(x_), max(x_))
    ax.legend(prop={"size": 6}, loc="lower right")


def x_linspace(x, num=300):
    start = min(x)
    start -= max(0.1, abs(start) * 0.03)
    end = max(x)
    end += max(0.1, abs(end) * 0.03)
    return np.linspace(start, end, num)


def model(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    main()
