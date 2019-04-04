import os
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

import plot_utils as pu


def x_linspace(x):
    start = min(x)
    start -= abs(start) * 0.03
    end = max(x)
    end += abs(start) * 0.03
    return np.linspace(start, end, 20)


# Generate some data:
np.random.seed(0)
x = np.random.random(10)
y = 1.6 * x + np.random.random(10)

# Perform the linear regression:
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Plot the data along with the fitted line:
pu.figure_setup()
fig, ax = pu.figure()
pu.subplots_adjust(fig)

x_ = x_linspace(x)

plt.plot(x, y, "o", label="Original data")
plt.plot(x_, intercept + slope * x_, "r", label="Linear Regression Model")
plt.xlim(x_[0], x_[-1])
plt.ylabel("$y$")
plt.xlabel("$x$")
plt.legend()
plt.savefig(
    os.path.join("images", os.path.splitext(os.path.basename(__file__))[0] + ".pgf")
)
plt.close()
