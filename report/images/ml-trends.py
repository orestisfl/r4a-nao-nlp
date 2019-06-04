import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

import plot_utils as pu

pu.figure_setup()
fig, ax = pu.figure()
pu.subplots_adjust(fig, left=0.09)

x = []
y = []
with open(os.path.join("images", "mlTimeline.csv")) as f:
    plots = iter(csv.reader(f, delimiter=","))
    headers = next(plots)
    for row in plots:
        x.append(datetime.strptime(row[0], "%Y-%m"))
        y.append(tuple(map(int, row[1:])))

plt.plot_date(x, y, xdate=True, fmt="-")
plt.ylabel("Τάσεις Google")
plt.xlabel("Έτη")
plt.legend(headers[1:])
plt.savefig(
    os.path.join("images", os.path.splitext(os.path.basename(__file__))[0] + ".pgf")
)
plt.close()
